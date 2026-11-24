"""FSDP OpenDWM observation renderer with commit-one/replan temporal adaptation.

Each native 19-slot clip has three committed RGB references then sixteen copies
of the current CATK pose/actors. Startup repeats history9 in all three reference
slots; the ring subsequently contains only committed generated observations.
Only the first new frame is committed, all other generated frames are discarded.
Reference geometry follows the matching committed state. Holds are synthetic
controls, not known future trajectories. No GT future dataset loader is called.
"""
import copy
import os
import numpy as np
from multi_render_common import initial_inputs, set_condition, publish


class Renderer:
    def __init__(self, cfg):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('OpenDWM GPU inference requires Slurm')
        import torch
        import dwm.common
        from infer_genbench_gt2 import runtime_config
        self.cfg = cfg
        self.n = 0
        self.rank = int(os.environ.get('LOCAL_RANK', '0'))
        world = int(os.environ.get('WORLD_SIZE', '1'))
        if world not in (2, 4) or torch.cuda.device_count() != world:
            raise RuntimeError('OpenDWM requires 2 or 4 visible GPUs and matching torchrun ranks')
        torch.cuda.set_device(self.rank)
        torch.distributed.init_process_group('nccl')
        self.device = torch.device('cuda', self.rank)
        config = runtime_config(cfg['render_config'], cfg['base_model'], cfg['render_checkpoint'], cfg['seed'])
        config['pipeline']['inference_config']['sequence_length'] = 19
        self.pipe = dwm.common.create_instance_from_config(config['pipeline'],
            output_path=None, config=config, device=self.device)
        self.inputs, images = initial_inputs(cfg)
        self.references = [self._tensor_views(images)] * 3
        self.states = None
        from dwm.datasets.genbench_gt2_renderer import prepare_map
        self.map_geometry = prepare_map(self.inputs.map_condition)

    @staticmethod
    def _tensor_views(images):
        import torch
        from PIL import Image
        return torch.stack([torch.from_numpy(np.asarray(im.resize((448,256),
            Image.Resampling.BILINEAR)).copy()).permute(2,0,1).float()/255 for im in images])

    def build_batch(self, state):
        import torch
        from genbench_gt2.constants import IMAGE_SIZE
        from genbench_gt2.geometry import scaled_calibration
        from dwm.datasets.genbench_gt2_renderer import render_camera_conditions
        if self.states is None:
            self.states = [copy.deepcopy(state) for _ in range(3)]
        states = self.states + [state] * 16
        images = torch.zeros((19,6,3,256,448))
        images[:3] = torch.stack(self.references)
        boxes = torch.empty_like(images)
        maps = torch.empty_like(images)
        for frame, condition in enumerate(states):
            set_condition(self.inputs, frame, condition)
            for view, camera in enumerate(self.inputs.camera_order):
                control = render_camera_conditions(self.inputs, frame, camera, (448,256),
                    map_geometry=self.map_geometry)
                for target, source in ((boxes, control.box_rgb), (maps, control.hdmap_rgb)):
                    target[frame,view] = torch.from_numpy(np.asarray(source).copy()).permute(2,0,1).float()/255
        cal = [scaled_calibration(self.inputs.calibrations[c], IMAGE_SIZE, (448,256))
               for c in self.inputs.camera_order]
        mask = torch.eye(6,dtype=torch.bool)
        for view in range(6):
            mask[view,(view-1)%6] = mask[view,(view+1)%6] = True
        item = dict(fps=torch.tensor(2.), pts=torch.arange(19).float()[:,None].repeat(1,6)*500,
            vae_images=images, **{'3dbox_images': boxes}, hdmap_images=maps,
            clip_text=[[self.inputs.prompt]*6 for _ in range(19)],
            camera_transforms=torch.tensor(np.stack([c.camera_to_ego for c in cal]),dtype=torch.float32)[None].repeat(19,1,1,1),
            camera_intrinsics=torch.tensor(np.stack([c.intrinsics for c in cal]),dtype=torch.float32)[None].repeat(19,1,1,1),
            ego_transforms=torch.tensor(self.inputs.ego_to_anchor[:19],dtype=torch.float32)[:,None].repeat(1,6,1,1),
            image_size=torch.tensor([448,256]).repeat(19,6,1), crossview_mask=mask)
        return {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k,v in item.items()}

    def call(self, q):
        import torch
        from PIL import Image
        batch = self.build_batch(q['state'])
        self.pipe.generator.manual_seed(self.cfg['seed'] + self.n)
        with torch.inference_mode():
            output = self.pipe.autoregressive_inference_pipeline((1,
                self.pipe.get_latent_sequence_length(19),6,self.pipe.vae.config.latent_channels,32,56),batch,'pt')
        generated = output['images'].unflatten(0,(16,6))
        if tuple(generated.shape) != (16,6,3,256,448):
            raise ValueError('Unexpected OpenDWM shape: ' + str(generated.shape))
        committed = generated[0].detach().float().cpu().clamp(0,1).clone()
        self.references = self.references[1:] + [committed]
        self.states = self.states[1:] + [copy.deepcopy(q['state'])]
        result = None
        if self.rank == 0:
            images = [Image.fromarray((im.permute(1,2,0).numpy()*255).round().astype(np.uint8))
                      .resize((400,224),Image.Resampling.BILINEAR) for im in committed]
            result = publish(self.inputs,images,q['state']['pose'],q['output'])
            result['conditioning'] = dict(reference='history9_repeated_then_committed_generated_ring',
                reference_slots=3,current_state_hold_slots=16,discarded_generated_slots=15,committed_slot=3)
        shared = [result]
        torch.distributed.broadcast_object_list(shared,src=0)
        self.n += 1
        return shared[0]

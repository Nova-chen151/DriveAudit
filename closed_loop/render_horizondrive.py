"""HorizonDrive receding-horizon renderer; native 21 frames, commit one.

The upstream two-process evaluator partitions scenes, not a single model. A
single persistent pipeline is therefore sufficient for this single-scene RPC.
"""
import sys
from pathlib import Path
import numpy as np
from multi_render_common import initial_inputs, set_condition, publish


class Renderer:
    def __init__(self, cfg):
        import torch
        from horizondrive.schemas.args import Args
        from horizondrive.trainer import WanUnifiedTrainer
        self.cfg = cfg
        self.inputs, images = initial_inputs(cfg)
        self.reference = images
        self.previous_state = None
        self.n = 0
        argv = sys.argv
        try:
            sys.argv = ['closed-loop', '--pretrained_model_name_or_path', cfg['wan_checkpoint'],
                        '--config_path', cfg['render_config'], '--transformer_path', cfg['render_checkpoint'],
                        '--vae_path', cfg['vae_checkpoint'], '--mixed_precision', 'bf16',
                        '--train_mode', 'unified', '--crossview_attn_type', 'full', '--low_vram',
                        '--seed', str(cfg['seed']), '--output_dir', cfg['runtime_dir']]
            args = Args.parse_args()
        finally:
            sys.argv = argv
        args.report_to = None
        self.trainer = WanUnifiedTrainer(args)
        self.trainer.components.transformer3d.eval()
        self.pipe = self.trainer.initialize_pipeline().to(self.trainer.accelerator.device)
        self.dtype = self.trainer.state.weight_dtype

    def call(self, q):
        import torch
        from PIL import Image
        from genbench_gt2.raster import render_camera_conditions
        from horizondrive.datasets.genbench_gt2_dataset import GenBenchGT2Dataset
        # Eleven native conditioning slots use only the last committed image;
        # no speculative frame is retained for the following RPC.
        previous = self.previous_state or q['state']
        set_condition(self.inputs, 9, previous)
        set_condition(self.inputs, 10, q['state'])
        cameras = GenBenchGT2Dataset.model_camera_order
        frames = [9] * 11 + [10] * 10
        h, w = 384, 768
        def tensor(im):
            return torch.from_numpy(np.array(im.resize((w, h)), dtype=np.float32)).permute(2,0,1) / 127.5 - 1
        pixels, boxes, maps = [], [], []
        for cam in cameras:
            reference = tensor(self.reference[self.inputs.camera_order.index(cam)])
            pixels.extend([reference] * 11 + [torch.zeros_like(reference)] * 10)
            for f in frames:
                cond = render_camera_conditions(self.inputs, f, cam, (w, h))
                boxes.append(torch.from_numpy(cond.box_rgb.copy()).permute(2,0,1).float()/127.5-1)
                maps.append(torch.from_numpy(cond.hdmap_rgb.copy()).permute(2,0,1).float()/127.5-1)
        actions = [GenBenchGT2Dataset._action(self.inputs.ego_to_anchor, f, frames[min(i+1,20)]) for i,f in enumerate(frames)]
        mask = torch.ones((6,21,1,h,w), dtype=torch.uint8)
        mask[:,:11] = 0
        with torch.inference_mode(), torch.autocast('cuda', dtype=self.dtype):
            videos = self.pipe(prompt=self.trainer._normalize_validation_prompt(self.inputs.prompt,6),
                video=torch.stack(pixels)[None], mask=mask.reshape(1,126,1,h,w),
                num_frames=21, num_cameras=6, height=h, width=w, guidance_scale=1.,
                generator=torch.Generator(device=self.trainer.accelerator.device).manual_seed(self.cfg['seed']+self.n),
                additional_conditions={'action':torch.stack(actions).repeat(6,1)[None],
                                       'bbox':torch.stack(boxes)[None], 'hdmap':torch.stack(maps)[None]},
                crossview_attn_type='full', step='final', t_compression_ratio=1,
                use_t_variant_noise=True, num_inference_steps=4,
                sigmas=np.array([1000,700,500,50], dtype=np.float32)/1000,
                num_condition_images=11, num_unroll_steps=1, latent_rollout=False).videos
        arr = videos.detach().float().cpu().numpy() if torch.is_tensor(videos) else np.asarray(videos)
        if arr.shape != (1,3,126,h,w) or not np.isfinite(arr).all():
            raise ValueError(f'Invalid HorizonDrive output {arr.shape}')
        by_cam = {cam: Image.fromarray((arr[0,:,i*21+11].transpose(1,2,0).clip(0,1)*255).round().astype('uint8')).resize((400,224),Image.Resampling.LANCZOS) for i,cam in enumerate(cameras)}
        images = [by_cam[c] for c in self.inputs.camera_order]
        result = publish(self.inputs, images, q['state']['pose'], q['output'])
        self.reference = images
        self.previous_state = q['state']
        self.n += 1
        result['temporal_commit'] = {'native_frames':21,'committed_slot':11,'discarded_generated_frames':9,'reference':'history_000009_then_last_committed'}
        return result

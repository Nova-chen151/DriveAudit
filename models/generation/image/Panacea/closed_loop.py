"""Replan an eight-slot Panacea clip on every closed-loop observation.

Chronological slots are previous committed RGB, then seven identical current
CATK states. Native input reverses all slots (known RGB last); only chronological
slot 1 is committed. The other six generated slots are discarded. No predicted
or recorded future trajectory is supplied. This current-state hold is an
explicit adaptation, not the checkpoint's original trajectory-conditioned task.
"""
import copy
import os
from types import SimpleNamespace
import numpy as np
from driveaudit.closed_loop.rendering import initial_inputs, set_condition, publish


class Renderer:
    def __init__(self, cfg):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Panacea GPU inference requires Slurm')
        import torch
        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config
        self.cfg = cfg
        self.n = 0
        self.inputs, _ = initial_inputs(cfg)
        self.condition = None
        self.previous = None
        self.device = torch.device('cuda', 0)
        config = OmegaConf.load(cfg['render_config'])
        config.model.params.vae_decode_batch_size = 1
        for embedder in config.model.params.conditioner_config.params.emb_models:
            if embedder.target == 'sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder':
                embedder.params.version = cfg['clip_checkpoint']
                break
        else:
            raise ValueError('OpenCLIP embedder missing')
        self.model = instantiate_from_config(config.model)
        state = torch.load(cfg['render_checkpoint'], map_location='cpu')
        if 'deepspeed' in cfg['render_checkpoint']:
            state = {k.replace('_forward_module.', ''): v for k, v in state.items()}
        else:
            state = state['state_dict']
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.load_info = dict(missing_keys=list(missing), unexpected_keys=list(unexpected))
        print('Panacea checkpoint:', self.load_info, flush=True)
        del state
        self.model.to(self.device).eval()

    def call(self, q):
        import torch
        from PIL import Image
        from sgm.data.genbench_gt2_dataset import (build_panacea_window,
            prepare_native_panacea_batch, restore_panacea_timeline, split_panacea_views)
        # Slot labels are scratch indices, never recording timestamps.
        set_condition(self.inputs, 9, self.previous or q['state'])
        for frame in range(1, 8):
            set_condition(self.inputs, frame, q['state'])
        window = SimpleNamespace(condition_frame=9, control_frames=tuple(range(1, 8)))
        batch = build_panacea_window(self.inputs, window, self.condition)
        native = prepare_native_panacea_batch(batch, self.device)
        torch.manual_seed(self.cfg['seed'] + self.n)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            samples = self.model.log_images(native, sample_with_muldecoders=False,
                skip_reconstructions=True)['samples'].detach().float().cpu().clamp(-1, 1)
        if tuple(samples.shape) != (8, 3, 256, 3072):
            raise ValueError('Unexpected Panacea samples: ' + str(samples.shape))
        committed = restore_panacea_timeline(samples)[1].clone()
        views = split_panacea_views(committed)
        images = [Image.fromarray(((views[name].permute(1, 2, 0).numpy()+1)*127.5)
                  .round().astype(np.uint8)).resize((400, 224), Image.Resampling.BILINEAR)
                  for name in self.inputs.camera_order]
        result = publish(self.inputs, images, q['state']['pose'], q['output'])
        result.update(conditioning=dict(reference='history9' if self.n == 0 else 'previous_generated',
            current_state_hold_slots=7, discarded_generated_slots=6,
            native_order='reverse_chronology_reference_last', committed_slot=1), model_load=self.load_info)
        self.condition = committed
        self.previous = copy.deepcopy(q['state'])
        self.n += 1
        return result

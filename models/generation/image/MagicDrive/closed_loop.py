"""Original MagicDrive-t layout-only closed-loop adapter (not MagicDriveV2)."""
from pathlib import Path
from types import SimpleNamespace
from driveaudit.closed_loop.rendering import initial_inputs, set_condition, publish

class Renderer:
    def __init__(self, cfg):
        import sys
        # Match the released launcher: pinned temporal package, native bbox
        # extensions and its vendored diffusers must precede installed copies.
        sys.path[:0] = [cfg['video_root'], cfg['bench'] + '/MagicDrive-bevfusion',
                        cfg['video_root'] + '/third_party/diffusers/src']
        import infer_genbench_gt2 as impl
        from genbench_gt2.temporal import magicdrive_plan
        self.impl, self.cfg, self.n = impl, cfg, 0
        self.inputs, _ = initial_inputs(cfg)
        self.pipe, self.model_cfg, self.dtype = impl.load_pipeline(SimpleNamespace(
            checkpoint=Path(cfg['render_checkpoint']), sd15_root=Path(cfg['sd15']),
            video_root=Path(cfg['video_root'])))
        self.length = len(magicdrive_plan().condition_windows[0])
        if self.length != 61:
            raise ValueError('Expected released MagicDrive-t 61-frame window')

    def call(self, q):
        import torch
        from genbench_gt2.constants import CAMERAS
        frame = 10 + self.n
        set_condition(self.inputs, frame, q['state'])
        # This native model has no image-reference conditioning interface.
        # All temporal slots receive the current state. Only slot0 is exposed
        # to the policy; no generated future state is executed or cached.
        frames = [frame] * self.length
        batch = self.impl.build_magicdrive_window(self.inputs, frames)
        with torch.inference_mode():
            output = self.pipe(prompt=batch.prompts,
                image=batch.bev_map.to('cuda', dtype=self.dtype),
                camera_param=batch.camera_param.to('cuda', dtype=self.dtype), height=224, width=400,
                generator=torch.Generator(device='cuda').manual_seed(self.cfg['seed'] + self.n),
                bev_controlnet_kwargs={'bboxes_3d_data': {
                    'bboxes': batch.boxes.to('cuda'), 'classes': batch.classes.to('cuda'),
                    'masks': batch.masks.to('cuda'),
                    'cam_params': batch.camera_param.to('cuda', dtype=self.dtype)}},
                guidance_scale=2, num_inference_steps=self.cfg.get('render_steps', 20),
                eta=0.0, controlnet_conditioning_scale=1.0, guess_mode=False,
                use_zero_map_as_unconditional=False, init_noise='rand_all',
                view_order=list(CAMERAS), double_cfg_inference=True, decode_bs=1, keyframe_rate=6)
        if len(output.images) != self.length:
            raise ValueError('Unexpected MagicDrive clip length')
        result = publish(self.inputs, output.images[0], q['state']['pose'], q['output'])
        result.update(dropped=batch.dropped_categories, conditioning={
            'method': 'native_layout_only_current_state_padded_clip',
            'control_frames': frames, 'history_frames_used': [],
            'reference_images_supported': False, 'published_slot': 0,
            'discarded_slots': self.length - 1, 'used_future_gt': False,
            'used_future_ego_trajectory': False, 'used_future_agent_trajectories': False})
        self.n += 1
        return result

"""DreamForge-t closed-loop adapter: two generated references, current-only controls."""
from pathlib import Path
from types import SimpleNamespace
from driveaudit.closed_loop.rendering import initial_inputs, set_condition, publish

class Renderer:
    def __init__(self, cfg):
        import infer_genbench_gt2 as impl
        import torch
        self.impl, self.cfg, self.n = impl, cfg, 0
        self.inputs, images = initial_inputs(cfg)
        self.pipe, self.model_cfg, self.dtype = impl.load_pipeline(SimpleNamespace(
            checkpoint=Path(cfg['render_checkpoint']), sd15_root=Path(cfg['sd15']),
            single_checkpoint=Path(cfg['single_checkpoint'])))
        self.length = int(self.model_cfg.model.video_length)
        if int(self.model_cfg.model.ref_length) != 2:
            raise ValueError('DreamForge checkpoint must use two reference frames')
        self.references = [9, 9]
        self.ref_values = torch.stack([impl.normalize_views(images)] * 2)

    def call(self, q):
        import torch
        frame = 10 + self.n
        if not self.n:
            set_condition(self.inputs, 9, q['state'])
        set_condition(self.inputs, frame, q['state'])
        # Native temporal attention needs a complete clip. Repeat only the
        # current simulated state; padded slots never contain future GT.
        frames = [frame] * self.length
        batch = self.impl.build_dreamforge_window(self.inputs, frames, self.references)
        params = dict(self.model_cfg.runner.pipeline_param)
        params.update(num_inference_steps=self.cfg.get('render_steps', 20),
                      decode_bs=1, init_noise='rand_all', double_cfg_inference=False)
        with torch.inference_mode():
            output = self.pipe(prompt=[self.inputs.prompt] * self.length,
                image=[batch.bev.to(self.dtype), batch.layout.to(self.dtype)],
                ref_image=self.ref_values.to(self.dtype), can_bus=batch.can_bus.to(self.dtype),
                camera_param=batch.camera.to(self.dtype), height=224, width=400,
                generator=torch.Generator(device='cuda').manual_seed(self.cfg['seed'] + self.n),
                bev_controlnet_kwargs={'bboxes_3d_data': {'bboxes': batch.boxes,
                    'classes': batch.classes, 'masks': batch.masks}}, img_metas=batch.meta, **params)
        if len(output.images) != self.length:
            raise ValueError('Unexpected DreamForge clip length')
        images = output.images[0]
        result = publish(self.inputs, images, q['state']['pose'], q['output'])
        result.update(dropped=batch.dropped, conditioning={
            'reference_frames': list(self.references), 'control_frames': frames,
            'history_frames_used': [9] if self.n < 2 else [],
            'method': 'two_reference_autoregression_current_state_padded_clip',
            'published_slot': 0, 'discarded_slots': self.length - 1,
            'used_future_gt': False, 'used_future_ego_trajectory': False,
            'used_future_agent_trajectories': False})
        self.ref_values = torch.stack([self.ref_values[-1], self.impl.normalize_views(images)])
        self.references = [self.references[-1], frame]
        self.n += 1
        return result

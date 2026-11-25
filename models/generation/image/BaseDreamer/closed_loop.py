"""Render a six-view BaseDreamer observation using the previous committed image."""
from pathlib import Path
from types import SimpleNamespace

from driveaudit.closed_loop.core import pose_matrix
from driveaudit.closed_loop.rendering import initial_inputs, publish, set_condition


class Renderer:
    def __init__(self, cfg):
        import infer_genbench_gt2 as impl

        self.impl = impl
        self.cfg = cfg
        self.n = 0
        self.inputs, images = initial_inputs(cfg)
        self.pipe, self.model_cfg, self.dtype = impl.load_pipeline(SimpleNamespace(
            checkpoint=Path(cfg["render_checkpoint"]), sd15_root=Path(cfg["sd15"])))
        self.reference = impl.image_tensor(images)

    def call(self, request):
        import torch

        frame = 10 + self.n
        state = request["state"]
        if self.n == 0:
            self.inputs.ego_to_anchor[9] = pose_matrix(state["pose"])
        set_condition(self.inputs, frame, state)
        kwargs, dropped = self.impl.build_conditions(self.inputs, frame, frame - 1)
        for key in ("bev_hdmap", "camera_param", "rel_pose", "layout_canvas"):
            kwargs[key] = kwargs[key].to(device="cuda", dtype=self.dtype)
        kwargs["ref_images"] = self.reference.to(device="cuda", dtype=self.dtype)
        with torch.inference_mode(), torch.autocast("cuda"):
            rows = self.pipe(prompt=[self.inputs.prompt], height=224, width=400,
                             generator=torch.Generator(device="cuda").manual_seed(self.cfg["seed"] + self.n),
                             **kwargs, **dict(self.model_cfg.runner.pipeline_param)).images
        images = rows[0]
        result = publish(self.inputs, images, state["pose"], request["output"])
        result["dropped"] = dropped
        self.reference = self.impl.image_tensor(images)
        self.n += 1
        return result

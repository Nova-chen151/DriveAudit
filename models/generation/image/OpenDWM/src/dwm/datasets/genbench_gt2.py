"""Native UniMLVG controls for the immutable GT2 prediction contract."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from genbench_gt2.constants import CAMERAS, IMAGE_SIZE
from genbench_gt2.dataset import load_generation_set, load_scene_inputs
from genbench_gt2.geometry import scaled_calibration
from dwm.datasets.genbench_gt2_renderer import render_camera_conditions, RENDERER_VERSION, prepare_map
from genbench_gt2.selection import select_scenes
from genbench_gt2.temporal import opendwm_plan


MODEL_IMAGE_SIZE = (448, 256)


class GenBenchGT2OpenDWMDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, generation_set="balanced_500", scene_indices=None):
        self.scenes = select_scenes(load_generation_set(Path(data_root), generation_set), False, scene_indices, None)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        return self.build_batch(self.scenes[index])

    def build_batch(self, scene):
        inputs = load_scene_inputs(scene)
        frames = opendwm_plan().sequence_frames
        size = MODEL_IMAGE_SIZE
        map_geometry = prepare_map(inputs.map_condition)
        # Allocate once: avoid a second full 83-frame stack in host memory.
        images = torch.zeros((len(frames), 6, 3, size[1], size[0]), dtype=torch.float32)
        boxes = torch.empty_like(images)
        maps = torch.empty_like(images)
        def tensor(array):
            return torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1).float().div_(255)
        for slot, frame in enumerate(frames):
            for view, camera in enumerate(CAMERAS):
                if frame < 10:
                    with Image.open(inputs.history_rgb[camera][frame]) as image:
                        images[slot, view] = tensor(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
                control = render_camera_conditions(inputs, frame, camera, size, map_geometry=map_geometry)
                boxes[slot, view] = tensor(control.box_rgb)
                maps[slot, view] = tensor(control.hdmap_rgb)
        calibrations = [scaled_calibration(inputs.calibrations[c], IMAGE_SIZE, size) for c in CAMERAS]
        timestamps = (inputs.timeline_us[list(frames)] - inputs.timeline_us[7]) / 1000.0
        fps = float(1e6 / np.median(np.diff(inputs.timeline_us[list(frames)])))
        mask = torch.eye(6, dtype=torch.bool)
        for view in range(6):
            mask[view, (view-1)%6] = mask[view, (view+1)%6] = True
        return {
            "fps": torch.tensor(fps),
            "pts": torch.tensor(timestamps, dtype=torch.float32)[:, None].repeat(1, 6),
            "vae_images": images, "3dbox_images": boxes, "hdmap_images": maps,
            "clip_text": [[inputs.prompt] * 6 for _ in frames],
            "camera_transforms": torch.tensor(np.stack([c.camera_to_ego for c in calibrations]), dtype=torch.float32)[None].repeat(83, 1, 1, 1),
            "camera_intrinsics": torch.tensor(np.stack([c.intrinsics for c in calibrations]), dtype=torch.float32)[None].repeat(83, 1, 1, 1),
            "ego_transforms": torch.tensor(inputs.ego_to_anchor[list(frames)], dtype=torch.float32)[:, None].repeat(1, 6, 1, 1),
            "image_size": torch.tensor(size).repeat(83, 6, 1),
            "crossview_mask": mask,
            "prediction_context": {"condition_renderer": RENDERER_VERSION, "native_map_missing_layers": ["lane"], "conditioning_fps": fps, "pts_unit": "milliseconds", "available_map_channels": list(inputs.map_condition.available_channels), "unavailable_map_channels": list(inputs.map_condition.unavailable_channels)},
        }

"""GenBench-GT2 adapter for HorizonDrive's camera-major validation path."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from genbench_gt2.constants import CAMERAS
from genbench_gt2.dataset import load_generation_set, load_scene_inputs
from genbench_gt2.raster import render_camera_conditions
from genbench_gt2.temporal import horizon_plan
from horizondrive.catk_generation import campaign_runtime, conditioning_provenance, verify_campaign_scene


class GenBenchGT2Dataset(Dataset):
    """Build HorizonDrive's fixed 91-slot input without reading future RGB."""

    camera_names = [
        "camera_front", "camera_front_left", "camera_front_right",
        "camera_rear_right", "camera_rear", "camera_rear_left",
    ]
    model_camera_order = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
    ]

    def __init__(self, data_root, generation_set="balanced_500", scene_indices=None,
                 scene_indices_csv="", limit=0, **_unused):
        scenes = load_generation_set(Path(data_root), generation_set)
        if scene_indices is None and scene_indices_csv:
            scene_indices = [int(value.strip()) for value in scene_indices_csv.split(",") if value.strip()]
        if scene_indices is not None:
            requested = set(scene_indices)
            scenes = [scene for scene in scenes if scene.index in requested]
            if len(scenes) != len(requested):
                raise ValueError("GenBench scene_indices contains an unknown scene")
        if int(limit):
            if int(limit) < 0:
                raise ValueError("GenBench limit must be nonnegative")
            scenes = scenes[:int(limit)]
        self.scenes = scenes
        self.inputs = [load_scene_inputs(scene) for scene in scenes]
        self.conditioning = {}
        self.runtime_provenance = None
        if os.environ.get("GENBENCH_CATK_AGENT_ONLY") == "1":
            if scene_indices is None:
                raise ValueError("CATK agent-only mode requires an explicit selected scene set")
            self.runtime_provenance = campaign_runtime()
            for scene in scenes:
                provenance = conditioning_provenance(scene)
                verify_campaign_scene(scene, provenance)
                self.conditioning[scene.index] = provenance
        self.length = len(self.inputs)

    def __len__(self):
        return self.length

    @staticmethod
    def _image(path, height, width):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, height))
            array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0

    @staticmethod
    def _action(ego_to_anchor, current, following):
        relative = np.linalg.inv(ego_to_anchor[current]) @ ego_to_anchor[following]
        yaw = np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))
        return torch.tensor((relative[0, 3], relative[1, 3], yaw), dtype=torch.float32)

    def __getitem__(self, kwargs):
        if kwargs.get("model_mode") != "i2v":
            raise ValueError("GenBench GT2 HorizonDrive requires i2v mode")
        if int(kwargs.get("infer_image_len", -1)) != 91:
            raise ValueError("GenBench GT2 HorizonDrive requires infer_image_len == 91")
        if int(kwargs.get("num_condition_images", 11)) != 11:
            raise ValueError("GenBench GT2 HorizonDrive requires eleven condition images")
        idx = int(kwargs["idx"])
        inputs = self.inputs[idx]
        _, height, width = kwargs["resolution"]
        # The production config fixes the checkpoint's 384x768 resolution;
        # rasterization and tensor assembly follow the requested spatial size.

        plan = horizon_plan()
        slot_frames = list(plan.condition_frames) + [frame for window in plan.generated_windows for frame in window]
        actions = []
        bbox = []
        hdmap = []
        pixels = []
        zeros = torch.zeros((3, height, width), dtype=torch.float32)
        for camera in self.model_camera_order:
            camera_pixels = [self._image(inputs.history_rgb[camera][frame], height, width) for frame in range(10)]
            camera_pixels.append(camera_pixels[9].clone())
            camera_pixels.extend(zeros.clone() for _ in range(80))
            pixels.extend(camera_pixels)
            for slot, frame in enumerate(slot_frames):
                following = slot_frames[slot + 1] if slot + 1 < len(slot_frames) else frame
                if camera == self.model_camera_order[0]:
                    actions.append(self._action(inputs.ego_to_anchor, frame, following))
                condition = render_camera_conditions(inputs, frame, camera, (width, height))
                bbox.append(torch.from_numpy(condition.box_rgb).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0)
                hdmap.append(torch.from_numpy(condition.hdmap_rgb).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0)

        pixel_values = torch.stack(pixels)
        mask = torch.ones((6, 91, 1, height, width), dtype=torch.uint8)
        mask[:, :11] = 0
        return {
            "pixel_values": pixel_values,
            "text": inputs.prompt,
            "clip_id": "{}__{}".format(inputs.scene.dataset, inputs.scene.scene_id),
            "data_type": "video",
            "idx": idx,
            "conditions": {"action": torch.stack(actions).repeat(6, 1),
                           "bbox": torch.stack(bbox), "hdmap": torch.stack(hdmap)},
            "mask": mask.view(6 * 91, 1, height, width),
            "model_mode": "i2v",
            "prediction_context": {
                "scene": inputs.scene,
                "model_camera_order": self.model_camera_order,
                "target_frames": list(range(10, 90)),
                "history_frames_used": list(range(10)),
                "repeated_reference_slots": [9],
                "map_available": list(inputs.map_condition.available_channels),
                "map_unavailable": list(inputs.map_condition.unavailable_channels),
                **({"conditioning_provenance": self.conditioning[inputs.scene.index],
                    "runtime_provenance": self.runtime_provenance}
                   if inputs.scene.index in self.conditioning else {}),
            },
        }

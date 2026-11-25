#!/usr/bin/env python3
"""Check which OpenDWM condition channels change when future actors are removed."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def check(scene, common_config, frames):
    import numpy as np
    import torch
    from dwm.datasets.genbench_gt2_renderer import prepare_map, render_camera_conditions
    from dwm.pipelines.ctsd import CrossviewTemporalSD
    from genbench_gt2.constants import CAMERAS, IMAGE_SIZE
    from genbench_gt2.dataset import load_scene_inputs
    from genbench_gt2.geometry import scaled_calibration

    inputs = load_scene_inputs(scene)
    size = (448, 256)
    calibrations = [scaled_calibration(inputs.calibrations[camera], IMAGE_SIZE, size) for camera in CAMERAS]
    geometry = prepare_map(inputs.map_condition)

    def tensor(items):
        return torch.tensor(np.asarray(items), dtype=torch.float32).unsqueeze(0)

    def model_condition(model_inputs):
        boxes, maps = [], []
        for frame in frames:
            controls = [render_camera_conditions(model_inputs, frame, camera, size, map_geometry=geometry) for camera in CAMERAS]
            boxes.append([item.box_rgb.transpose(2, 0, 1) for item in controls])
            maps.append([item.hdmap_rgb.transpose(2, 0, 1) for item in controls])
        batch = {
            "3dbox_images": tensor(boxes) / 255,
            "hdmap_images": tensor(maps) / 255,
            "pts": torch.tensor([[(inputs.timeline_us[f] - inputs.timeline_us[7]) / 1000] * 6 for f in frames]).unsqueeze(0),
            "camera_intrinsics": tensor([[calib.intrinsics for calib in calibrations]] * len(frames)),
            "camera_transforms": tensor([[calib.camera_to_ego for calib in calibrations]] * len(frames)),
            "ego_transforms": tensor([[inputs.ego_to_anchor[f]] * 6 for f in frames]),
            "image_size": tensor([[size] * 6] * len(frames)),
        }
        # Isolate native geometry assembly. Text and denoising weights are unnecessary for this tensor test.
        values = CrossviewTemporalSD.get_conditions(
            None, None, None, common_config, [1, len(frames), 6, 16, 32, 56], batch, "cpu", torch.float32
        )
        return values["condition_image_tensor"]

    original = model_condition(inputs)
    empty = model_condition(replace(inputs, agents_by_frame={}))
    changed = original != empty
    box_values = int(changed[:, :, :, :3].sum())
    map_values = int(changed[:, :, :, 3:].sum())
    if map_values or torch.any(empty[:, :, :, :3]) or tuple(original.shape) != (1, len(frames), 6, 6, 256, 448):
        raise ValueError(f"Unexpected native conditions in {scene.scene_id}")
    return dict(scene=scene.scene_id, frames=frames, shape=list(original.shape),
                changed_box_values=box_values, changed_map_values=map_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True, help="Staged DriveAudit conditions")
    parser.add_argument("--engine-code-root", type=Path, required=True, help="Native engine installation")
    parser.add_argument("--frames", nargs="+", type=int, default=[10, 50], choices=range(10, 90), metavar="TARGET_FRAME")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    opendwm = args.engine_code_root / "gen-benchmark/OpenDWM-main"
    sys.path[:0] = [str(root / "driveaudit/generation"), str(opendwm / "src")]
    # Use this checkout's rasterizer with the installed native pipeline.
    import dwm.datasets
    dwm.datasets.__path__ = [str(root / "models/generation/image/OpenDWM/src/dwm/datasets"), *dwm.datasets.__path__]
    from genbench_gt2.dataset import load_generation_set

    config = json.loads((opendwm / "examples/ctsd_unimlvg_6views_video_generation.json").read_text())
    common = config["pipeline"]["common_config"]
    rows = [check(scene, common, args.frames) for scene in load_generation_set(args.conditions, "participant_joint")]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

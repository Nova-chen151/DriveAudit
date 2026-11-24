"""Run a frozen DrivoR policy on each generated or matched recorded observation."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from .runtime import asset_root, model_root


CAMERAS = ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
COMMANDS = {"right": 0, "left": 1, "straight": 2}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_frames(scene):
    timeline = json.loads((scene / "timeline.json").read_text())["frames"]
    times = {frame["frame_index"]: frame["reference_time_us"] for frame in timeline}
    poses = {}
    for file in ("history/ego.csv", "future/ego_trajectory.csv"):
        with (scene / file).open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["valid"].lower() != "true":
                continue
            poses[int(row["frame_index"])] = tuple(float(row[k]) for k in ("x", "y", "yaw"))
    future = [int(row["frame_index"]) for row in rows]
    if not future or future != list(range(future[0], future[-1] + 1)):
        raise ValueError("Expected contiguous public decision frames")
    return future, times, poses


def ego_state(frame, times, poses):
    """Three causal poses; rotate both finite differences into current ego axes."""
    import numpy as np

    times_us = [times[i] for i in range(frame-2, frame+1)]
    if not times_us[0] < times_us[1] < times_us[2]:
        raise ValueError("Non-increasing causal odometry timestamps")
    prev, curr = np.diff(times_us) * 1e-6
    p0, p1, p2 = (np.asarray(poses[i][:2]) for i in range(frame-2, frame+1))
    v0, v1 = (p1-p0)/prev, (p2-p1)/curr
    a = (v1-v0) / ((prev+curr)/2)
    yaw = poses[frame][2]
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    return {"velocity": [*map(float, rotation @ v1), 0.0],
            "acceleration": [*map(float, rotation @ a), 0.0],
            "timestamp": times[frame]}


def model_input(rgb, frame, cameras, state, command, channels=CAMERAS):
    import numpy as np
    from PIL import Image

    images, calibration, digests = {}, {}, {}
    for name in channels:
        path = rgb / name / f"{frame:06d}.jpg"
        resolved = path.resolve(strict=True)
        if rgb.resolve() not in resolved.parents:
            raise ValueError(f"RGB file escapes selected scene: {path}")
        images[name] = str(path)
        digests[name] = hash_file(path)
        info = cameras[name]
        transform = np.asarray(info["camera_to_ego_4x4"], dtype=float)
        K = np.array(info["challenge_intrinsics"], dtype=float)
        with Image.open(path) as image:
            w, h = image.size
        base_w, base_h = info["challenge_image_size"]
        K[0] *= w/base_w
        K[1] *= h/base_h
        # NAVSIM's legacy field name refers to its planning frame. DrivoR consumes
        # camera features in this ego-aligned setting; no physical LiDAR is loaded.
        calibration[name] = {"camera_intrinsics": K.tolist(),
                             "sensor2lidar_rotation": transform[:3, :3].tolist(),
                             "sensor2lidar_translation": transform[:3, 3].tolist()}
    return {"camera_images": images, "camera_calibration": calibration, "ego_state": state,
            "command": {"code": COMMANDS[command], "name": command.upper()}}, digests


def selected_rgb(args):
    if args.source == "generated":
        if args.real_mapping or not args.rgb_scenes:
            raise ValueError("Generated inference requires only --rgb-scenes")
        return {case: args.rgb_scenes / case / "future/rgb" for case in args.cases}
    if args.rgb_scenes or not args.real_mapping:
        raise ValueError("Real inference requires only --real-mapping")
    rows = json.loads(args.real_mapping.read_text())
    mapping = {row["public_scene"]: Path(row["source_path"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Ambiguous private scene mapping")
    return {case: mapping[case] / "future/rgb" for case in args.cases}


def run(args):
    roots = selected_rgb(args)
    code = model_root("DrivoR")
    sys.path.insert(0, str(code))
    # The nuPlan devkit can be installed as a regular package or supplied as a checkout.
    if os.environ.get("NUPLAN_DEVKIT_ROOT"):
        sys.path.insert(1, os.environ["NUPLAN_DEVKIT_ROOT"])
    from .native.drivor import DrivoRAdapter
    checkpoint = args.checkpoint or asset_root("DrivoR") / "weights/drivor_Nav1_25epochs.pth"
    backbone = args.backbone or asset_root("DrivoR") / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
    identity = {"checkpoint_sha256": hash_file(checkpoint), "backbone_sha256": hash_file(backbone),
                "config_sha256": hash_file(code / "navsim/planning/script/config/common/agent/drivoR.yaml")}
    policy = DrivoRAdapter(checkpoint, backbone)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False,
                                     prefix=args.output.name + ".", suffix=".partial") as handle:
        tmp = Path(handle.name)
        try:
            for case, rgb in roots.items():
                scene = args.public_scenes / case
                entries = json.loads((scene / "calibration/cameras.json").read_text())["cameras"]
                cameras = {cam["camera"]: cam for cam in entries}
                future, times, poses = scene_frames(scene)
                for index, frame in enumerate(future):
                    state = ego_state(frame, times, poses)
                    inputs, digests = model_input(rgb, frame, cameras, state, args.command)
                    prediction = policy.predict(inputs)
                    row = {"case_id": case, "policy": "drivor", "source": args.source,
                           "frame_id": index, "frame_index": frame, "reference_time_us": times[frame],
                           "dt_s": 0.5, "coordinate_system": prediction["coordinate_system"],
                           "trajectory_xy": prediction["planning_traj"], "native_trajectory": prediction["native_trajectory"],
                           "native_time_offsets_s": prediction["native_time_offsets"],
                           "mode_index": prediction["mode_index"], "mode_score": prediction["mode_score"],
                           "request_seed": prediction["request_seed"], "strict_state_dict_tensors": policy.tensor_count,
                           "command": args.command, "ego_state": state, "rgb_sha256": digests, **identity}
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                print(f"{args.source} {case}: {len(future)} decisions", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-scenes", type=Path, required=True)
    parser.add_argument("--rgb-scenes", type=Path)
    parser.add_argument("--real-mapping", type=Path)
    parser.add_argument("--source", choices=["generated", "real"], required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", choices=tuple(COMMANDS), default="straight")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or any(case in (".", "..") or Path(case).name != case for case in args.cases):
        parser.error("Expected unique scene directory names")
    run(args)


if __name__ == "__main__":
    main()

"""Build the anonymized Challenger-style player input from a frozen selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from .discover_dataset import discover
from .pipeline import _boxes, _calibration, _finite_differences, _resize
from .schema import CAMERAS, read_yaml, write_csv, write_json


ROOT = Path(__file__).resolve().parents[3]
_MAPS = {}
_WORKER = {}

_README = """# DriveAudit Challenger input

The contest contract is fixed: use 10 historical frames (0-9) to generate future RGB, Agent trajectories,
and Ego trajectory for 80 frames (10-89). Participants choose their own adversarial target and behavior.

Each `scenes/adv_case_XXXX/` contains:

- `history/rgb/<camera>/000000.jpg` ... `000009.jpg`: six 400x224 RGB views.
- `history/agents.csv`: historical Agent boxes and motion in the frame-9 Ego coordinate system.
- `history/ego.csv`: historical Ego poses in the same coordinate system.
- `calibration/cameras.json`: per-rig intrinsics/extrinsics; quaternion order is WXYZ.
- `map/semantic_mask.npy` and `distance_fields.npy`: local 100m x 100m map on a 200 x 200 grid (0.5m/pixel).

Map bit/channel order is: 0 `drivable_area`, 1 `road_segment`, 2 `lane`, 3 `ped_crossing`,
4 `walkway`, 5 `stop_line`, 6 `carpark_area`, 7 `road_divider`, 8 `lane_divider`.

Camera order is `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`.
All scenes use a fixed 12 FPS frame index and the six camera views are treated as synchronized.
Trajectory coordinates are `history_last_ego_x_forward_y_left_meters`; yaw is counter-clockwise radians.

Submit only:

```
scenes/<scene_id>/
  future_rgb/<camera>/000010.jpg ... 000089.jpg
  future_trajectories.csv
  ego_trajectory.csv
```

Each scene must contain 480 future JPEGs. `future_trajectories.csv` columns are
`frame_index,agent_id,category,x,y,z,width,length,height,yaw,valid`.
`ego_trajectory.csv` columns are `frame_index,x,y,yaw,valid`.
`history/agents.csv` columns are
`frame_index,agent_id,category,is_dynamic,x,y,z,width,length,height,yaw,vx,vy,valid`.
`history/ego.csv` columns are `frame_index,x,y,yaw,valid`.
Agent IDs are positive integer indexes. Reuse known indexes; assign new entrants unused indexes.

No future RGB or source nuScenes identifiers are included.
"""


def _angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_local_map(nusc, location: str, pose: dict, target: Path) -> None:
    from nuscenes.map_expansion.map_api import NuScenesMap
    from scipy.ndimage import distance_transform_edt

    api = _MAPS.get(location)
    if api is None:
        api = _MAPS[location] = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
    layers = [name for name in ("drivable_area", "road_segment", "lane", "lane_connector", "ped_crossing",
                "walkway", "stop_line", "carpark_area", "road_divider", "lane_divider")
              if name in api.non_geometric_layers]
    x, y = pose["ego2global_translation"][:2]
    yaw = Quaternion(pose["ego2global_rotation"]).yaw_pitch_roll[0]
    masks = api.get_map_mask((x, y, 100, 100), math.degrees(yaw), layers, (200, 200)).astype(np.uint8)
    packed = sum((masks[index].astype(np.int32) << index) for index in range(len(layers)))
    distances = np.stack([(distance_transform_edt(mask) - distance_transform_edt(1 - mask)).astype(np.float32)
                          for mask in masks])
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "semantic_mask.npy", packed, allow_pickle=False)
    np.save(target / "distance_fields.npy", distances, allow_pickle=False)
    assert layers == ["drivable_area", "road_segment", "lane", "ped_crossing", "walkway",
                      "stop_line", "carpark_area", "road_divider", "lane_divider"]


def _trajectory_rows(boxes: list[dict], poses: list[dict], aliases: dict[str, str]) -> list[dict]:
    anchor = poses[9]
    anchor_q = Quaternion(anchor["ego2global_rotation"])
    anchor_t = np.asarray(anchor["ego2global_translation"], dtype=float)
    anchor_yaw = anchor_q.yaw_pitch_roll[0]
    result = []
    for row in boxes:
        center = anchor_q.inverse.rotate(np.asarray([row["x"], row["y"], row["z"]]) - anchor_t)
        velocity = anchor_q.inverse.rotate([row["vx"] or 0.0, row["vy"] or 0.0, 0.0])
        result.append({
            "frame_index": row["frame_index"],
            "agent_id": aliases[row["instance_token"]],
            "category": row["category_mapped"],
            "is_dynamic": row["is_dynamic"],
            "x": float(center[0]), "y": float(center[1]), "z": float(center[2]),
            "width": row["width"], "length": row["length"], "height": row["height"],
            "yaw": _angle(row["yaw"] - anchor_yaw),
            "vx": float(velocity[0]), "vy": float(velocity[1]),
            "valid": True,
        })
    return result


def _ego_rows(poses: list[dict]) -> list[dict]:
    anchor = poses[9]
    anchor_q = Quaternion(anchor["ego2global_rotation"])
    anchor_t = np.asarray(anchor["ego2global_translation"], dtype=float)
    anchor_yaw = anchor_q.yaw_pitch_roll[0]
    result = []
    for pose in poses:
        point = anchor_q.inverse.rotate(np.asarray(pose["ego2global_translation"], dtype=float) - anchor_t)
        yaw = _angle(Quaternion(pose["ego2global_rotation"]).yaw_pitch_roll[0] - anchor_yaw)
        result.append({"frame_index": pose["frame_index"], "x": float(point[0]), "y": float(point[1]),
                       "yaw": yaw, "valid": True})
    return result


def _build_scene(nusc, record: dict, source_root: Path, target: Path, public_id: str, size: tuple[int, int]) -> dict:
    timeline = json.loads((source_root / "scenes" / record["scene_name"] / "timeline.json").read_text())["frames"]
    calibration, _, poses = _calibration(nusc, timeline, record["scene_name"])
    boxes = []
    for frame in timeline:
        boxes += _boxes(nusc, record["scene_name"], frame["cameras"]["CAM_FRONT"]["sample_data_token"], frame["frame_index"], None)
    _finite_differences(boxes)
    instance_tokens = sorted({row["instance_token"] for row in boxes})
    aliases = {token: str(index) for index, token in enumerate(instance_tokens, 1)}
    public_boxes = _trajectory_rows(boxes, poses, aliases)
    ego = _ego_rows(poses)

    for frame in timeline[:10]:
        index = frame["frame_index"]
        for camera in CAMERAS:
            _resize(Path(frame["cameras"][camera]["source_path"]), target / "history/rgb" / camera / f"{index:06d}.jpg", size)

    first = {camera: next(row for row in calibration if row["camera"] == camera) for camera in CAMERAS}
    cameras = []
    for camera in CAMERAS:
        row = first[camera]
        native = np.asarray(row["camera_intrinsics"], dtype=float)
        challenge = native.copy()
        challenge[0] *= size[0] / int(row["image_width"])
        challenge[1] *= size[1] / int(row["image_height"])
        cameras.append({"camera": camera, "sensor2ego_translation_m": row["sensor2ego_translation"],
                        "sensor2ego_rotation_wxyz": row["sensor2ego_rotation"],
                        "native_image_size": [row["image_width"], row["image_height"]],
                        "challenge_image_size": list(size), "native_intrinsics": native.tolist(),
                        "challenge_intrinsics": challenge.tolist(), "resize_protocol": "direct_resize_no_crop_no_pad"})
    write_json(target / "calibration/cameras.json", {"camera_order": CAMERAS, "quaternion_convention": "wxyz", "cameras": cameras})

    write_csv(target / "history/agents.csv", [row for row in public_boxes if row["frame_index"] < 10])
    write_csv(target / "history/ego.csv", ego[:10])
    _write_local_map(nusc, record["location"], poses[9], target / "map")
    images = sorted((target / "history/rgb").rglob("*.jpg"))
    return {"scene_id": public_id, "split": record["challenge_split"], "history_image_count": len(images),
            "history_sha256": _sha256(images), "history_agent_rows": sum(row["frame_index"] < 10 for row in public_boxes),
            "agent_id_map": {value: key for key, value in aliases.items()}}


def validate(root: Path, expected: int = 300) -> dict:
    import cv2

    errors = []
    expected_ids = [f"adv_case_{index:04d}" for index in range(1, expected + 1)]
    scene_root = root / "scenes"
    actual_ids = sorted(path.name for path in scene_root.iterdir() if path.is_dir()) if scene_root.is_dir() else []
    if actual_ids != expected_ids: errors.append("scene_count")
    if {path.name for path in root.iterdir()} != {"README.md", "scenes"}: errors.append("root_schema")
    for scene_id in expected_ids:
        base = scene_root / scene_id
        if not base.is_dir():
            errors.append(f"{scene_id}:missing")
            continue
        if {path.name for path in base.iterdir()} != {"history", "calibration", "map"}:
            errors.append(f"{scene_id}:schema")
        images = sorted((base / "history/rgb").rglob("*.jpg"))
        if len(images) != 60 or any(cv2.imread(str(path)).shape[:2] != (224, 400) for path in images): errors.append(f"{scene_id}:history_images")
        history = list(csv.DictReader((base / "history/agents.csv").open()))
        agent_keys = ("frame_index", "agent_id", "category", "is_dynamic", "x", "y", "z", "width", "length",
                      "height", "yaw", "vx", "vy", "valid")
        if not history or tuple(history[0]) != agent_keys or min(int(row["frame_index"]) for row in history) < 0 or max(int(row["frame_index"]) for row in history) > 9:
            errors.append(f"{scene_id}:history")
        elif any(not row["agent_id"].isdigit() or int(row["agent_id"]) < 1 for row in history):
            errors.append(f"{scene_id}:agent_id")
        elif any(not math.isfinite(float(row[key])) for row in history for key in ("x", "y", "z", "width", "length", "height", "yaw")):
            errors.append(f"{scene_id}:nonfinite_history")
        ego = list(csv.DictReader((base / "history/ego.csv").open()))
        if len(ego) != 10 or tuple(ego[0]) != ("frame_index", "x", "y", "yaw", "valid"):
            errors.append(f"{scene_id}:ego")
        if np.load(base / "map/semantic_mask.npy", mmap_mode="r").shape != (200, 200): errors.append(f"{scene_id}:map")
        if any(path.suffix.lower() in {".jpg", ".png"} and "history" not in path.parts for path in base.rglob("*")): errors.append(f"{scene_id}:future_rgb")
        for path in base.rglob("*"):
            if path.suffix.lower() in {".json", ".csv", ".md"}:
                text = path.read_text(errors="ignore")
                if any(value in text for value in ("/media/", "scene_token", "sample_data_token", "instance_token", "source_path")):
                    errors.append(f"{scene_id}:source_identifier")
    return {"status": "completed" if not errors else "failed", "expected_scenes": expected,
            "validated_scenes": expected - len({e.split(":")[0] for e in errors if ":" in e}), "errors": errors}


def _scene_summary(target: Path, public_id: str, split: str) -> dict:
    images = sorted((target / "history/rgb").rglob("*.jpg"))
    with (target / "history/agents.csv").open(newline="") as handle:
        history_rows = sum(1 for _ in csv.DictReader(handle))
    return {"scene_id": public_id, "split": split, "history_image_count": len(images),
            "history_sha256": _sha256(images), "history_agent_rows": history_rows}


def _work_scene(job: tuple[int, dict]) -> tuple[int, dict, dict]:
    index, record = job; public_id = f"adv_case_{index:04d}"
    output, source, size = _WORKER["output"], _WORKER["source"], _WORKER["size"]
    target = output / "scenes" / public_id
    if target.exists(): shutil.rmtree(target)
    building = target.with_name(target.name + ".building")
    if building.exists(): shutil.rmtree(building)
    item = _build_scene(_WORKER["nusc"], record, source, building, public_id, size)
    public_item = {key: value for key, value in item.items() if key != "agent_id_map"}
    private_item = {"scene_id": public_id, "source_scene_id": record["scene_name"],
                    "source_scene_token": record["scene_token"], "agent_id_map": item["agent_id_map"]}
    building.rename(target)
    return index, public_item, private_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="driveaudit_eval/data/preparation.yaml")
    parser.add_argument("--output", default="challenge/input-op/nuscenes300_challenger_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    config = read_yaml(args.config); source = ROOT / "challenge/input" / config["dataset"]["name"]
    output = ROOT / args.output; expected = int(config["dataset"]["scene_count"])
    if args.validate_only:
        result = validate(output, expected)
        write_json(ROOT / "challenge/eval_output" / config["dataset"]["name"] / "player_input_validation.json", result)
        print(json.dumps(result, indent=2)); raise SystemExit(result["status"] != "completed")
    audit = discover(config, ROOT / "challenge/eval_output" / config["dataset"]["name"] / "data_source_audit.json")
    if audit["status"] != "completed": raise SystemExit("nuScenes trainval data not found")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=config["dataset"]["version"], dataroot=audit["dataroot"], verbose=False)
    selection = json.loads((source / "scene_selection.json").read_text())
    if output.exists() and args.force: shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    completed, jobs = {}, []
    private_partial = ROOT / "challenge/gt" / config["dataset"]["name"] / "player_input_private_scenes"
    size = tuple(config["challenge"]["resolution"])
    for index, record in enumerate(selection["scenes"], 1):
        public_id = f"adv_case_{index:04d}"; target = output / "scenes" / public_id
        private_path = private_partial / f"{public_id}.json"
        if target.is_dir() and private_path.is_file() and args.resume:
            completed[index] = (_scene_summary(target, public_id, record["challenge_split"]), json.loads(private_path.read_text()))
        else:
            jobs.append((index, record))
    _WORKER.update(nusc=nusc, output=output, source=source, size=size)
    if args.max_workers > 1 and jobs:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=mp.get_context("fork")) as executor:
            iterator = executor.map(_work_scene, jobs)
            for index, public_item, private_item in iterator:
                write_json(private_partial / f"adv_case_{index:04d}.json", private_item)
                completed[index] = (public_item, private_item)
                print(f"[{len(completed):03d}/{expected}] {public_item['scene_id']}", flush=True)
    else:
        for job in jobs:
            index, public_item, private_item = _work_scene(job)
            write_json(private_partial / f"adv_case_{index:04d}.json", private_item)
            completed[index] = (public_item, private_item)
            print(f"[{len(completed):03d}/{expected}] {public_item['scene_id']}", flush=True)
    public = [completed[index][0] for index in sorted(completed)]
    private = [completed[index][1] for index in sorted(completed)]
    write_json(ROOT / "challenge/gt" / config["dataset"]["name"] / "player_input_private_mapping.json",
               {"version": "challenger_player_input_mapping_v1", "public_root": str(output), "scenes": private})
    (output / "README.md").write_text(_README)
    result = validate(output, expected); write_json(ROOT / "challenge/eval_output" / config["dataset"]["name"] / "player_input_validation.json", result)
    if result["status"] != "completed": raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps({"status": "completed", "output": str(output), "scene_count": expected,
                      "history_images": sum(row["history_image_count"] for row in public)}, indent=2))


if __name__ == "__main__":
    main()

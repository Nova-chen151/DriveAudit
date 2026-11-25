from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, image_path

from driveaudit_eval.research.consistency.verification_benchmark.common import load_yaml, write_json


def discover(config: dict[str, Any]) -> dict[str, Any]:
    pair_manifest = load_yaml(config["inputs"]["pair_manifest"])
    scenes, invalid = [], []
    for pair in pair_manifest.get("pairs", []):
        missing = [str(image_path(pair, camera, frame)) for camera in CAMERAS for frame in range(90)
                   if not image_path(pair, camera, frame).is_file()]
        required = [pair["trajectory_sequence"]["trajectory_path"], pair["calibration"]["path"],
                    pair["ego_pose"]["path"], pair["history_metadata_path"]]
        missing.extend(str(path) for path in required if not Path(path).is_file())
        row = {"pair_id": pair["pair_id"], "scene_id": pair["image_sequence"]["scene_id"],
               "image_history_root": pair["image_sequence"]["history_root"],
               "generated_future_root": pair["image_sequence"]["generated_future_root"],
               "trajectory_path": pair["trajectory_sequence"]["trajectory_path"],
               "calibration_path": pair["calibration"]["path"], "ego_pose_path": pair["ego_pose"]["path"],
               "strict_pair_valid": not missing, "missing": missing}
        (scenes if not missing else invalid).append(row)
    result = {"available_history_scene_count": len(pair_manifest.get("pairs", [])),
              "available_generated_scene_count": sum(Path(row["generated_future_root"]).is_dir() for row in scenes),
              "available_submission_trajectory_count": sum(Path(row["trajectory_path"]).is_file() for row in scenes),
              "available_calibration_count": sum(Path(row["calibration_path"]).is_file() for row in scenes),
              "available_ego_pose_count": sum(Path(row["ego_pose_path"]).is_file() for row in scenes),
              "strictly_pairable_scene_count": len(scenes), "scenes": scenes, "invalid_scenes": invalid,
              "cross_scene_pairing_available": len(scenes) >= 2}
    write_json(Path(config["output"]) / "manifest/discovery.json", result)
    return result


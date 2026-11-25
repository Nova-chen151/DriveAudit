from __future__ import annotations

import json
from driveaudit_eval.agents.registry import dependencies_root
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, json_cell, read_csv, true_cell, write_json


def _gpu_inventory() -> dict[str, Any]:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(", ")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return {"name": query[0], "memory_total_mib": int(query[1]), "memory_used_mib": int(query[2]),
                "memory_free_mib": int(query[3]), "compute_processes": processes}
    except Exception as exc:
        return {"status": f"unavailable:{type(exc).__name__}"}


def audit_data(config: dict[str, Any], pair: dict[str, Any], pair_validation: dict[str, Any],
               root: str | Path = ".") -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    image_cfg, timeline = pair["image_sequence"], pair["timeline"]
    condition_root = root / image_cfg["conditions_root"]
    image_root = root / image_cfg["future_root"]
    history_image_root = root / image_cfg["history_root"]
    metadata = json.loads((condition_root / "metadata.json").read_text(encoding="utf-8"))
    frames = read_csv(condition_root / "frames.csv")
    poses = read_csv(condition_root / "ego_pose.csv")
    calibrations = read_csv(condition_root / "camera_calibration.csv")
    boxes = [row for row in read_csv(condition_root / "boxes.csv") if true_cell(row.get("valid_flag", True))]
    history_n, future_n = int(timeline["image_history_frames"]), int(timeline["image_future_frames"])
    future_indices, history_indices = range(history_n, history_n + future_n), range(history_n)
    counts = {camera: sum((image_root / camera / f"{frame:06d}.jpg").is_file() for frame in future_indices)
              for camera in CAMERA_ORDER}
    history_counts = {camera: sum((history_image_root / camera / f"{frame:06d}.jpg").is_file() for frame in history_indices)
                      for camera in CAMERA_ORDER}
    resolutions = {}
    for camera in CAMERA_ORDER:
        images = sorted((image_root / camera).glob("*.jpg"))
        if images:
            with Image.open(images[0]) as image:
                resolutions[camera] = list(image.size)
    timestamps = [float(row["timestamp"]) for row in frames]
    dt = (timestamps[-1] - timestamps[0]) / 1e6 / max(1, len(timestamps) - 1)
    history_ids = {row["obj_id"] for row in boxes if int(row["frame_index"]) < history_n}
    future_reference_ids = {row["obj_id"] for row in boxes if int(row["frame_index"]) >= history_n}
    paired_scene = pair_validation.get("trajectory_scene") if pair_validation.get("pair_validation_status") == "valid" else None
    backend_cfg = config["backend"]
    checkpoint = root / backend_cfg["checkpoint"] if backend_cfg.get("checkpoint") else None
    cache = root / backend_cfg["cache"] if backend_cfg.get("cache") else None
    issues = []
    if any(count != future_n for count in counts.values()):
        issues.append("incomplete_future_six_camera_images")
    if len(calibrations) != len(frames) * len(CAMERA_ORDER):
        issues.append("calibration_row_count_mismatch")
    issues.extend(pair_validation.get("errors", []))
    gpu = _gpu_inventory()
    if isinstance(gpu.get("memory_free_mib"), int) and gpu["memory_free_mib"] < 4096:
        issues.append("fresh_bev_inference_blocked_active_gpu")
    inventory = {
        "pair_id": pair["pair_id"], "image_scene_id": image_cfg["scene_id"],
        "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"], "history_frames": history_n,
        "future_frames": future_n, "fps": 1.0 / dt, "dt_seconds": dt,
        "camera_order": CAMERA_ORDER, "history_image_root": str(history_image_root), "history_image_counts": history_counts,
        "future_image_root": str(image_root), "future_image_counts": counts, "image_resolutions": resolutions,
        "image_source": {"path": str(image_root), "role": "gt_image_sequence_frame_alignment"},
        "future_boxes_source": {"path": str(condition_root / "boxes.csv"),
                                "role": "gt_image_attached_metadata_reference_not_pixel_detection"},
        "six_camera_complete_ratio": sum(v == future_n for v in counts.values()) / len(CAMERA_ORDER),
        "calibration_rows": len(calibrations), "calibration_fields": list(calibrations[0]) if calibrations else [],
        "intrinsics_available": all(bool(json_cell(row["camera_intrinsics"])) for row in calibrations),
        "sensor2ego_available": all(bool(json_cell(row["sensor2ego_translation"])) for row in calibrations),
        "ego2global_available": len(poses) == len(frames), "ego_pose_path": str(condition_root / "ego_pose.csv"),
        "lidar2img": "derivable_from_sensor2ego_ego2global_and_intrinsics",
        "history_boxes_path": str(condition_root / "boxes.csv"), "history_known_agent_count": len(history_ids),
        "history_known_agent_ids": sorted(history_ids), "future_reference_agent_count": len(future_reference_ids),
        "history_box_rows": sum(int(row["frame_index"]) < history_n for row in boxes),
        "future_reference_box_rows": sum(int(row["frame_index"]) >= history_n for row in boxes),
        "categories": dict(Counter(row["class_name"] for row in boxes)),
        "box_shape_fields": {"length": True, "width": True, "height": True},
        "valid_mask": "available_in_explicitly_paired_submitted_trajectory" if paired_scene else "not_available",
        "submitted_trajectory_scene": paired_scene["scene_id"] if paired_scene else None,
        "trajectory_manifest_source": pair["trajectory_sequence"]["manifest_path"],
        "checkpoint": {"path": str(checkpoint) if checkpoint else None, "exists": bool(checkpoint and checkpoint.exists()),
                       "bytes": checkpoint.stat().st_size if checkpoint and checkpoint.exists() else None},
        "cached_detection_output": {"path": str(cache) if cache else None, "exists": bool(cache and cache.exists())},
        "available_bev_extractors": {
            "standalone_bevformer_or_bevdet": "not_found",
            "uniad_perception_candidate": {"checkpoint": str(dependencies_root() / "DrivingAgents/UniAD/ckpts/uniad_base_e2e.pth"),
                                            "exists": (dependencies_root() / "DrivingAgents/UniAD/ckpts/uniad_base_e2e.pth").exists()},
            "vad_perception_candidate": "not_used",
            "cached_vad_detection": False,
            "gt_metadata_reference": backend_cfg["name"] == "gt_metadata_boxes",
            "independent_2d_detector_or_depth_estimator": "not_found",
            "tracker": "UniADTrack exists; evaluator constant-velocity Hungarian tracker implemented",
        },
        "camera_geometry_helpers": ["driveaudit_eval/agents/core.py", "driveaudit_eval/agents/agent_input_adapter.py"],
        "backend": backend_cfg["name"], "backend_diagnostic_only": bool(backend_cfg.get("diagnostic_only", False)),
        "gpu": gpu, "pair_validation_status": pair_validation["pair_validation_status"],
        "pair_warnings": pair_validation.get("warnings", []), "issues": sorted(set(issues)),
    }
    manifest = {
        "schema_version": config["schema_version"], "pair_id": pair["pair_id"], "scene_id": pair["pair_id"],
        "image_scene_id": image_cfg["scene_id"], "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"],
        "camera_order": CAMERA_ORDER, "history_frame_indices": list(range(history_n)),
        "future_frame_indices": list(range(history_n, history_n + future_n)),
        "timestamps": timestamps, "conditions": str(condition_root), "future_images": str(image_root),
        "submitted_trajectory": paired_scene, "evaluation_only_future_boxes": str(condition_root / "boxes.csv"),
        "backend": {**backend_cfg, "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
                    "cache_exists": bool(cache and cache.exists())},
        "pair_validation": {key: value for key, value in pair_validation.items() if key != "trajectory_scene"},
        "formal_evaluation_ready": paired_scene is not None and not backend_cfg.get("diagnostic_only", False),
        "blockers": sorted(set(issues)),
    }
    return inventory, manifest


def write_audit(config: dict[str, Any], pair: dict[str, Any], pair_validation: dict[str, Any],
                output: Path, docs: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory, manifest = audit_data(config, pair, pair_validation)
    write_json(output / "data_inventory.json", inventory)
    write_json(output / "manifest.json", manifest)
    table = "\n".join(f"- {key}: `{value}`" for key, value in inventory.items())
    report = "# 六视角图像—轨迹一致性：数据与代码审计\n\n" + table + "\n"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "data_and_code_audit.md").write_text(report, encoding="utf-8")
    (docs / "code_and_data_audit.md").write_text(report, encoding="utf-8")
    return inventory, manifest

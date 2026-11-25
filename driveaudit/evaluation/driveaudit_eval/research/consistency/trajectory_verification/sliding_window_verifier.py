from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification.backends import OpenCVLocalBackend
from driveaudit_eval.research.consistency.trajectory_verification.common import read_image, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.local_evidence import verify_local_window
from driveaudit_eval.research.consistency.trajectory_verification.schema import validate_evidence


def verify_windows(pair: dict[str, Any], expectations: list[dict[str, Any]], config: dict[str, Any],
                   output: str | Path, evidence_mode: str = "fusion", frame_end: int = 90) -> dict[str, Any]:
    backend = OpenCVLocalBackend(config["local_backend"])
    image_size = (int(config["projection"]["image_width"]), int(config["projection"]["image_height"]))
    cache: dict[tuple[str, int], Any] = {}

    def image(camera: str, frame: int):
        key = (camera, frame)
        if key not in cache:
            cache[key] = read_image(pair, camera, frame, image_size)
        return cache[key]

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for expectation in expectations:
        if not expectation.get("expected_visible") or int(expectation["target_frame_index"]) >= frame_end:
            continue
        camera, frame, target = expectation["camera_name"], int(expectation["frame_index"]), int(expectation["target_frame_index"])
        result = verify_local_window(backend, image(camera, frame), image(camera, target), expectation, evidence_mode)
        row = {"pair_id": pair["pair_id"], "agent_id": expectation["agent_id"],
               "agent_type": expectation["agent_type"], "history_anchored": expectation["history_anchored"],
               "camera_name": camera, "frame_index": frame, "target_frame_index": target,
               "timestamp": expectation["timestamp"], "target_timestamp": expectation["target_timestamp"],
               "window_length": expectation["window_length"], "expected_visible": True,
               "metric_type": "conditional_visual_verification", "projected_area": expectation["source_projection"]["projected_area"],
               "projected_depth": expectation["source_projection"]["projected_depth"],
               "occlusion_ratio": expectation["source_projection"]["occlusion_ratio"],
               "expected_camera_count": expectation["source_projection"]["expected_camera_count"],
               "source_projected_box": expectation["source_projection"]["projected_box_2d"],
               "target_projected_box": expectation["target_projection"]["projected_box_2d"],
               "expected_motion_magnitude": expectation["expected_motion_magnitude"],
               "expected_scale_change": expectation["expected_scale_change"],
               "expected_rotation_change": expectation["expected_rotation_change"], **result}
        validate_evidence(row)
        rows.append(row)
    output = Path(output)
    write_csv(output / "per_window.csv", rows)
    write_json(output / "per_window.json", rows)
    profile = {"backend": "opencv_lk_farneback_orb_ransac", "evidence_mode": evidence_mode,
               "window_count": len(rows), "total_seconds": time.perf_counter() - started,
               "mean_window_seconds": (time.perf_counter() - started) / max(1, len(rows)),
               "long_horizon_state_carried": False, "full_image_search_used": False,
               "gpu_used": False, "peak_gpu_memory_mib": 0.0, "future_gt_used": False,
               "random_initialization_used": False}
    write_json(output / "inference_profile.json", profile)
    return {"rows": rows, "profile": profile}


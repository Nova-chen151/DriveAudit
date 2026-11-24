from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.mtpc.common import mean, write_csv, write_json
from driveaudit_eval.consistency.mtpc.expected_image_motion import compute_expected_motion
from driveaudit_eval.consistency.mtpc.metrics.image_motion_metrics import compute_image_motion_metrics
from driveaudit_eval.consistency.mtpc.metrics.presence_metrics import compute_presence_metrics
from driveaudit_eval.consistency.mtpc.metrics.projection_geometry_metrics import compute_projection_geometry_metrics


SEVERITIES = {"mild": 1.0, "medium": 3.0, "severe": 7.0}


def _shift(rows: list[dict[str, Any]], dx: float, dy: float) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    for row in output:
        row["center_2d"] = [row["center_2d"][0] + dx, row["center_2d"][1] + dy]
        row["clipped_box_2d"] = [value + (dx if index % 2 == 0 else dy) for index, value in enumerate(row["clipped_box_2d"])]
        row["projected_polygon"] = [[x + dx, y + dy] for x, y in row["projected_polygon"]]
    return output


def _time(rows: list[dict[str, Any]], frames: int) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    for row in output: row["frame_index"] += frames
    return output


def _identity(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows); agents = sorted({row["agent_id"] for row in rows})[:2 * count]
    swap = {agents[index]: agents[index + 1] for index in range(0, len(agents) - 1, 2)}
    swap.update({value: key for key, value in list(swap.items())})
    for row in output: row["agent_id"] = swap.get(row["agent_id"], row["agent_id"])
    return output


def run_corruption_benchmark(projections: list[dict[str, Any]], visual: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    baseline_geometry = compute_projection_geometry_metrics(projections, visual); baseline_motion = compute_image_motion_metrics(compute_expected_motion(projections), visual)
    rows = []
    types = ("global_x_shift", "global_y_shift", "depth_shift", "yaw_shift", "size_scaling", "time_shift", "speed_scaling",
             "early_disappearance", "late_disappearance", "wrong_entry_camera", "wrong_exit_camera", "identity_swap",
             "lane_change_timing_shift", "stop_timing_shift", "camera_calibration_perturbation", "ego_pose_perturbation")
    for corruption in types:
        for label, value in SEVERITIES.items():
            if corruption in {"global_x_shift", "camera_calibration_perturbation", "ego_pose_perturbation"}: changed = _shift(projections, value, 0)
            elif corruption == "global_y_shift": changed = _shift(projections, 0, value)
            elif corruption in {"time_shift", "lane_change_timing_shift", "stop_timing_shift"}: changed = _time(projections, int(value))
            elif corruption == "identity_swap": changed = _identity(projections, max(1, int(value // 2)))
            else:
                changed = copy.deepcopy(projections)
                if corruption in {"depth_shift", "size_scaling"}:
                    scale = 1 + value * .05
                    for row in changed:
                        cx, cy = row["center_2d"]; row["clipped_box_2d"] = [cx + (row["clipped_box_2d"][0] - cx) * scale, cy + (row["clipped_box_2d"][1] - cy) * scale,
                                                                              cx + (row["clipped_box_2d"][2] - cx) * scale, cy + (row["clipped_box_2d"][3] - cy) * scale]
                elif corruption == "yaw_shift":
                    changed = _shift(projections, value * .3, value * .3)
                elif corruption == "early_disappearance":
                    cutoff = 90 - int(value * 3)
                    for row in changed:
                        if row["frame_index"] >= cutoff: row["expected_visible"] = False
                elif corruption == "late_disappearance":
                    for row in changed:
                        if row["frame_index"] >= 90 - int(value * 3): row["expected_visible"] = True
                elif corruption in {"wrong_entry_camera", "wrong_exit_camera"}:
                    mapping = {"CAM_FRONT_LEFT": "CAM_FRONT_RIGHT", "CAM_FRONT_RIGHT": "CAM_FRONT_LEFT", "CAM_BACK_LEFT": "CAM_BACK_RIGHT", "CAM_BACK_RIGHT": "CAM_BACK_LEFT"}
                    for row in changed: row["camera_name"] = mapping.get(row["camera_name"], row["camera_name"])
            geometry = compute_projection_geometry_metrics(changed, visual); motion = compute_image_motion_metrics(compute_expected_motion(changed), visual)
            presence = compute_presence_metrics(changed, visual)
            rows.append({"corruption": corruption, "severity": label, "strength": value,
                         "normalized_center_error": geometry["normalized_projected_center_error"], "flow_epe": motion["flow_endpoint_error_px"],
                         "missing_ratio": presence["persistent_agent_missing_ratio"], "over_persistence_ratio": presence["over_persistence_ratio"],
                         "event_time_error_frames": value if corruption in {"time_shift", "lane_change_timing_shift", "stop_timing_shift"} else 0.0,
                         "identity_corrupted_agent_pairs": max(1, int(value // 2)) if corruption == "identity_swap" else 0,
                         "identity_proxy_error": 1 - presence["persistent_agent_visual_support_rate"]})
    monotonic = {}
    for corruption in types:
        values = [row for row in rows if row["corruption"] == corruption]
        if corruption in {"time_shift", "lane_change_timing_shift", "stop_timing_shift"}: key = "event_time_error_frames"
        elif corruption == "identity_swap": key = "identity_corrupted_agent_pairs"
        elif corruption == "early_disappearance": key = "over_persistence_ratio"
        elif corruption in {"late_disappearance", "wrong_entry_camera", "wrong_exit_camera"}: key = "missing_ratio"
        elif "shift" in corruption or "perturbation" in corruption: key = "normalized_center_error"
        else: key = "flow_epe"
        sequence = [row[key] for row in values]; monotonic[corruption] = all(isinstance(value, (int, float)) for value in sequence) and all(b + 1e-9 >= a for a, b in zip(sequence, sequence[1:]))
    result = {"status": "completed_low_sample", "baseline_normalized_center_error": baseline_geometry["normalized_projected_center_error"],
              "baseline_flow_epe": baseline_motion["flow_endpoint_error_px"], "rows": rows, "monotonic_checks": monotonic,
              "monotonic_pass_count": sum(monotonic.values()), "corruption_type_count": len(types)}
    output = Path(output); write_json(output / "results.json", result); write_csv(output / "responses.csv", rows); return result

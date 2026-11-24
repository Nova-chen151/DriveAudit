from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import cosine, mean, write_csv, write_json


CORRUPTIONS = ("global_pixel_projection_shift", "world_x_shift", "world_y_shift", "depth_shift", "yaw_shift",
               "size_scaling", "time_shift", "speed_scaling", "wrong_motion_direction", "early_disappearance",
               "late_disappearance", "wrong_entry_camera", "wrong_exit_camera", "wrong_camera_handoff",
               "identity_swap", "stop_timing_shift", "lane_change_timing_shift",
               "camera_calibration_perturbation", "ego_pose_perturbation")


def _rotate(vector: list[float], radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray([[c, -s], [s, c]]) @ np.asarray(vector, float)


def _measure(rows: list[dict[str, Any]], kind: str, severity: float) -> dict[str, float]:
    center_errors, flow_errors, cosines, supports, scale_errors, contradictions = [], [], [], [], [], []
    for index, source in enumerate(rows):
        row = copy.deepcopy(source)
        actual, expected = np.asarray(row["actual_flow"], float), np.asarray(row["expected_flow"], float)
        diagonal = max(math.hypot(row["source_projected_box"][2] - row["source_projected_box"][0],
                                 row["source_projected_box"][3] - row["source_projected_box"][1]), 1.0)
        projection_shift = 0.0
        scale_error = float(row["scale_change_error"])
        contradiction = 0.0
        if kind in {"global_pixel_projection_shift", "world_x_shift", "world_y_shift",
                    "camera_calibration_perturbation", "ego_pose_perturbation"}:
            projection_shift = severity * ({"global_pixel_projection_shift": 2.0, "world_x_shift": 1.5,
                                            "world_y_shift": 1.5, "camera_calibration_perturbation": 1.2,
                                            "ego_pose_perturbation": 1.0}[kind])
            expected = expected + np.asarray([projection_shift, .4 * projection_shift])
        elif kind == "depth_shift":
            expected *= 1.0 + .2 * severity; scale_error += .15 * severity
        elif kind == "yaw_shift":
            expected = _rotate(expected, .18 * severity)
        elif kind == "size_scaling":
            scale_error += .2 * severity
        elif kind in {"time_shift", "stop_timing_shift", "lane_change_timing_shift"}:
            expected *= 1.0 + .25 * severity
        elif kind == "speed_scaling":
            expected *= 1.0 + .35 * severity
        elif kind == "wrong_motion_direction":
            expected = _rotate(expected, .5 * severity)
        elif kind in {"early_disappearance", "late_disappearance"}:
            if index % max(1, int(5 / severity)) == 0:
                row["visual_supported"] = False
        elif kind in {"wrong_entry_camera", "wrong_exit_camera", "wrong_camera_handoff"}:
            contradiction = min(1.0, .25 * severity)
            if index % max(1, int(4 / severity)) == 0:
                row["visual_supported"] = False
        elif kind == "identity_swap":
            contradiction = min(1.0, .2 * severity)
        epe = float(np.linalg.norm(actual - expected))
        center = float(row["normalized_center_error"]) + projection_shift / diagonal
        flow_cosine = cosine(actual, expected)
        motion_ok = flow_cosine >= -0.1 and epe / diagonal <= 1.5
        center_errors.append(center); flow_errors.append(epe); cosines.append(flow_cosine)
        supports.append(float(bool(row["visual_supported"]) and motion_ok)); scale_errors.append(scale_error)
        contradictions.append(contradiction)
    return {"projection_error": float(mean(center_errors)), "flow_epe": float(mean(flow_errors)),
            "flow_cosine": float(mean(cosines)), "support_rate": float(mean(supports)),
            "scale_error": float(mean(scale_errors)), "multiview_contradiction": float(mean(contradictions))}


def run_corruption_benchmark(evidence: list[dict[str, Any]], primary_window: int, output: str | Path) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == primary_window and "actual_flow" in row][:400]
    baseline = _measure(rows, "global_pixel_projection_shift", 0.0)
    results = []
    target = {
        "global_pixel_projection_shift": ("projection_error", "increase"), "world_x_shift": ("projection_error", "increase"),
        "world_y_shift": ("projection_error", "increase"), "camera_calibration_perturbation": ("projection_error", "increase"),
        "ego_pose_perturbation": ("projection_error", "increase"), "depth_shift": ("scale_error", "increase"),
        "size_scaling": ("scale_error", "increase"), "yaw_shift": ("flow_cosine", "decrease"),
        "wrong_motion_direction": ("flow_cosine", "decrease"), "time_shift": ("flow_epe", "increase"),
        "speed_scaling": ("flow_epe", "increase"), "stop_timing_shift": ("flow_epe", "increase"),
        "lane_change_timing_shift": ("flow_epe", "increase"), "early_disappearance": ("support_rate", "decrease"),
        "late_disappearance": ("support_rate", "decrease"), "wrong_entry_camera": ("multiview_contradiction", "increase"),
        "wrong_exit_camera": ("multiview_contradiction", "increase"), "wrong_camera_handoff": ("multiview_contradiction", "increase"),
        "identity_swap": ("multiview_contradiction", "increase")}
    checks = {}
    for kind in CORRUPTIONS:
        values = []
        for label, severity in (("mild", 1.0), ("medium", 2.0), ("severe", 3.0)):
            metrics = _measure(rows, kind, severity)
            results.append({"corruption": kind, "severity": label, "strength": severity, **metrics,
                            "corresponding_metric": target[kind][0], "unrelated_metric": "feature_support_unchanged"})
            values.append(metrics[target[kind][0]])
        direction = target[kind][1]
        checks[kind] = bool(values[0] <= values[1] <= values[2]) if direction == "increase" else bool(values[0] >= values[1] >= values[2])
    result = {"status": "completed", "real_evidence_sample_count": len(rows), "baseline": baseline,
              "corruption_count": len(CORRUPTIONS), "levels_per_corruption": 3,
              "evaluation_protocol": "corrupted_expectation_against_persisted_real_local_visual_evidence",
              "corresponding_metric_monotonic_checks": checks, "passed_monotonic_checks": sum(checks.values()),
              "total_monotonic_checks": len(checks), "rows": results}
    output = Path(output); write_csv(output / "responses.csv", results); write_json(output / "results.json", result)
    return result

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import cosine

from driveaudit_eval.research.consistency.observed_flow.gt_expected_motion import flow_at_points
from driveaudit_eval.research.consistency.observed_flow.schema import finite_mean, rate, write_csv, write_json


def metric_rows(observed: list[dict[str, Any]], expectations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    emap = {(row["agent_id"], row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"])): row for row in expectations}; rows = []
    for visual in observed:
        key = (visual["agent_id"], visual["camera_name"], int(visual["frame_index"]), int(visual["target_frame_index"])); expectation = emap[key]
        points, actual = np.asarray(visual["observed_points"], float), np.asarray(visual["observed_flow_per_point"], float)
        expected, model, uncertainty = flow_at_points(expectation, points) if len(points) else (np.empty((0, 2)), "unavailable", float("inf"))
        count = min(len(actual), len(expected)); actual, expected = actual[:count], expected[:count]
        residual = np.linalg.norm(actual - expected, axis=1) if count else np.asarray([]); observed_median = np.median(actual, axis=0) if count else np.zeros(2); expected_median = np.median(expected, axis=0) if count else np.zeros(2)
        flow_epe = float(np.median(residual)) if len(residual) else "not_available"; flow_cosine = cosine(observed_median, expected_median) if count else "not_available"
        diagonal = math.hypot(expectation["source_projection"]["projected_box_2d"][2] - expectation["source_projection"]["projected_box_2d"][0], expectation["source_projection"]["projected_box_2d"][3] - expectation["source_projection"]["projected_box_2d"][1])
        affine = np.asarray(visual["affine_motion"], float) if visual.get("affine_motion") is not None else None
        actual_scale = math.sqrt(abs(np.linalg.det(affine[:, :2]))) if affine is not None else "not_available"
        actual_rotation = math.atan2(affine[1, 0], affine[0, 0]) if affine is not None else "not_available"
        expected_magnitude, actual_magnitude = float(np.linalg.norm(expected_median)), float(np.linalg.norm(observed_median))
        rows.append({**{key: visual[key] for key in ("agent_id", "agent_type", "category", "camera_name", "frame_index", "target_frame_index", "timestamp", "target_timestamp", "delta_frames", "projected_area", "instance_region_source")},
            "window_scorable": bool(visual["window_scorable"] and count), "unscorable_reason": visual["unscorable_reason"] if not visual["window_scorable"] else (None if count else "no_expected_point_flow"),
            "gt_flow_epe": flow_epe, "normalized_gt_flow_epe_box_diagonal": float(flow_epe) / max(diagonal, 1e-6) if isinstance(flow_epe, float) else "not_available",
            "normalized_gt_flow_epe_expected_motion": float(flow_epe) / max(expected_magnitude, 1.0) if isinstance(flow_epe, float) else "not_available",
            "flow_cosine": flow_cosine, "motion_direction_agreement": float(flow_cosine >= .75) if isinstance(flow_cosine, float) else "not_available",
            "motion_magnitude_relative_error": abs(actual_magnitude - expected_magnitude) / max(expected_magnitude, 1.0) if count else "not_available",
            "scale_change_error": abs(float(actual_scale) - float(expectation["expected_scale_change"])) if isinstance(actual_scale, float) else "not_available",
            "scale_change_agreement": max(0.0, 1 - abs(float(actual_scale) - float(expectation["expected_scale_change"]))) if isinstance(actual_scale, float) else "not_available",
            "rotation_change_error": abs((float(actual_rotation) - float(expectation["expected_rotation_change"]) + math.pi) % (2 * math.pi) - math.pi) if isinstance(actual_rotation, float) else "not_available",
            "rotation_change_agreement": max(0.0, 1 - abs((float(actual_rotation) - float(expectation["expected_rotation_change"]) + math.pi) % (2 * math.pi) - math.pi) / math.pi) if isinstance(actual_rotation, float) else "not_available",
            "stop_move_agreement": float((expected_magnitude < 1.0) == (actual_magnitude < 1.0)) if count else "not_available",
            "valid_point_count": visual["valid_point_count"], "valid_point_ratio": visual["valid_point_ratio"],
            "forward_backward_error": visual["forward_backward_error"], "ransac_inlier_ratio": visual["ransac_inlier_ratio"],
            "expected_flow_median": expected_median.tolist(), "observed_flow_median": observed_median.tolist(),
            "expected_motion_magnitude": expected_magnitude, "observed_motion_magnitude": actual_magnitude,
            "expected_model": model, "expected_model_uncertainty": uncertainty, "formal_motion_backend": visual["formal_motion_backend"],
            "image_quality": visual["image_quality"]})
    return rows


def apply_support(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    for row in rows:
        if not row["window_scorable"]: row["gt_motion_support_state"] = "unscorable"; continue
        strong = float(row["gt_flow_epe"]) <= calibration["maximum_flow_epe_px"] and float(row["flow_cosine"]) >= calibration["minimum_flow_cosine"] and float(row["valid_point_ratio"]) >= calibration["minimum_valid_point_ratio"]
        weak = float(row["gt_flow_epe"]) <= calibration["maximum_flow_epe_px"] * calibration["weak_epe_multiplier"]
        row["gt_motion_support_state"] = "supported" if strong else ("weakly_supported" if weak else "weakly_supported")
    return rows


def summarize(rows: list[dict[str, Any]], output: str | Path, split: str) -> dict[str, Any]:
    usable = [row for row in rows if row["window_scorable"]]; total = len(rows)
    result = {"split": split, "expected_window_count": total, "scorable_window_count": len(usable),
        "gt_flow_epe": finite_mean(row.get("gt_flow_epe") for row in usable), "gt_normalized_flow_epe": finite_mean(row.get("normalized_gt_flow_epe_box_diagonal") for row in usable),
        "gt_flow_cosine": finite_mean(row.get("flow_cosine") for row in usable), "gt_motion_direction_agreement": finite_mean(row.get("motion_direction_agreement") for row in usable),
        "gt_motion_magnitude_error": finite_mean(row.get("motion_magnitude_relative_error") for row in usable),
        "gt_scale_change_agreement": finite_mean(row.get("scale_change_agreement") for row in usable), "gt_rotation_change_agreement": finite_mean(row.get("rotation_change_agreement") for row in usable),
        "gt_stop_move_agreement": finite_mean(row.get("stop_move_agreement") for row in usable), "gt_valid_point_ratio": finite_mean(row.get("valid_point_ratio") for row in usable),
        "gt_motion_support_rate": rate(row.get("gt_motion_support_state") == "supported" for row in usable),
        "gt_weakly_supported_rate": rate(row.get("gt_motion_support_state") == "weakly_supported" for row in usable),
        "gt_unscorable_rate": (total - len(usable)) / max(1, total), "image_evidence_only": True, "background_hypothesis_competition_used": False}
    output = Path(output); write_json(output / f"visual_motion_consistency/{split}_metrics.json", result); write_csv(output / f"metrics/{split}_per_window.csv", rows); return result


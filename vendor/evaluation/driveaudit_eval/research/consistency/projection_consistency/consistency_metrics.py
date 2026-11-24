from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.geometry_flow.visual_consistency import compare, metrics, summarize

from driveaudit_eval.research.consistency.projection_consistency.schema import INSUFFICIENT_EVENT, NOT_AVAILABLE, mean, rate


def augment_row(row: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    value = dict(row)
    residual = _number(row.get("guided_residual_epe"))
    expected = np.asarray(row.get("guided_expected_flow_median", [0, 0]), dtype=float)
    observed = np.asarray(row.get("guided_observed_flow_median", [0, 0]), dtype=float)
    image_diagonal = math.hypot(*image_size)
    value["guided_normalized_residual_epe_image_diagonal"] = residual / image_diagonal if residual is not None else None
    value["guided_projected_center_residual"] = float(np.linalg.norm(observed - expected)) if expected.size == 2 and observed.size == 2 else None
    value["guided_stop_move_agreement"] = bool((np.linalg.norm(expected) < 1.0) == (np.linalg.norm(observed) < 1.0))
    affine = row.get("guided_residual_affine")
    if isinstance(affine, str):
        try: affine = json.loads(affine)
        except json.JSONDecodeError: affine = None
    if affine is not None:
        matrix = np.asarray(affine, dtype=float)
        linear = matrix[:, :2]
        value["guided_polygon_deformation_error"] = float(np.linalg.norm(linear - np.eye(2), ord="fro"))
    else:
        value["guided_polygon_deformation_error"] = None
    value["image_observation_source"] = "candidate_rgb_geometry_initialized_lk"
    value["trajectory_prediction_used_as_observation"] = bool(row.get("guided_gt_prediction_used_as_observation", False))
    return value


def summarize_generic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize(rows, "guided")
    usable = [row for row in rows if row.get("guided_support_state") != "unscorable"]
    base.update({
        "image_normalized_epe": mean(row.get("guided_normalized_residual_epe_image_diagonal") for row in usable),
        "box_normalized_epe": mean(row.get("guided_normalized_residual_epe_box_diagonal") for row in usable),
        "motion_normalized_epe": mean(row.get("guided_normalized_residual_epe_expected_motion") for row in usable),
        "stop_move_agreement": mean(row.get("guided_stop_move_agreement") for row in usable),
        "polygon_deformation_error": mean(row.get("guided_polygon_deformation_error") for row in usable),
        "projected_center_residual": mean(row.get("guided_projected_center_residual") for row in usable),
        "forward_backward_error": mean(row.get("guided_forward_backward_error") for row in usable),
        "ransac_inlier_ratio": mean(row.get("guided_ransac_inlier_ratio") for row in usable),
        "rgb_observation_window_rate": rate(not row.get("trajectory_prediction_used_as_observation", True) for row in rows),
        "support_state_counts": dict(Counter(row.get("guided_support_state", NOT_AVAILABLE) for row in rows)),
        "appearance_time_error": INSUFFICIENT_EVENT,
        "disappearance_time_error": INSUFFICIENT_EVENT,
        "visibility_duration_error": INSUFFICIENT_EVENT,
        "motion_continuity": _continuity(rows),
        "support_continuity": _support_continuity(rows),
    })
    return base


def relative_to_gt(submission: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    return {
        "support_retention": _ratio(submission.get("strong_support_rate"), gt.get("strong_support_rate")),
        "valid_point_retention": _ratio(submission.get("valid_point_ratio"), gt.get("valid_point_ratio")),
        "excess_normalized_epe": _difference(submission.get("box_normalized_epe"), gt.get("box_normalized_epe")),
        "flow_cosine_drop": _difference(gt.get("flow_cosine"), submission.get("flow_cosine")),
        "direction_drop": _difference(gt.get("direction_agreement"), submission.get("direction_agreement")),
        "multiview_contradiction_increase": _difference(submission.get("multiview_contradiction"), gt.get("multiview_contradiction")),
    }


def _continuity(rows: list[dict[str, Any]]) -> float | str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["agent_id"], row["camera_name"]), []).append(row)
    values = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["frame_index"]))
        states = [row.get("guided_flow_cosine") for row in ordered]
        states = [_number(value) for value in states]
        states = [value for value in states if value is not None]
        if len(states) > 1:
            values.append(float(np.clip(1 - np.mean(np.abs(np.diff(states))), 0, 1)))
    return mean(values)


def _support_continuity(rows: list[dict[str, Any]]) -> float | str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["agent_id"], row["camera_name"]), []).append(row)
    values = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["frame_index"]))
        supported = [row.get("guided_support_state") in {"supported", "weakly_supported"} for row in ordered]
        if len(supported) > 1:
            values.append(1 - sum(a != b for a, b in zip(supported, supported[1:])) / (len(supported) - 1))
    return mean(values)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _ratio(a: Any, b: Any) -> float | str:
    return float(a) / float(b) if _number(a) is not None and _number(b) not in {None, 0.0} else NOT_AVAILABLE


def _difference(a: Any, b: Any) -> float | str:
    return float(a) - float(b) if _number(a) is not None and _number(b) is not None else NOT_AVAILABLE


__all__ = ["augment_row", "compare", "metrics", "relative_to_gt", "summarize", "summarize_generic"]


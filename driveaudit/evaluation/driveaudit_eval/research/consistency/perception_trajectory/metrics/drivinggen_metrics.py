from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def _score(value: Any) -> Any:
    return 1.0 / (1.0 + max(0.0, float(value))) if isinstance(value, (int, float)) else NOT_AVAILABLE


def compute_drivinggen_metrics(formal: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    geometry, motion, presence, identity = (formal[key] for key in ("geometry", "motion", "presence", "identity"))
    expected = int(presence["expected_visible_agent_frames"])
    matched = int(coverage["matched_visible_agent_frames"])
    missing = max(0, expected - matched)
    trajectory_raw = {"speed_mae": motion["speed_mae"], "acceleration_mae": motion["acceleration_mae"],
                      "jerk_mae": motion["jerk_difference"], "yaw_rate_mae": motion["yaw_rate_mae"],
                      "curvature_mae": motion["curvature_difference"], "heading_mae": geometry["heading_mae"],
                      "direction_agreement": motion["motion_direction_agreement"]}
    comparable_scores = [_score(value) for key, value in trajectory_raw.items() if key != "direction_agreement"]
    comparable_scores = [value for value in comparable_scores if isinstance(value, (int, float))]
    trajectory_consistency = float(np.mean(comparable_scores)) if comparable_scores else NOT_AVAILABLE
    agent_parts = [identity["track_completeness"], identity["idf1_like"],
                   formal["detection"]["class_accuracy"], motion["motion_direction_agreement"],
                   presence["object_survival_consistency"]]
    agent_parts = [float(value) for value in agent_parts if isinstance(value, (int, float))]
    return {
        "reference_semantics": "submitted_trajectory_as_instruction_reference",
        "drivinggen_alignment_ade": geometry["bev_center_ade"],
        "drivinggen_alignment_fde": geometry["visible_fde"],
        "drivinggen_alignment_dtw": geometry["dtw_distance"],
        "matched_agent_frame_coverage": coverage["matched_coverage"],
        "expected_visible_agent_frames": expected, "matched_image_agent_frames": matched,
        "missing_visible_agent_frames": missing, "agent_missing_ratio": missing / expected if expected else NOT_AVAILABLE,
        "persistent_agent_missing_ratio": presence.get("by_group", {}).get("persistent_agents", {}).get("missing_ratio", NOT_AVAILABLE),
        "future_entry_missing_ratio": (1.0 - float(presence["future_entry_recall"])) if isinstance(presence["future_entry_recall"], (int, float)) else NOT_AVAILABLE,
        "agent_track_completeness": identity["track_completeness"],
        "agent_identity_consistency": identity["idf1_like"],
        "agent_class_consistency": formal["detection"]["class_accuracy"],
        "agent_size_consistency": _score((formal["detection"]["box_length_mae"] + formal["detection"]["box_width_mae"])
                                         if all(isinstance(formal["detection"][key], (int, float)) for key in ("box_length_mae", "box_width_mae")) else NOT_AVAILABLE),
        "agent_motion_direction_consistency": motion["motion_direction_agreement"],
        "agent_survival_consistency": presence["object_survival_consistency"],
        "agent_consistency": float(np.mean(agent_parts)) if agent_parts else NOT_AVAILABLE,
        "trajectory_consistency": trajectory_consistency, "trajectory_consistency_raw_errors": trajectory_raw,
        "trajectory_quality": trajectory_consistency, "trajectory_alignment": _score(geometry["bev_center_ade"]),
        "official_ftd": NOT_AVAILABLE,
        "official_ftd_unavailable_reason": "requires_official_mtr_encoder_checkpoint_reference_distribution_and_sufficient_samples",
        "implementation_level": "DIRECT_WITH_PROTOCOL_DIFFERENCE",
    }


from __future__ import annotations

import math
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def compute_task_fidelity_metrics(formal: dict[str, Any]) -> dict[str, Any]:
    detection, identity, geometry, motion = (formal[key] for key in ("detection", "identity", "geometry", "motion"))
    det_a = detection["true_positive"] / max(detection["true_positive"] + detection["false_positive"] + detection["false_negative"], 1)
    ass_a = identity["idf1_like"] if isinstance(identity["idf1_like"], (int, float)) else 0.0
    return {
        "reference_semantics": "submitted_trajectory_as_instruction_reference",
        "submission_ref_precision": detection["visible_detection_precision"],
        "submission_ref_recall": detection["visible_detection_recall"],
        "submission_ref_f1": detection["visible_detection_f1"],
        "submission_ref_id_precision": identity["id_precision"],
        "submission_ref_id_recall": identity["id_recall"],
        "submission_ref_idf1": identity["idf1_like"],
        "submission_ref_hota_adapted": math.sqrt(max(0.0, det_a * ass_a)),
        "hota_implementation_level": "CONCEPTUAL_ADAPTATION",
        "track_completeness": identity["track_completeness"],
        "trajectory_smoothness": 1.0 / (1.0 + float(motion["jerk_difference"])) if isinstance(motion["jerk_difference"], (int, float)) else NOT_AVAILABLE,
        "speed_consistency": 1.0 / (1.0 + float(motion["speed_mae"])) if isinstance(motion["speed_mae"], (int, float)) else NOT_AVAILABLE,
        "direction_consistency": motion["motion_direction_agreement"],
        "matched_subset_ade": geometry["bev_center_ade"],
    }

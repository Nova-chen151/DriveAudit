from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def compute_worldlens_metrics(formal: dict[str, Any], act: dict[str, Any], drivinggen: dict[str, Any]) -> dict[str, Any]:
    reprojection = formal["reprojection"]
    camera = [row.get("projected_box_valid_ratio") for row in reprojection.get("per_camera", {}).values()]
    camera = [float(value) for value in camera if isinstance(value, (int, float))]
    return {
        "reference_semantics": "submitted_trajectory_as_instruction_reference",
        "worldlens_temporal_object_consistency_adapted": drivinggen["trajectory_consistency"],
        "worldlens_cross_view_object_support_adapted": float(np.mean(camera)) if camera else reprojection.get("cross_view_support_consistency", NOT_AVAILABLE),
        "worldlens_object_persistence_adapted": formal["presence"]["object_survival_consistency"],
        "worldlens_geometric_trajectory_consistency_adapted": drivinggen["trajectory_alignment"],
        "worldlens_behavior_consistency_adapted": act["act_agent_motion_accuracy_adapted"],
        "worldlens_submission_action_following_adapted": act["act_agent_motion_accuracy_adapted"],
        "implementation_level": "CONCEPTUAL_ADAPTATION",
        "standard_action_following": NOT_AVAILABLE, "standard_downstream_task_gap": NOT_AVAILABLE,
        "unavailable_reason": "WorldLens official action/downstream protocols require paired data and different evaluators",
    }


from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import mean


def direction_agreement(a: list[float], b: list[float], minimum_motion: float = .5) -> float | str:
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if max(na, nb) < minimum_motion: return 1.0
    if min(na, nb) < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_image_motion_metrics(expected_motion: list[dict[str, Any]], visual: list[dict[str, Any]], minimum_motion: float = .5) -> dict[str, Any]:
    actual = {(row["agent_id"], row["camera_name"], row["frame_index"]): row for row in visual if row["visible"]}
    rows = []
    for expected in expected_motion:
        if not expected["history_anchored"] or not expected["expected_visible"]: continue
        observation = actual.get((expected["agent_id"], expected["camera_name"], expected["frame_index"]))
        if not observation or not observation["flow_vectors"]: continue
        observed = np.median(np.asarray(observation["flow_vectors"], float), axis=0); target = np.asarray(expected["expected_center_flow"], float)
        epe = float(np.linalg.norm(observed - target)); magnitude = float(np.linalg.norm(target)); agreement = direction_agreement(observed.tolist(), target.tolist(), minimum_motion)
        rows.append({"agent_id": expected["agent_id"], "camera_name": expected["camera_name"], "frame_index": expected["frame_index"],
                     "flow_endpoint_error_px": epe, "normalized_flow_endpoint_error": epe / max(1.0, magnitude),
                     "motion_direction_agreement": agreement, "motion_magnitude_relative_error": abs(np.linalg.norm(observed) - magnitude) / max(1.0, magnitude),
                     "turning_direction_agreement": agreement, "scale_change_agreement": NOT_AVAILABLE,
                     "stop_move_state_accuracy": float((np.linalg.norm(observed) < minimum_motion) == (magnitude < minimum_motion)),
                     "optical_flow_cosine_similarity": agreement, "actual_flow": observed.tolist(), "expected_flow": target.tolist()})
    return {"flow_endpoint_error_px": mean(row["flow_endpoint_error_px"] for row in rows),
            "normalized_flow_endpoint_error": mean(row["normalized_flow_endpoint_error"] for row in rows),
            "motion_direction_agreement": mean(row["motion_direction_agreement"] for row in rows),
            "motion_magnitude_relative_error": mean(row["motion_magnitude_relative_error"] for row in rows),
            "turning_direction_agreement": mean(row["turning_direction_agreement"] for row in rows),
            "scale_change_agreement": NOT_AVAILABLE, "stop_move_state_accuracy": mean(row["stop_move_state_accuracy"] for row in rows),
            "acceleration_trend_agreement": NOT_AVAILABLE, "reversal_error": mean(float(row["motion_direction_agreement"] < 0) for row in rows),
            "optical_flow_cosine_similarity": mean(row["optical_flow_cosine_similarity"] for row in rows),
            "matched_motion_samples": len(rows), "per_frame": rows}


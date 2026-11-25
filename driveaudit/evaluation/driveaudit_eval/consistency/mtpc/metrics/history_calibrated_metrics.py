from __future__ import annotations

from typing import Any

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE


def _gap(future: Any, history: Any, error: bool = True) -> Any:
    if not isinstance(future, (int, float)) or not isinstance(history, (int, float)): return NOT_AVAILABLE
    return max(0.0, future - history) if error else history - future


def compute_history_calibrated(history: dict[str, Any], geometry: dict[str, Any], motion: dict[str, Any],
                               presence: dict[str, Any], identity: dict[str, Any], multiview: dict[str, Any]) -> dict[str, Any]:
    return {"future_center_error_px": geometry["projected_center_error_px"], "history_center_error_px": history["history_center_tracking_error_px"],
            "excess_center_error_px": _gap(geometry["projected_center_error_px"], history["history_center_tracking_error_px"]),
            "future_flow_error_px": motion["flow_endpoint_error_px"], "history_flow_error_px": history["history_flow_error_px"],
            "excess_flow_error_px": _gap(motion["flow_endpoint_error_px"], history["history_flow_error_px"]),
            "future_support_rate": presence["persistent_agent_visual_support_rate"], "history_support_rate": history["history_identity_persistence"],
            "support_drop": _gap(presence["persistent_agent_visual_support_rate"], history["history_identity_persistence"], False),
            "future_identity_persistence": identity["identity_persistence"], "history_identity_persistence": history["history_identity_persistence"],
            "identity_drop": _gap(identity["identity_persistence"], history["history_identity_persistence"], False),
            "future_cross_view_consistency": multiview["camera_support_recall"], "history_cross_view_consistency": history["history_cross_view_consistency"],
            "cross_view_drop": _gap(multiview["camera_support_recall"], history["history_cross_view_consistency"], False),
            "interpretation_caution": "history-to-future change mixes real/generated domain, scene difficulty, distance, occlusion, and image-trajectory consistency"}


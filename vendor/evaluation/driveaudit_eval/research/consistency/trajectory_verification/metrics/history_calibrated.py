from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE


def _delta(future: Any, history: Any, nonnegative: bool = False) -> float | str:
    if not isinstance(future, (int, float)) or not isinstance(history, (int, float)):
        return NOT_AVAILABLE
    value = float(future) - float(history)
    return max(0.0, value) if nonnegative else value


def _ratio(future: Any, history: Any) -> float | str:
    return float(future) / (float(history) + 1e-9) if isinstance(future, (int, float)) and isinstance(history, (int, float)) else NOT_AVAILABLE


def history_calibrated_metrics(history: dict[str, Any], future: dict[str, Any]) -> dict[str, Any]:
    hp, hm, hg, hv = (history[key] for key in ("presence", "local_motion", "projection_geometry", "multiview"))
    fp, fm, fg, fv = (future[key] for key in ("presence", "local_motion", "projection_geometry", "multiview"))
    return {"history_support_rate": hp["projection_support_rate"], "future_support_rate": fp["projection_support_rate"],
            "support_retention": _ratio(fp["projection_support_rate"], hp["projection_support_rate"]),
            "support_drop": _delta(hp["projection_support_rate"], fp["projection_support_rate"], True),
            "history_flow_epe": hm["local_flow_epe"], "future_flow_epe": fm["local_flow_epe"],
            "excess_local_flow_epe": _delta(fm["local_flow_epe"], hm["local_flow_epe"], True),
            "history_flow_cosine": hm["flow_cosine_similarity"], "future_flow_cosine": fm["flow_cosine_similarity"],
            "direction_agreement_drop": _delta(hm["motion_direction_agreement"], fm["motion_direction_agreement"], True),
            "history_normalized_center_error": hg["normalized_center_error"],
            "future_normalized_center_error": fg["normalized_center_error"],
            "excess_normalized_center_error": _delta(fg["normalized_center_error"], hg["normalized_center_error"], True),
            "history_multiview_support": hv["camera_support_recall"],
            "future_multiview_support": fv["camera_support_recall"],
            "multiview_support_drop": _delta(hv["camera_support_recall"], fv["camera_support_recall"], True),
            "interpretation_caution": "history-to-future change mixes real/generated domain, difficulty, occlusion, and image-trajectory inconsistency"}


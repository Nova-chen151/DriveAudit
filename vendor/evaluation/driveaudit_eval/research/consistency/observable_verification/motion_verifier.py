from __future__ import annotations

from typing import Any

import numpy as np


def verify(geometry: dict[str, Any], observed: dict[str, Any] | None, competition: dict[str, Any] | None,
           contamination: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    status = geometry["geometry_observability_status"]
    if status != "geometry_observable": return _result("geometry_unobservable", status)
    if observed is None or observed.get("image_quality_status", "unscorable").startswith("unscorable"):
        return _result("visually_unscorable", "image_quality_or_missing_visual_evidence")
    contamination = contamination or {}
    if float(contamination.get("contamination_probability", 0)) >= float(config["motion"]["maximum_contamination_probability"]):
        return _result("visually_unscorable_contamination", "core_or_neighbor_contamination")
    competition = competition or {}; expected = np.asarray(competition.get("target_expected_flow_per_point", []), float)
    actual = np.asarray(observed.get("observed_flow_vectors", []), float)
    if not len(actual) or competition.get("target_hypothesis_status") != "valid": return _result("visually_unscorable", "insufficient_points_or_projection")
    expected_med = np.median(expected, axis=0); actual_med = np.median(actual, axis=0)
    denom = np.linalg.norm(expected_med)*np.linalg.norm(actual_med)
    cosine = float(np.dot(expected_med, actual_med)/denom) if denom > 1e-6 else 0.0
    margin = float(competition["target_hypothesis_margin"]); inliers = float(competition["target_inlier_ratio"])
    if margin >= float(config["motion"]["support_margin"]) and inliers >= float(config["motion"]["minimum_target_inlier_ratio"]):
        result = _result("observable_supported", "target_hypothesis_wins")
    elif margin <= -float(config["motion"]["contradiction_margin"]): result = _result("observable_contradicted", competition["winning_hypothesis"]+"_wins")
    else: result = _result("observable_ambiguous", "hypothesis_margin_insufficient")
    return {**result, "target_observed_cosine": cosine}


def _result(state: str, reason: str) -> dict[str, Any]:
    return {"motion_state": state, "motion_reason": reason,
            "geometry_observable": state != "geometry_unobservable",
            "visually_scorable": state in {"observable_supported", "observable_contradicted", "observable_ambiguous"},
            "observable_supported": state == "observable_supported", "observable_contradicted": state == "observable_contradicted",
            "observable_ambiguous": state == "observable_ambiguous",
            "geometry_unobservable_not_penalized": state == "geometry_unobservable"}

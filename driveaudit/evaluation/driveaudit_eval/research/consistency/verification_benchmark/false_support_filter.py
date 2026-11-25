from __future__ import annotations

import math
from typing import Any

import numpy as np


def rule_verify(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rule = config["verifiers"]["rule_based"]
    expected_relative = float(np.linalg.norm(row.get("expected_relative_flow", [0.0, 0.0])))
    residual_epe = float(row.get("residual_flow_epe", 99.0)); residual_cosine = float(row.get("residual_flow_cosine", -1.0))
    relative_needed = min(1.0, expected_relative / float(rule["relative_motion_scale_px"]))
    direction_score = max(0.0, (residual_cosine + 1.0) / 2.0) if relative_needed > .2 else 1.0
    residual_agreement = math.exp(-residual_epe / float(rule["residual_epe_scale_px"])) * direction_score
    separation = min(1.0, float(row.get("foreground_background_motion_separation", 0.0)) / float(rule["separation_scale_px"]))
    separation_evidence = relative_needed * separation + (1.0 - relative_needed) * math.exp(-residual_epe / float(rule["residual_epe_scale_px"]))
    leakage_bad = relative_needed * min(1.0, float(row.get("background_flow_leakage_ratio", 1.0)))
    fixed = {"sparse": float(row.get("point_support_score", 0.0)),
             "fb_ransac": math.exp(-float(row.get("forward_backward_error") or 9.0) / 2.5) * float(row.get("ransac_inlier_ratio", 0.0)),
             "residual": residual_agreement, "separation": separation_evidence,
             "appearance": max(0.0, float(row.get("foreground_contrast_score", 0.0))),
             "feature": float(row.get("feature_support_score", 0.0)), "scale": math.exp(-4.0 * float(row.get("scale_change_error", 1.0)))}
    weights = rule["weights"]; raw = sum(float(weights[key]) * fixed[key] for key in fixed) / sum(float(weights[key]) for key in fixed)
    score = float(np.clip(raw * (1.0 - float(rule["leakage_penalty"]) * leakage_bad), 0.0, 1.0))
    reasons = []
    if not row.get("background_ring_valid"): reasons.append("insufficient_background_ring")
    if relative_needed > .2 and residual_cosine < float(rule["minimum_residual_cosine"]): reasons.append("residual_direction_mismatch")
    if relative_needed > .5 and separation < float(rule["minimum_separation"]): reasons.append("camera_motion_only")
    if leakage_bad > float(rule["maximum_leakage"]): reasons.append("background_flow_leakage")
    if int(row.get("valid_points", 0)) < int(rule["minimum_valid_points"]): reasons.append("insufficient_sparse_points")
    decision = score >= float(rule["support_threshold"]) and not any(reason in reasons for reason in
        ("residual_direction_mismatch", "camera_motion_only", "insufficient_sparse_points"))
    return {**row, "raw_support_score": float(row.get("support_confidence", 0.0)),
            "background_compensated_support_score": score, "false_support_probability": 1.0 - score,
            "tcvv_rule_support": score, "tcvv_rule_motion": residual_agreement,
            "tcvv_rule_multiview": float(row.get("expected_camera_count", 1) > 0),
            "support_decision": bool(decision), "rejection_reason": reasons or ["none"], "rule_components": fixed}


def filter_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule_verify(row, config) for row in rows]

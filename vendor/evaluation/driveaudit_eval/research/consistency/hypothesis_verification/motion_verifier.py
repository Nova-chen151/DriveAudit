from __future__ import annotations

from typing import Any
import numpy as np


def verify(observed:dict[str,Any],competition:dict[str,Any],presence:dict[str,Any],contamination:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    rules=[];expected=np.asarray(competition["target_expected_flow_per_point"],float);actual=np.asarray(observed["observed_flow_vectors"],float)
    expected_magnitude=float(np.median(np.linalg.norm(expected,axis=1))) if len(expected) else 0.0
    observed_magnitude=float(np.median(np.linalg.norm(actual,axis=1))) if len(actual) else 0.0
    cosine=_cos(np.median(actual,axis=0) if len(actual) else np.zeros(2),np.median(expected,axis=0) if len(expected) else np.zeros(2))
    if observed["image_quality_status"].startswith("unscorable"):return _result("motion_unscorable","image_quality",rules,competition,cosine,expected_magnitude,observed_magnitude)
    if presence["presence_state"] in {"presence_contradicted","presence_unscorable"}:return _result("motion_unscorable","presence_not_scorable",rules,competition,cosine,expected_magnitude,observed_magnitude)
    if contamination["contamination_probability"]>=float(config["maximum_contamination_probability"]):
        rules.append("core_or_neighbor_contamination");return _result("motion_unscorable","contamination",rules,competition,cosine,expected_magnitude,observed_magnitude)
    if len(actual)<int(config["minimum_points"]) or competition["target_hypothesis_status"]!="valid":return _result("motion_unscorable","insufficient_points_or_projection",rules,competition,cosine,expected_magnitude,observed_magnitude)
    if expected_magnitude>=float(config["minimum_expected_motion_px"]) and competition["zero_motion_likelihood"]>competition["target_likelihood"]+float(config["hard_margin"]):rules.append("static_or_frozen")
    if competition["best_background_likelihood"]>competition["target_likelihood"]+float(config["hard_margin"]):rules.append("background_explains_better")
    if expected_magnitude>=float(config["minimum_expected_motion_px"]) and cosine<float(config["reverse_cosine_threshold"]):rules.append("reverse_direction")
    if expected_magnitude<float(config["static_expected_threshold_px"]) and observed_magnitude>float(config["unexpected_motion_threshold_px"]) and competition["other_motion_likelihood"]>competition["best_background_likelihood"]:rules.append("unexpected_motion")
    if rules:return _result("motion_contradicted","hard_rule:"+rules[0],rules,competition,cosine,expected_magnitude,observed_magnitude)
    if competition["target_hypothesis_margin"]>=float(config["support_margin"]) and competition["target_inlier_ratio"]>=float(config["minimum_target_inlier_ratio"]):state,reason="motion_supported","target_hypothesis_wins"
    elif competition["target_hypothesis_margin"]<=-float(config["contradiction_margin"]):state,reason="motion_contradicted",competition["winning_hypothesis"]+"_wins"
    else:state,reason="motion_ambiguous","hypothesis_margin_insufficient"
    return _result(state,reason,rules,competition,cosine,expected_magnitude,observed_magnitude)


def _cos(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/norm) if norm>1e-6 else float(norm<1e-6)


def _result(state,reason,rules,competition,cosine,expected,observed):
    return {"motion_state":state,"motion_reason":reason,"hard_rule_triggers":rules,"motion_scorable":state not in {"motion_unscorable"},
            "motion_supported":state=="motion_supported","motion_contradicted":state=="motion_contradicted",
            "motion_ambiguous":state=="motion_ambiguous","motion_unscorable":state=="motion_unscorable",
            "target_observed_cosine":cosine,"expected_motion_magnitude":expected,"observed_motion_magnitude":observed,
            "static_contradiction":"static_or_frozen" in rules,"background_contradiction":"background_explains_better" in rules,
            "reverse_direction_contradiction":"reverse_direction" in rules}


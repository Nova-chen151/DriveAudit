from __future__ import annotations

import math

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.common import mean_available


def _clip(value):
    return float(np.clip(value, 0.0, 100.0))


def _mean_score(values):
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(values)) if values else NOT_AVAILABLE


def _absolute_scores(schema_errors, validity, kinematics, map_rules, collisions, interactions):
    validity_agents = list(validity.get("per_agent", {}).values())
    timestamp_errors = sum(
        item["timestamp_error"].get("non_increasing_count", 0) + item["timestamp_error"].get("duplicate_count", 0)
        for item in validity_agents
    )
    continuity_penalties = [np.mean([
        item.get("continuity_gap_ratio", 0), item.get("teleportation_rate", 0),
        item.get("reverse_motion_ratio", 0), item.get("stationary_collapse_ratio", 0),
    ]) for item in validity_agents]
    violation_ratios = [
        metric.get("violation_ratio", 0)
        for agent in kinematics.values()
        for name in ("speed", "longitudinal_acceleration", "lateral_acceleration", "longitudinal_jerk",
                     "lateral_jerk", "yaw_rate", "yaw_acceleration", "curvature")
        for metric in [agent.get(name)] if isinstance(metric, dict)
    ]
    map_score = NOT_AVAILABLE
    if map_rules.get("status") == "available":
        map_score = _clip(100 * (1 - np.mean([map_rules.get("off_road_frame_ratio", 0), map_rules.get("wrong_way_ratio", 0)])))
    collision_score = _clip(100 * (1 - max(collisions.get("collision_frame_ratio", 0), collisions.get("collision_agent_pair_ratio", 0))))
    ttc_risk = interactions.get("ttc_below_1s_ratio", 0) * 0.5 + interactions.get("ttc_below_2s_ratio", 0) * 0.3
    return {
        "submission_format_validity": 0.0 if schema_errors or timestamp_errors else 100.0,
        "trajectory_continuity": _clip(100 * (1 - (float(np.mean(continuity_penalties)) if continuity_penalties else 1))),
        "kinematic_feasibility": _clip(100 * (1 - (float(np.mean(violation_ratios)) if violation_ratios else 1))),
        "map_rule_compliance": map_score,
        "collision_safety": collision_score,
        "interaction_realism": _clip(100 * (1 - min(1.0, ttc_risk + (0.5 if collisions.get("collision_any") else 0)))),
    }


def _relative_scores(availability, paired, map_baseline, kinematic_baseline, risk_baseline, config):
    temporal = _clip(100 * np.mean([
        availability.get("mean_valid_mask_precision", 0), availability.get("mean_valid_mask_recall", 0),
        availability.get("mean_valid_mask_iou", 0), 1 - min(1.0, availability.get("mean_unexplained_missing_ratio", 1)),
    ]))
    ades = [value.get("ADE") for value in paired.values() if isinstance(value, dict) and isinstance(value.get("ADE"), (int, float))]
    geometry = _clip(100 * math.exp(-float(np.mean(ades)) / float(config.get("reference_ade_scale_m", 5.0)))) if ades else NOT_AVAILABLE
    map_fidelity = NOT_AVAILABLE
    if map_baseline.get("status") == "available":
        map_fidelity = _clip(100 * (1 - map_baseline.get("new_offroad_ratio", 0) - max(0.0, map_baseline.get("offroad_gap", 0))))
    kine_gaps = [
        max(0.0, metric.get("violation_ratio_gap", 0), metric.get("maximum_exceedance_gap", 0) / max(1.0, metric.get("generated_maximum_exceedance", 0)))
        for agent in kinematic_baseline.get("per_agent", {}).values()
        for metric in agent.values() if isinstance(metric, dict)
    ]
    kine_fidelity = _clip(100 * (1 - float(np.mean(kine_gaps)))) if kine_gaps else NOT_AVAILABLE
    risk_penalty = 0.0
    amplification = risk_baseline.get("risk_amplification_ratio")
    if isinstance(amplification, (int, float)):
        risk_penalty += max(0.0, amplification - 1.0)
    gap = risk_baseline.get("ttc_gap")
    if isinstance(gap, (int, float)):
        risk_penalty += max(0.0, -gap) / 3.0
    risk_fidelity = _clip(100 * (1 - min(1.0, risk_penalty)))
    reference = _mean_score([temporal, geometry, map_fidelity, kine_fidelity, risk_fidelity])
    return {
        "temporal_availability_fidelity": temporal,
        "trajectory_geometry_fidelity": geometry,
        "map_fidelity": map_fidelity,
        "kinematic_fidelity": kine_fidelity,
        "risk_fidelity": risk_fidelity,
        "reference_fidelity": reference,
    }


def score_scene(schema_errors, validity, kinematics, map_rules, collisions, interactions, paired, config,
                protocol="A", availability=None, map_baseline=None, kinematic_baseline=None, risk_baseline=None):
    absolute = _absolute_scores(schema_errors, validity, kinematics, map_rules, collisions, interactions)
    if protocol == "B":
        relative = _relative_scores(availability or {}, paired, map_baseline or {}, kinematic_baseline or {}, risk_baseline or {}, config)
        temporal = relative["temporal_availability_fidelity"]
        reference = relative["reference_fidelity"]
    else:
        valid_ratio = mean_available(item.get("valid_point_ratio") for item in validity.get("per_agent", {}).values())
        temporal = _clip(100 * valid_ratio) if isinstance(valid_ratio, float) else NOT_AVAILABLE
        relative = {name: NOT_AVAILABLE for name in (
            "temporal_availability_fidelity", "trajectory_geometry_fidelity", "map_fidelity",
            "kinematic_fidelity", "risk_fidelity", "reference_fidelity",
        )}
        reference = NOT_AVAILABLE
    scores = dict(absolute, temporal_availability_fidelity=temporal, reference_fidelity=reference)
    scorecards = {
        "Absolute Scene Quality": {
            "dimensions": {key: absolute[key] for key in (
                "trajectory_continuity", "kinematic_feasibility", "map_rule_compliance", "collision_safety", "interaction_realism")},
            "score": _mean_score(absolute[key] for key in (
                "trajectory_continuity", "kinematic_feasibility", "map_rule_compliance", "collision_safety", "interaction_realism")),
        },
        "Reference-relative Fidelity": {"dimensions": relative, "score": reference},
    }
    if config.get("enable_provisional_total", False):
        weighted = [(scores[name], float(weight)) for name, weight in config.get("weights", {}).items()
                    if isinstance(scores.get(name), (int, float)) and float(weight) > 0]
        scores["provisional_total_score"] = (
            float(sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted))
            if weighted else NOT_AVAILABLE
        )
    return scores, scorecards


def _physical_reasons(kinematics, config, prefix=""):
    reasons = []
    limit = float(config["severe_violation_ratio_max"])
    duration_limit = float(config.get("max_consecutive_violation_seconds_max", 0.5))
    extreme_factor = float(config.get("extreme_max_threshold_multiplier", 3.0))
    for agent_id, metrics in kinematics.items():
        thresholds = metrics.get("thresholds", {})
        for name, key in (
            ("longitudinal_acceleration", "longitudinal_acc_max"), ("lateral_acceleration", "lateral_acc_max"),
            ("longitudinal_jerk", "longitudinal_jerk_max"), ("lateral_jerk", "lateral_jerk_max"),
            ("yaw_rate", "yaw_rate_max"), ("curvature", "curvature_max"),
        ):
            value = metrics.get(name)
            if not isinstance(value, dict):
                continue
            if value.get("violation_ratio", 0) > limit:
                reasons.append(f"{prefix}{agent_id} {name} violation_ratio {value['violation_ratio']:.6f} > {limit:.6f}")
            if value.get("max_consecutive_violation_seconds", 0) > duration_limit:
                reasons.append(f"{prefix}{agent_id} {name} maximum continuous violation exceeds {duration_limit:.3f}s")
            threshold = thresholds.get(key)
            extreme = max(abs(value.get("min", 0)), abs(value.get("max", 0)))
            if threshold and extreme > threshold * extreme_factor:
                reasons.append(f"{prefix}{agent_id} {name} extreme maximum {extreme:.6f} > {extreme_factor:.1f}x threshold")
    return reasons


def _absolute_reasons(schema_errors, validity, kinematics, map_rules, collisions, interactions, config, protocol, prefix=""):
    reasons = [prefix + reason for reason in schema_errors]
    trajectory_cfg = config["trajectory"]
    for agent_id, item in validity.get("per_agent", {}).items():
        if protocol == "A" and item.get("valid_point_ratio", 0) < trajectory_cfg["valid_point_ratio_min"]:
            reasons.append(f"{prefix}{agent_id} absolute valid_point_ratio below threshold without explicit termination")
        for metric, threshold_name in (("continuity_gap_ratio", "continuity_gap_ratio_max"),
                                       ("teleportation_rate", "teleportation_rate_max"),
                                       ("reverse_motion_ratio", "reverse_motion_ratio_max")):
            if item.get(metric, 0) > trajectory_cfg[threshold_name]:
                reasons.append(f"{prefix}{agent_id} {metric} exceeds threshold")
    reasons.extend(_physical_reasons(kinematics, config["physical"], prefix))
    scene_cfg = config["scene"]
    collision_limit = scene_cfg.get("collision_count_max", scene_cfg.get("background_collision_count_max", 0))
    if collisions.get("collision_count", 0) > collision_limit:
        reasons.append(prefix + "confirmed collision events exceed threshold")
    if map_rules.get("status") == "available":
        if map_rules.get("off_road_frame_ratio", 0) > scene_cfg["offroad_frame_ratio_max"]:
            reasons.append(prefix + "absolute off_road_frame_ratio exceeds threshold")
        if map_rules.get("wrong_way_ratio", 0) > scene_cfg["wrong_way_ratio_max"]:
            reasons.append(prefix + "absolute wrong_way_ratio exceeds threshold")
    return reasons


def hard_gates(schema_errors, validity, kinematics, map_rules, collisions, interactions, config, protocol="A",
               availability=None, map_baseline=None, kinematic_baseline=None, risk_baseline=None,
               paired=None, gt_validity=None, gt_kinematics=None, gt_map=None, gt_collisions=None, gt_interactions=None):
    absolute = _absolute_reasons(schema_errors, validity, kinematics, map_rules, collisions, interactions, config, protocol)
    relative = list(schema_errors)
    if protocol == "B":
        rel_cfg = config.get("reference_relative", {})
        for agent_id, item in (availability or {}).get("per_agent", {}).items():
            if item["valid_mask_precision"] < rel_cfg.get("valid_mask_precision_min", 0.99):
                relative.append(f"{agent_id} valid_mask_precision degraded")
            if item["valid_mask_recall"] < rel_cfg.get("valid_mask_recall_min", 0.99):
                relative.append(f"{agent_id} valid_mask_recall degraded")
            if item["valid_mask_iou"] < rel_cfg.get("valid_mask_iou_min", 0.99):
                relative.append(f"{agent_id} valid_mask_iou degraded")
        if isinstance((map_baseline or {}).get("new_offroad_ratio"), (int, float)):
            if map_baseline["new_offroad_ratio"] > rel_cfg.get("new_offroad_ratio_max", 0.01):
                relative.append("new_offroad_ratio exceeds relative threshold")
            if map_baseline["offroad_gap"] > rel_cfg.get("offroad_gap_max", 0.01):
                relative.append("offroad_gap exceeds relative threshold")
        for agent_id, variables in (kinematic_baseline or {}).get("per_agent", {}).items():
            for name, item in variables.items():
                if isinstance(item, dict) and item.get("violation_ratio_gap", 0) > rel_cfg.get("kinematic_violation_ratio_gap_max", 0.01):
                    relative.append(f"{agent_id} {name} violation_ratio added relative to GT")
                if isinstance(item, dict) and item.get("maximum_exceedance_gap", 0) > rel_cfg.get("kinematic_maximum_exceedance_gap_max", 1.0):
                    relative.append(f"{agent_id} {name} extreme exceedance added relative to GT")
        amplification = (risk_baseline or {}).get("risk_amplification_ratio")
        if isinstance(amplification, (int, float)) and amplification > rel_cfg.get("risk_amplification_ratio_max", 1.10):
            relative.append("TTC exposure amplified relative to GT")
        ttc_gap = (risk_baseline or {}).get("ttc_gap")
        if isinstance(ttc_gap, (int, float)) and ttc_gap < rel_cfg.get("minimum_ttc_gap_min", -0.1):
            relative.append("minimum TTC degraded relative to GT")
        if gt_collisions and collisions.get("collision_count", 0) > gt_collisions.get("collision_count", 0):
            relative.append("new collision events relative to GT")
        for agent_id, item in (paired or {}).items():
            if isinstance(item, dict) and item.get("ADE", 0) > rel_cfg.get("ade_max", 5.0):
                relative.append(f"{agent_id} ADE exceeds relative threshold")
    else:
        relative = []

    warnings = []
    if protocol == "B":
        warnings = _absolute_reasons([], gt_validity or {}, gt_kinematics or {}, gt_map or {},
                                     gt_collisions or {}, gt_interactions or {}, config, "A", "reference: ")
        warning_ttc = (gt_interactions or {}).get("minimum_ttc")
        if isinstance(warning_ttc, (int, float)) and warning_ttc < config.get("reference_warning", {}).get("minimum_ttc_seconds", 3.0):
            warnings.append(f"reference: high-risk collision-free minimum TTC {warning_ttc:.6f}s")
    gates = {
        "Absolute Quality Gate": {
            "passed": not absolute, "blocking": protocol == "A" or bool(schema_errors),
            "reasons": absolute or ["no supported absolute quality failure"],
        },
        "Reference-relative Degradation Gate": {
            "passed": not relative if protocol == "B" else NOT_AVAILABLE, "blocking": protocol == "B",
            "reasons": relative or (["no generated degradation relative to GT"] if protocol == "B" else ["Protocol A has no reference baseline"]),
        },
        "Reference Quality Warning": {
            "status": "warning" if warnings else ("clear" if protocol == "B" else NOT_AVAILABLE),
            "blocking": False,
            "reasons": warnings or (["no supported reference quality warning"] if protocol == "B" else ["Protocol A has no reference"]),
        },
    }
    failures = []
    if schema_errors or (protocol == "A" and absolute):
        failures.append("Absolute Quality Gate")
    if protocol == "B" and relative:
        failures.append("Reference-relative Degradation Gate")
    return gates, failures

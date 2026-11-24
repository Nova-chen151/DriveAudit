from __future__ import annotations

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.common import stats, wrap_angle


def kinematic_series(track):
    n = len(track.timestamps)
    result = {name: np.full(n, np.nan) for name in (
        "speed", "longitudinal_velocity", "lateral_velocity", "longitudinal_acceleration",
        "lateral_acceleration", "longitudinal_jerk", "lateral_jerk", "yaw_rate",
        "yaw_acceleration", "curvature", "curvature_rate", "forward_speed"
    )}
    valid = track.valid_mask
    t = track.timestamps
    p = track.positions
    h = track.headings
    pairs = np.flatnonzero(valid[1:] & valid[:-1]) + 1
    if len(pairs):
        dt = t[pairs] - t[pairs - 1]
        velocity = (p[pairs] - p[pairs - 1]) / dt[:, None]
        cos_h, sin_h = np.cos(h[pairs]), np.sin(h[pairs])
        result["speed"][pairs] = np.linalg.norm(velocity, axis=1)
        result["longitudinal_velocity"][pairs] = velocity[:, 0] * cos_h + velocity[:, 1] * sin_h
        result["forward_speed"][pairs] = result["longitudinal_velocity"][pairs]
        result["lateral_velocity"][pairs] = -velocity[:, 0] * sin_h + velocity[:, 1] * cos_h
        result["yaw_rate"][pairs] = wrap_angle(h[pairs] - h[pairs - 1]) / dt
    triples = np.flatnonzero(valid[2:] & valid[1:-1] & valid[:-2]) + 2
    if len(triples):
        dt = t[triples] - t[triples - 1]
        usable = np.isfinite(result["longitudinal_velocity"][triples - 1])
        idx = triples[usable]
        dt = dt[usable]
        result["longitudinal_acceleration"][idx] = (
            result["longitudinal_velocity"][idx] - result["longitudinal_velocity"][idx - 1]
        ) / dt
        result["lateral_acceleration"][idx] = (
            result["lateral_velocity"][idx] - result["lateral_velocity"][idx - 1]
        ) / dt
        result["yaw_acceleration"][idx] = (result["yaw_rate"][idx] - result["yaw_rate"][idx - 1]) / dt
    quadruples = np.flatnonzero(valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3]) + 3
    if len(quadruples):
        dt = t[quadruples] - t[quadruples - 1]
        usable = np.isfinite(result["longitudinal_acceleration"][quadruples - 1])
        idx, dt = quadruples[usable], dt[usable]
        result["longitudinal_jerk"][idx] = (
            result["longitudinal_acceleration"][idx] - result["longitudinal_acceleration"][idx - 1]
        ) / dt
        result["lateral_jerk"][idx] = (
            result["lateral_acceleration"][idx] - result["lateral_acceleration"][idx - 1]
        ) / dt
    speed_ok = np.isfinite(result["yaw_rate"]) & (np.abs(result["forward_speed"]) > 0.1)
    result["curvature"][speed_ok] = result["yaw_rate"][speed_ok] / result["forward_speed"][speed_ok]
    curve_pairs = np.flatnonzero(valid[1:] & valid[:-1] & np.isfinite(result["curvature"][1:]) & np.isfinite(result["curvature"][:-1])) + 1
    if len(curve_pairs):
        result["curvature_rate"][curve_pairs] = (
            result["curvature"][curve_pairs] - result["curvature"][curve_pairs - 1]
        ) / (t[curve_pairs] - t[curve_pairs - 1])
    result["timestamps"] = t.copy()
    return result


def _violation(name, values, limits):
    if name == "speed":
        return values > limits["speed_max"]
    if name == "longitudinal_acceleration":
        return (values < limits["longitudinal_acc_min"]) | (values > limits["longitudinal_acc_max"])
    mapping = {
        "lateral_acceleration": "lateral_acc_max",
        "longitudinal_jerk": "longitudinal_jerk_max",
        "lateral_jerk": "lateral_jerk_max",
        "yaw_rate": "yaw_rate_max",
        "yaw_acceleration": "yaw_acceleration_max",
        "curvature": "curvature_max",
        "curvature_rate": "curvature_rate_max",
    }
    return np.abs(values) > limits[mapping[name]] if name in mapping and mapping[name] in limits else np.zeros(len(values), dtype=bool)


def evaluate_track_kinematics(track, config):
    series = kinematic_series(track)
    limits = config[track.agent_type]
    start = int(track.metadata.get("history_frames") or 0)
    output = {}
    for name in (
        "speed", "longitudinal_velocity", "lateral_velocity", "longitudinal_acceleration",
        "lateral_acceleration", "longitudinal_jerk", "lateral_jerk", "yaw_rate",
        "yaw_acceleration", "curvature", "curvature_rate", "forward_speed"
    ):
        values = series[name][start:]
        valid = np.isfinite(values)
        violations = _violation(name, values[valid], limits)
        output[name] = stats(values[valid], violations, series["timestamps"][start:][valid])
    output["series"] = {name: values.tolist() for name, values in series.items()}
    output["thresholds"] = dict(limits)
    return output


def evaluate_scene_kinematics(tracks, config):
    return {track.agent_id: evaluate_track_kinematics(track, config) for track in tracks}


def _maximum_exceedance(name, metric, limits):
    if not isinstance(metric, dict):
        return 0.0
    if name == "speed":
        return max(0.0, metric["max"] - limits["speed_max"])
    if name == "longitudinal_acceleration":
        return max(0.0, metric["max"] - limits["longitudinal_acc_max"], limits["longitudinal_acc_min"] - metric["min"])
    key = {
        "lateral_acceleration": "lateral_acc_max", "longitudinal_jerk": "longitudinal_jerk_max",
        "lateral_jerk": "lateral_jerk_max", "yaw_rate": "yaw_rate_max",
        "yaw_acceleration": "yaw_acceleration_max", "curvature": "curvature_max",
        "curvature_rate": "curvature_rate_max",
    }.get(name)
    return max(0.0, max(abs(metric["min"]), abs(metric["max"])) - limits[key]) if key else 0.0


def compare_kinematic_baseline(generated, ground_truth):
    per_agent = {}
    variables = (
        "speed", "longitudinal_acceleration", "lateral_acceleration", "longitudinal_jerk",
        "lateral_jerk", "yaw_rate", "yaw_acceleration", "curvature", "curvature_rate",
    )
    for agent_id in sorted(set(generated) & set(ground_truth)):
        a, b = generated[agent_id], ground_truth[agent_id]
        limits = a.get("thresholds", {})
        per_agent[agent_id] = {}
        for name in variables:
            ga, gb = a.get(name), b.get(name)
            if not isinstance(ga, dict) or not isinstance(gb, dict):
                per_agent[agent_id][name] = NOT_AVAILABLE
                continue
            gaps = {key: float(ga[key] - gb[key]) for key in ("mean", "P90", "P99", "max")}
            per_agent[agent_id][name] = {
                **{"generated_" + key: ga[key] for key in ("mean", "P90", "P99", "max")},
                **{"gt_" + key: gb[key] for key in ("mean", "P90", "P99", "max")},
                "distribution_gap": gaps,
                "generated_violation_ratio": ga.get("violation_ratio", 0.0),
                "gt_violation_ratio": gb.get("violation_ratio", 0.0),
                "violation_ratio_gap": float(ga.get("violation_ratio", 0.0) - gb.get("violation_ratio", 0.0)),
                "generated_maximum_exceedance": _maximum_exceedance(name, ga, limits),
                "gt_maximum_exceedance": _maximum_exceedance(name, gb, limits),
                "maximum_exceedance": _maximum_exceedance(name, ga, limits),
                "maximum_exceedance_gap": float(_maximum_exceedance(name, ga, limits) - _maximum_exceedance(name, gb, limits)),
                "generated_max_consecutive_violation_seconds": ga.get("max_consecutive_violation_seconds", 0.0),
                "gt_max_consecutive_violation_seconds": gb.get("max_consecutive_violation_seconds", 0.0),
            }
    return {"per_agent": per_agent}


def audit_peak_jerk(tracks, ground_truth, generated_metrics, ground_truth_metrics):
    track_by_id = {track.agent_id: track for track in tracks}
    gt_by_id = {track.agent_id: track for track in ground_truth}
    best = None
    for agent_id, metrics in generated_metrics.items():
        series = np.asarray(metrics.get("series", {}).get("longitudinal_jerk", []), dtype=float)
        if np.any(np.isfinite(series)):
            idx = int(np.nanargmax(np.abs(series)))
            if best is None or abs(series[idx]) > abs(best[2]):
                best = (agent_id, idx, float(series[idx]))
    if best is None:
        return NOT_AVAILABLE
    agent_id, idx, value = best
    track = track_by_id[agent_id]
    gt_value = float(ground_truth_metrics[agent_id]["series"]["longitudinal_jerk"][idx])
    valid_run_boundary = idx < 3 or not bool(np.all(track.valid_mask[max(0, idx - 3):idx + 1]))
    scalar_jerk = None
    if idx > 0 and np.isfinite(track.accelerations[idx]) and np.isfinite(track.accelerations[idx - 1]):
        scalar_jerk = float((track.accelerations[idx] - track.accelerations[idx - 1]) / (track.timestamps[idx] - track.timestamps[idx - 1]))
    return {
        "agent_id": agent_id,
        "frame_index": idx,
        "timestamp": float(track.timestamps[idx]),
        "generated_longitudinal_jerk": value,
        "gt_longitudinal_jerk": gt_value,
        "gt_has_same_spike": bool(value == gt_value),
        "at_valid_mask_boundary": valid_run_boundary,
        "computed_by_numerical_differencing": True,
        "jerk_from_longitudinal_velocity_then_acceleration": value,
        "jerk_from_submitted_scalar_acceleration": scalar_jerk if scalar_jerk is not None else NOT_AVAILABLE,
        "difference_between_methods": float(value - scalar_jerk) if scalar_jerk is not None else NOT_AVAILABLE,
        "difference_methods_consistent": bool(scalar_jerk is not None and np.isclose(value, scalar_jerk, rtol=0.01, atol=0.1)),
        "generated_and_gt_track_fields_equal": bool(
            np.array_equal(track.positions, gt_by_id[agent_id].positions)
            and np.array_equal(track.valid_mask, gt_by_id[agent_id].valid_mask)
        ),
    }

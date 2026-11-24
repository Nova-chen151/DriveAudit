from __future__ import annotations

import math

import numpy as np

from driveaudit_eval.agents.core import (EGO_LENGTH, EGO_WIDTH, comfort, dac_from_ratios, ego_to_world,
                                           line_progress, route_progress)
from driveaudit_eval.agents.open_loop_trajectory_evaluator import evaluate_open_loop_trajectory
from driveaudit_eval.utils.map_provider.geometry import trajectory_headings, wrap_angle

from driveaudit_eval.agents.score_audit.score_reconstructor import formula_matches, pdms_open_loop
from driveaudit_eval.agents.score_audit.zero_reason import classify_frame


def _number(value):
    if value in (None, "", "None", "not_available"): return None
    if isinstance(value, bool): return value
    try: return float(value)
    except (TypeError, ValueError): return value


def normalize_legacy(row):
    return {key: _number(value) for key, value in row.items()}


def evaluate_with_provider(record, frame, frame_lookup, provider, legacy_row=None, dt=0.5):
    base = evaluate_open_loop_trajectory(record, frame, frame_lookup, None, dt)
    local = np.asarray(record.get("trajectory", []), dtype=float)
    output = {"scene_id": record.get("scene_id"), "frame_id": record.get("frame_id"), "timestamp": record.get("timestamp"),
              "model": record.get("agent"), "prediction_valid": bool(base.get("valid_trajectory")),
              "trajectory_length": int(len(local)) if local.ndim == 2 else 0, "trajectory_dt": dt,
              "trajectory_coordinate": record.get("coordinate_system"), "scoreable": bool(record.get("scoreable", True)),
              "failure_reason": base.get("reason")}
    if not output["prediction_valid"]:
        first, reason, components = classify_frame(output)
        output.update({"first_zero_component": first, "zero_reason": reason, "all_zero_components": components,
                       "hard_gate_triggered": False, "map_query_status": "not_queried"})
        return output
    state = frame["model_input"]["ego_state"]; world = np.asarray(base["trajectory_world"], dtype=float)
    headings = trajectory_headings(np.vstack((np.asarray(state["position"][:2], dtype=float), world)), float(state["yaw"]))[1:]
    try:
        centers = provider.is_drivable(world); coverage = provider.footprint_coverage(world, headings, EGO_LENGTH, EGO_WIDTH)
        signed = provider.signed_distance_to_drivable(world); lane = provider.nearest_lane_center(world)
        map_status = "available"
    except Exception as exc:
        centers = coverage = signed = np.asarray([]); lane = []; map_status = "failed:%s:%s" % (type(exc).__name__, exc)
    map_dac = dac_from_ratios(coverage.tolist()) if len(coverage) else None
    center_dac = 1.0 if len(centers) and bool(np.all(centers)) else (0.0 if len(centers) else None)
    offsets = np.asarray(record.get("time_offsets") or [(i + 1) * dt for i in range(len(world))], dtype=float)
    future_ids = [int(frame["frame_id"]) + int(round(float(value) * 12.0)) for value in offsets]
    gt_world = np.asarray([frame_lookup[index]["model_input"]["ego_state"]["position"][:2] for index in future_ids], dtype=float)
    predicted_endpoint = float(np.linalg.norm(world[-1] - np.asarray(state["position"][:2])))
    gt_endpoint = float(np.linalg.norm(gt_world[-1] - np.asarray(state["position"][:2])))
    valid_steps = len(world)
    if base.get("first_collision_time") is not None:
        valid_steps = min(valid_steps, int(np.argmin(np.abs(offsets - float(base["first_collision_time"])))))
    offroad_indices = np.flatnonzero(coverage < .2) if len(coverage) else []
    if len(offroad_indices): valid_steps = min(valid_steps, int(offroad_indices[0]))
    route = np.asarray(frame["evaluation_only"]["reference_route_positions"], dtype=float)
    progress, _ = route_progress(route, np.asarray(state["position"][:2]), world, max(valid_steps, 0))
    expected = max(1e-3, line_progress(route, route[0], route[min(len(route) - 1, int(round(offsets[-1] * 12)))]) - line_progress(route, route[0], route[0]))
    ep = float(np.clip(progress / expected, 0, 1)); c = comfort(local[:, :2], dt)
    new_pdms = pdms_open_loop(base["NC"], map_dac, base["TTC_score"], c["comfort_score"], ep)
    legacy = normalize_legacy(legacy_row or {})
    output.update({"collision": base["collision_any"], "NC": base["NC"], "legacy_DAC": legacy.get("DAC"),
                   "collision_object_id": base.get("collision_object_id"), "first_collision_time": base.get("first_collision_time"),
                   "map_provider_DAC_center": center_dac, "map_provider_DAC": map_dac,
                   "map_provider_DAC_footprint": map_dac, "TTC_score": base["TTC_score"], "Comfort": c["comfort_score"],
                   "EP": ep, "legacy_EP": legacy.get("EP"), "route_progress": progress, "route_completion": ep,
                   "expected_progress": expected, "PDMS_open_loop": new_pdms, "legacy_PDMS_open_loop": legacy.get("PDMS_open_loop"),
                   "legacy_PDMS_reconstructed": pdms_open_loop(legacy.get("NC"), legacy.get("DAC"), legacy.get("TTC_score"), legacy.get("comfort_score"), legacy.get("EP")) if legacy else None,
                   "formula_match": formula_matches(legacy.get("PDMS_open_loop"), pdms_open_loop(legacy.get("NC"), legacy.get("DAC"), legacy.get("TTC_score"), legacy.get("comfort_score"), legacy.get("EP"))) if legacy else None,
                   "map_query_status": map_status, "map_location": getattr(provider, "location", None),
                   "offroad_point_ratio": float(np.mean(coverage < .99)) if len(coverage) else None,
                   "offroad_center_ratio": float(np.mean(~centers)) if len(centers) else None,
                   "minimum_drivable_ratio": float(np.min(coverage)) if len(coverage) else None,
                   "maximum_offroad_distance": float(np.max(np.maximum(0.0, -signed))) if len(signed) else None,
                   "lane_match_rate": float(np.mean([x.get("status") == "matched" for x in lane])) if lane else None,
                   "lane_direction_error_mean": float(np.mean(np.abs(wrap_angle(headings - np.asarray([x.get("heading", np.nan) for x in lane]))))) if lane else None,
                   "trajectory_endpoint_progress": predicted_endpoint, "gt_endpoint_progress": gt_endpoint,
                   "endpoint_progress_ratio": predicted_endpoint / max(gt_endpoint, 1e-6)})
    zero_view = {"prediction_valid": True, "NC": output["NC"], "DAC": map_dac, "TTC_score": output["TTC_score"],
                 "comfort_score": output["Comfort"], "EP": output["EP"], "PDMS_open_loop": output["PDMS_open_loop"],
                 "map_query_status": map_status}
    first, reason, components = classify_frame(zero_view)
    output.update({"first_zero_component": first, "zero_reason": reason, "all_zero_components": "|".join(components),
                   "hard_gate_triggered": first in {"NC", "DAC"}})
    legacy_view = {"prediction_valid": True, "NC": legacy.get("NC"), "DAC": legacy.get("DAC"), "TTC_score": legacy.get("TTC_score"),
                   "comfort_score": legacy.get("comfort_score"), "EP": legacy.get("EP"), "PDMS_open_loop": legacy.get("PDMS_open_loop"), "map_query_status": "available"}
    lf, lr, lc = classify_frame(legacy_view)
    output.update({"legacy_first_zero_component": lf, "legacy_zero_reason": lr, "legacy_all_zero_components": "|".join(lc)})
    return output

from __future__ import annotations

import itertools
import math

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.collision_metrics import obb_overlap
from driveaudit_eval.trajectory.common import stats, wrap_angle
from driveaudit_eval.trajectory.map_rule_metrics import obb_polygon
from driveaudit_eval.trajectory.schema import share_fixed_metric_frame


def _velocity(track, idx):
    if idx <= 0 or not track.valid_mask[idx - 1]:
        return np.zeros(2)
    dt = track.timestamps[idx] - track.timestamps[idx - 1]
    return (track.positions[idx] - track.positions[idx - 1]) / dt if dt > 0 else np.zeros(2)


def _type(a, b, relative):
    if {a.agent_type, b.agent_type} & {"pedestrian"} and {a.agent_type, b.agent_type} & {"vehicle", "ego"}:
        return "pedestrian_crossing"
    angle = abs(float(wrap_angle(a.headings[relative[0]] - b.headings[relative[1]])))
    delta = b.positions[relative[1]] - a.positions[relative[0]]
    lateral = abs(-math.sin(a.headings[relative[0]]) * delta[0] + math.cos(a.headings[relative[0]]) * delta[1])
    if angle < math.radians(30) and lateral < 4.0:
        return "following"
    if angle > math.radians(60):
        return "crossing"
    return "other"


def _pet(a, b, config):
    ia = np.flatnonzero(a.valid_mask)
    ib = np.flatnonzero(b.valid_mask)
    if not len(ia) or not len(ib):
        return NOT_AVAILABLE
    # ponytail: O(n^2) is clear and sufficient for short challenge horizons; use a spatial index for very long tracks.
    distances = np.linalg.norm(a.positions[ia, None, :] - b.positions[None, ib, :], axis=2)
    candidates = np.where(distances <= float(config.get("conflict_distance_m", 2.0)))
    if not len(candidates[0]):
        return NOT_AVAILABLE
    time_gaps = np.abs(a.timestamps[ia[candidates[0]]] - b.timestamps[ib[candidates[1]]])
    positive = time_gaps[time_gaps > 1e-9]
    return float(np.min(positive)) if len(positive) else 0.0


def _interframe_collision(a, b, map_a, map_b, common):
    for left, right in zip(common[:-1], common[1:]):
        ia0, ia1, ib0, ib1 = map_a[left], map_a[right], map_b[left], map_b[right]
        for fraction in np.linspace(0.1, 0.9, 9):
            pa = a.positions[ia0] * (1 - fraction) + a.positions[ia1] * fraction
            pb = b.positions[ib0] * (1 - fraction) + b.positions[ib1] * fraction
            ha = a.headings[ia0] + float(wrap_angle(a.headings[ia1] - a.headings[ia0])) * fraction
            hb = b.headings[ib0] + float(wrap_angle(b.headings[ib1] - b.headings[ib0])) * fraction
            if obb_overlap(pa, ha, a.shape["length"], a.shape["width"], pb, hb, b.shape["length"], b.shape["width"]):
                return {"detected": True, "timestamp": float(left + (right - left) * fraction)}
    return {"detected": False, "timestamp": NOT_AVAILABLE}


def evaluate_interactions(tracks, collision_result, config):
    pair_results = []
    ttc_all, thw_all, drac_all, pet_all, min_distances = [], [], [], [], []
    total_exposure = 0.0
    for a, b in itertools.combinations(tracks, 2):
        if not share_fixed_metric_frame(a, b):
            continue
        start_a, start_b = int(a.metadata.get("history_frames") or 0), int(b.metadata.get("history_frames") or 0)
        map_a = {round(float(t), 9): i for i, t in enumerate(a.timestamps) if i >= start_a and a.valid_mask[i]}
        map_b = {round(float(t), 9): i for i, t in enumerate(b.timestamps) if i >= start_b and b.valid_mask[i]}
        common = sorted(set(map_a) & set(map_b))
        if not common:
            continue
        distances, ttcs, thws, dracs, exposure_times, risk_samples = [], [], [], [], [], []
        interaction_types = []
        for timestamp in common:
            ia, ib = map_a[timestamp], map_b[timestamp]
            box_a = obb_polygon(a.positions[ia], a.headings[ia], a.shape["length"], a.shape["width"])
            box_b = obb_polygon(b.positions[ib], b.headings[ib], b.shape["length"], b.shape["width"])
            distance = float(box_a.distance(box_b))
            distances.append(distance)
            va, vb = _velocity(a, ia), _velocity(b, ib)
            relative_position = b.positions[ib] - a.positions[ia]
            center_distance = float(np.linalg.norm(relative_position))
            relative_velocity = vb - va
            closing_speed = -float(np.dot(relative_position, relative_velocity)) / center_distance if center_distance > 1e-9 else float("inf")
            ttc = distance / closing_speed if closing_speed > 1e-9 else float("inf")
            if 0 <= ttc <= float(config.get("ttc_horizon_seconds", 10.0)):
                ttcs.append(ttc)
                exposure_times.append(timestamp)
                drac = closing_speed ** 2 / (2 * max(distance, 1e-6))
                dracs.append(drac)
                risk_samples.append({
                    "timestamp": timestamp, "ttc": float(ttc), "distance": distance,
                    "relative_speed": float(np.linalg.norm(relative_velocity)),
                    "closing_speed": closing_speed, "drac": float(drac),
                })
            forward_speed = float(np.dot(va, [math.cos(a.headings[ia]), math.sin(a.headings[ia])]))
            longitudinal_gap = float(np.dot(relative_position, [math.cos(a.headings[ia]), math.sin(a.headings[ia])]))
            interaction_type = _type(a, b, (ia, ib))
            interaction_types.append(interaction_type)
            if interaction_type == "following" and forward_speed > 0.1 and longitudinal_gap > 0:
                thws.append(max(0.0, distance) / forward_speed)
        dt = float(np.median(np.diff(common))) if len(common) > 1 else 0.0
        exposure = float(sum(value < 3.0 for value in ttcs) * dt)
        pet = _pet(a, b, config)
        if isinstance(pet, float):
            pet_all.append(pet)
        pair_type = max(set(interaction_types), key=interaction_types.count) if interaction_types else "other"
        minimum_event = min(risk_samples, key=lambda value: value["ttc"]) if risk_samples else NOT_AVAILABLE
        if isinstance(minimum_event, dict):
            minimum_event = dict(minimum_event, agent_ids=[a.agent_id, b.agent_id], interaction_type=pair_type,
                                 minimum_distance=float(min(distances)), pet=pet)
        pair_result = {
            "agent_ids": [a.agent_id, b.agent_id], "interaction_type": pair_type,
            "minimum_distance": float(min(distances)), "minimum_ttc": float(min(ttcs)) if ttcs else NOT_AVAILABLE,
            "ttc_below_1s_ratio": float(np.mean(np.asarray(ttcs) < 1)) if ttcs else 0.0,
            "ttc_below_2s_ratio": float(np.mean(np.asarray(ttcs) < 2)) if ttcs else 0.0,
            "ttc_below_3s_ratio": float(np.mean(np.asarray(ttcs) < 3)) if ttcs else 0.0,
            "thw": stats(thws), "pet": pet, "drac": stats(dracs),
            "conflict_exposure_duration": exposure,
            "minimum_ttc_event": minimum_event,
            "interframe_collision": _interframe_collision(a, b, map_a, map_b, common),
            "accepted_gap": NOT_AVAILABLE, "response_delay": NOT_AVAILABLE, "risk_recovery_time": NOT_AVAILABLE,
            "ttc_series": ttcs, "ttc_timestamps": exposure_times,
        }
        pair_results.append(pair_result)
        min_distances.append(min(distances))
        ttc_all.extend(ttcs)
        thw_all.extend(thws)
        drac_all.extend(dracs)
        total_exposure += exposure
    risks = [result["conflict_exposure_duration"] for result in pair_results]
    top_share = float(max(risks) / sum(risks)) if risks and sum(risks) > 0 else 0.0
    collision = collision_result.get("collision_any", False)
    minimum_ttc = min(ttc_all) if ttc_all else None
    classification = "invalid_inevitable_collision" if collision else ("high_risk_collision_free" if minimum_ttc is not None and minimum_ttc < 3 else "normal")
    grouped = {name: [] for name in ("following", "lane_change", "merging", "crossing", "yielding",
                                        "overtaking", "pedestrian_crossing", "other")}
    for result in pair_results:
        grouped.setdefault(result["interaction_type"], []).append(result)
    return {
        "minimum_distance": float(min(min_distances)) if min_distances else NOT_AVAILABLE,
        "minimum_ttc": float(minimum_ttc) if minimum_ttc is not None else NOT_AVAILABLE,
        "ttc_below_1s_ratio": float(np.mean(np.asarray(ttc_all) < 1)) if ttc_all else 0.0,
        "ttc_below_2s_ratio": float(np.mean(np.asarray(ttc_all) < 2)) if ttc_all else 0.0,
        "ttc_below_3s_ratio": float(np.mean(np.asarray(ttc_all) < 3)) if ttc_all else 0.0,
        "thw": stats(thw_all), "pet": stats(pet_all), "drac": stats(drac_all),
        "conflict_exposure_duration": total_exposure, "top_pair_risk_share": top_share,
        "accepted_gap": NOT_AVAILABLE, "response_delay": NOT_AVAILABLE, "risk_recovery_time": NOT_AVAILABLE,
        "risk_classification": classification, "pairs": pair_results,
        "by_interaction_type": grouped,
        "unavailable_reasons": {
            "accepted_gap": "merge/yield conflict-zone semantics are absent",
            "response_delay": "leader-event annotations are absent",
            "risk_recovery_time": "risk-onset and recovery event semantics are absent",
        },
    }


def compare_risk_baseline(generated, ground_truth):
    def pairs(result):
        return {tuple(sorted(item["agent_ids"])): item for item in result.get("pairs", [])}

    gen_pairs, gt_pairs = pairs(generated), pairs(ground_truth)
    gen_ttc, gt_ttc = generated.get("minimum_ttc"), ground_truth.get("minimum_ttc")
    gen_exposure = float(generated.get("conflict_exposure_duration", 0.0))
    gt_exposure = float(ground_truth.get("conflict_exposure_duration", 0.0))
    if gt_exposure > 0:
        amplification = gen_exposure / gt_exposure
    elif gen_exposure == 0:
        amplification = 1.0
    else:
        amplification = NOT_AVAILABLE
    pair_gaps = {}
    for pair in sorted(set(gen_pairs) & set(gt_pairs)):
        a, b = gen_pairs[pair], gt_pairs[pair]
        a_ttc, b_ttc = a.get("minimum_ttc"), b.get("minimum_ttc")
        pair_gaps["|".join(pair)] = {
            "generated_min_ttc": a_ttc,
            "gt_min_ttc": b_ttc,
            "ttc_gap": float(a_ttc - b_ttc) if isinstance(a_ttc, (int, float)) and isinstance(b_ttc, (int, float)) else NOT_AVAILABLE,
            "generated_ttc_exposure": a.get("conflict_exposure_duration", 0.0),
            "gt_ttc_exposure": b.get("conflict_exposure_duration", 0.0),
        }
    return {
        "generated_min_ttc": gen_ttc,
        "gt_min_ttc": gt_ttc,
        "ttc_gap": float(gen_ttc - gt_ttc) if isinstance(gen_ttc, (int, float)) and isinstance(gt_ttc, (int, float)) else NOT_AVAILABLE,
        "generated_ttc_exposure": gen_exposure,
        "gt_ttc_exposure": gt_exposure,
        "risk_amplification_ratio": float(amplification) if isinstance(amplification, (int, float)) else amplification,
        "pair_gaps": pair_gaps,
    }

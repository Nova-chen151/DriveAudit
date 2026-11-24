from __future__ import annotations

import itertools
import math

import numpy as np

from driveaudit_eval.trajectory.common import wrap_angle
from driveaudit_eval.trajectory.schema import share_fixed_metric_frame


DEFAULT_COLLISION_CONFIG = {
    "core_inset_m": 0.10,
    "scored_pair_types": {
        "vehicle-vehicle", "ego-vehicle", "ego-pedestrian",
        "ego-bicycle", "vehicle-pedestrian", "vehicle-bicycle",
    },
}


def obb_corners(position, heading, length, width):
    local = np.asarray([[length / 2, width / 2], [length / 2, -width / 2],
                        [-length / 2, -width / 2], [-length / 2, width / 2]])
    rotation = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    return local.dot(rotation.T) + np.asarray(position, dtype=float)


def obb_overlap(position_a, heading_a, length_a, width_a, position_b, heading_b, length_b, width_b):
    a = obb_corners(position_a, heading_a, length_a, width_a)
    b = obb_corners(position_b, heading_b, length_b, width_b)
    for polygon in (a, b):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.asarray([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm == 0:
                continue
            axis /= norm
            projection_a, projection_b = a.dot(axis), b.dot(axis)
            if projection_a.max() < projection_b.min() or projection_b.max() < projection_a.min():
                return False
    return True


def _velocity(track, idx):
    if idx <= 0 or not (track.valid_mask[idx] and track.valid_mask[idx - 1]):
        return np.zeros(2)
    dt = track.timestamps[idx] - track.timestamps[idx - 1]
    return (track.positions[idx] - track.positions[idx - 1]) / dt if dt > 0 else np.zeros(2)


def _collision_type(a, b):
    types = {a.agent_type, b.agent_type}
    if len(types) == 1:
        return f"{a.agent_type}-{a.agent_type}"
    for pair_type in (
        "ego-vehicle", "ego-pedestrian", "ego-bicycle",
        "vehicle-pedestrian", "vehicle-bicycle", "pedestrian-bicycle",
    ):
        if types == set(pair_type.split("-")):
            return pair_type
    return "-".join(sorted(types))


def _events(hits, common, pair, pair_type, dt_fallback):
    if not hits:
        return []
    dt = float(np.median(np.diff(common))) if len(common) > 1 else float(dt_fallback or 0.0)
    groups, current = [], []
    for hit in hits:
        if current and hit[0] - current[-1][0] > dt * 1.5:
            groups.append(current)
            current = []
        current.append(hit)
    if current:
        groups.append(current)
    return [{
        "collision_pair": list(pair), "collision_type": pair_type,
        "first_collision_time": group[0][0], "collision_duration": float(len(group) * dt),
        "collision_frame_count": len(group), "relative_speed": float(max(item[1] for item in group)),
        "collision_angle": float(group[0][2]), "agent_ids": list(pair),
    } for group in groups]


def evaluate_collisions(tracks, config=None):
    config = config or DEFAULT_COLLISION_CONFIG
    inset = max(0.0, float(config.get("core_inset_m", DEFAULT_COLLISION_CONFIG["core_inset_m"])))
    scored_types = set(config.get("scored_pair_types", DEFAULT_COLLISION_CONFIG["scored_pair_types"]))
    events = []
    raw_events = []
    total_timestamps = sorted({
        round(float(track.timestamps[i]), 9)
        for track in tracks
        for i in range(int(track.metadata.get("history_frames") or 0), len(track.timestamps))
        if track.valid_mask[i]
    })
    colliding_frames = set()
    raw_frames = set()
    possible_pairs = set()
    colliding_pairs = set()
    raw_pairs = set()
    for a, b in itertools.combinations(tracks, 2):
        if not share_fixed_metric_frame(a, b):
            continue
        start_a, start_b = int(a.metadata.get("history_frames") or 0), int(b.metadata.get("history_frames") or 0)
        map_a = {round(float(t), 9): i for i, t in enumerate(a.timestamps) if i >= start_a and a.valid_mask[i]}
        map_b = {round(float(t), 9): i for i, t in enumerate(b.timestamps) if i >= start_b and b.valid_mask[i]}
        common = sorted(set(map_a) & set(map_b))
        if not common:
            continue
        pair = tuple(sorted((a.agent_id, b.agent_id)))
        pair_type = _collision_type(a, b)
        possible_pairs.add(pair)
        hits = []
        raw_hits = []
        for timestamp in common:
            ia, ib = map_a[timestamp], map_b[timestamp]
            if obb_overlap(a.positions[ia], a.headings[ia], a.shape["length"], a.shape["width"],
                           b.positions[ib], b.headings[ib], b.shape["length"], b.shape["width"]):
                relative = _velocity(a, ia) - _velocity(b, ib)
                hit = (timestamp, float(np.linalg.norm(relative)),
                       float(abs(wrap_angle(a.headings[ia] - b.headings[ib]))))
                raw_hits.append(hit)
                raw_frames.add(timestamp)
                if pair_type in scored_types and obb_overlap(
                    a.positions[ia], a.headings[ia], max(np.finfo(float).eps, a.shape["length"] - 2 * inset),
                    max(np.finfo(float).eps, a.shape["width"] - 2 * inset), b.positions[ib], b.headings[ib],
                    max(np.finfo(float).eps, b.shape["length"] - 2 * inset),
                    max(np.finfo(float).eps, b.shape["width"] - 2 * inset),
                ):
                    hits.append(hit)
                    colliding_frames.add(timestamp)
        fallback = a.metadata.get("dt") or b.metadata.get("dt") or 0.0
        if raw_hits:
            raw_pairs.add(pair)
            raw_events.extend(_events(raw_hits, common, pair, pair_type, fallback))
        if hits:
            colliding_pairs.add(pair)
            events.extend(_events(hits, common, pair, pair_type, fallback))
    type_counts = {}
    for event in events:
        type_counts[event["collision_type"]] = type_counts.get(event["collision_type"], 0) + 1
    raw_type_counts = {}
    for event in raw_events:
        raw_type_counts[event["collision_type"]] = raw_type_counts.get(event["collision_type"], 0) + 1
    return {
        "collision_any": bool(events),
        "collision_count": len(events),
        "collision_pairs": [list(pair) for pair in sorted(colliding_pairs)],
        "events": events,
        "collision_frame_ratio": float(len(colliding_frames) / len(total_timestamps)) if total_timestamps else 0.0,
        "collision_agent_pair_ratio": float(len(colliding_pairs) / len(possible_pairs)) if possible_pairs else 0.0,
        "collision_type_counts": type_counts,
        "raw_overlap_any": bool(raw_events), "raw_overlap_count": len(raw_events),
        "raw_overlap_pairs": [list(pair) for pair in sorted(raw_pairs)], "raw_overlap_events": raw_events,
        "raw_overlap_frame_ratio": float(len(raw_frames) / len(total_timestamps)) if total_timestamps else 0.0,
        "raw_overlap_pair_ratio": float(len(raw_pairs) / len(possible_pairs)) if possible_pairs else 0.0,
        "raw_overlap_pair_type_counts": raw_type_counts,
        "collision_core_inset_m": inset, "scored_pair_types": sorted(scored_types),
        "coordinate_requirement": "same timestamp and fixed metric frame; raw OBB overlap is diagnostic; scored collision uses inset OBB cores",
    }

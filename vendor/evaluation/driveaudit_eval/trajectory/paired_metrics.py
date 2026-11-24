from __future__ import annotations

import math

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.common import wrap_angle
from driveaudit_eval.trajectory.schema import share_fixed_metric_frame


def _interpolate_valid(track, query):
    valid = np.flatnonzero(track.valid_mask)
    if len(valid) < 2:
        return None
    timestamps = track.timestamps[valid]
    positions = track.positions[valid]
    headings = np.unwrap(track.headings[valid])
    base_dt = track.metadata.get("dt") or float(np.median(np.diff(timestamps)))
    output_pos, output_heading, keep = [], [], []
    for value in query:
        right = int(np.searchsorted(timestamps, value, side="left"))
        if right == 0 or right == len(timestamps):
            if right < len(timestamps) and abs(timestamps[right] - value) < 1e-9:
                left = right
            else:
                continue
        else:
            left = right - 1
        if right != left and timestamps[right] - timestamps[left] > base_dt * 1.5:
            continue
        output_pos.append([np.interp(value, timestamps, positions[:, axis]) for axis in (0, 1)])
        output_heading.append(np.interp(value, timestamps, headings))
        keep.append(value)
    return np.asarray(keep), np.asarray(output_pos), wrap_angle(np.asarray(output_heading))


def _motion(positions, timestamps):
    if len(timestamps) < 2:
        return np.asarray([]), np.asarray([])
    velocity_vectors = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]
    speed = np.linalg.norm(velocity_vectors, axis=1)
    acceleration = np.diff(speed) / np.diff(timestamps)[1:] if len(speed) > 1 else np.asarray([])
    return speed, acceleration


def dtw_distance(a, b):
    table = np.full((len(a) + 1, len(b) + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[-1, -1] / max(1, len(a) + len(b)))


def frechet_distance(a, b):
    cache = np.full((len(a), len(b)), -1.0)
    def visit(i, j):
        if cache[i, j] >= 0:
            return cache[i, j]
        distance = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            value = distance
        elif i > 0 and j == 0:
            value = max(visit(i - 1, 0), distance)
        elif i == 0 and j > 0:
            value = max(visit(0, j - 1), distance)
        else:
            value = max(min(visit(i - 1, j), visit(i - 1, j - 1), visit(i, j - 1)), distance)
        cache[i, j] = value
        return value
    return float(visit(len(a) - 1, len(b) - 1))


def evaluate_pair(generated, ground_truth):
    if generated.agent_id != ground_truth.agent_id:
        raise ValueError("agent_id mismatch: ADE/FDE identity matching is strict")
    if not share_fixed_metric_frame(generated, ground_truth):
        raise ValueError("paired metrics require the same fixed metric coordinate frame")
    start = int(generated.metadata.get("history_frames") or 0)
    query = generated.timestamps[start:][generated.valid_mask[start:]]
    interpolated = _interpolate_valid(ground_truth, query)
    if interpolated is None:
        return NOT_AVAILABLE
    common, gt_position, gt_heading = interpolated
    if len(common) < 2:
        return NOT_AVAILABLE
    gen_indices = np.asarray([int(np.argmin(abs(generated.timestamps - value))) for value in common])
    gen_position = generated.positions[gen_indices]
    gen_heading = generated.headings[gen_indices]
    displacement = np.linalg.norm(gen_position - gt_position, axis=1)
    heading_error = np.abs(wrap_angle(gen_heading - gt_heading))
    gen_speed, gen_acc = _motion(gen_position, common)
    gt_speed, gt_acc = _motion(gt_position, common)
    endpoint = gen_position[-1] - gt_position[-1]
    forward = np.asarray([math.cos(gt_heading[-1]), math.sin(gt_heading[-1])])
    lateral = np.asarray([-forward[1], forward[0]])
    return {
        "comparison_points": len(common), "common_timestamps": common.tolist(),
        "ADE": float(np.mean(displacement)), "FDE": float(displacement[-1]),
        "heading_ADE": float(np.mean(heading_error)), "heading_FDE": float(heading_error[-1]),
        "velocity_MAE": float(np.mean(np.abs(gen_speed - gt_speed))) if len(gen_speed) else NOT_AVAILABLE,
        "acceleration_MAE": float(np.mean(np.abs(gen_acc - gt_acc))) if len(gen_acc) else NOT_AVAILABLE,
        "DTW_distance": dtw_distance(gen_position, gt_position),
        "Frechet_curve_distance": frechet_distance(gen_position, gt_position),
        "endpoint_longitudinal_error": float(np.dot(endpoint, forward)),
        "endpoint_lateral_error": float(np.dot(endpoint, lateral)),
        "caveat": "ADE/FDE only measure consistency with one recorded future; they do not make another plausible future wrong.",
    }


def evaluate_paired(generated, ground_truth):
    gen = {track.agent_id: track for track in generated if track.metadata.get("predict", True)}
    gt = {track.agent_id: track for track in ground_truth if track.metadata.get("predict", True)}
    if set(gen) != set(gt):
        raise ValueError("agent_id sets differ; nearest-position identity matching is forbidden")
    return {agent_id: evaluate_pair(gen[agent_id], gt[agent_id]) for agent_id in sorted(gen)}


def audit_gt_copy(generated, ground_truth):
    gen = {track.agent_id: track for track in generated}
    ref = {track.agent_id: track for track in ground_truth}
    if set(gen) != set(ref):
        return {
            "is_exact_gt_copy": False,
            "agent_id_sets_equal": False,
            "generated_only_agent_ids": sorted(set(gen) - set(ref)),
            "reference_only_agent_ids": sorted(set(ref) - set(gen)),
            "per_agent": {},
        }
    per_agent = {}
    for agent_id in sorted(gen):
        a, b = gen[agent_id], ref[agent_id]
        same_length = len(a.timestamps) == len(b.timestamps)
        joint = a.valid_mask & b.valid_mask if same_length else np.asarray([], dtype=bool)
        position_error = np.linalg.norm(a.positions - b.positions, axis=1) if a.positions.shape == b.positions.shape else np.asarray([np.inf])
        heading_error = np.abs(wrap_angle(a.headings - b.headings)) if a.headings.shape == b.headings.shape else np.asarray([np.inf])
        mask_equal = bool(np.array_equal(a.valid_mask, b.valid_mask))
        timestamp_equal = bool(np.array_equal(a.timestamps, b.timestamps))
        shape_equal = bool(a.shape == b.shape)
        item = {
            "agent_id_equal": a.agent_id == b.agent_id,
            "position_max_error": float(np.max(position_error)),
            "heading_max_error": float(np.max(heading_error)),
            "valid_mask_equal": mask_equal,
            "timestamp_equal": timestamp_equal,
            "shape_equal": shape_equal,
            "valid_intersection_position_max_error": float(np.max(position_error[joint])) if len(joint) and np.any(joint) else NOT_AVAILABLE,
            "valid_intersection_heading_max_error": float(np.max(heading_error[joint])) if len(joint) and np.any(joint) else NOT_AVAILABLE,
        }
        item["equal_only_on_valid_intersection"] = bool(
            not mask_equal and item["valid_intersection_position_max_error"] == 0.0
            and item["valid_intersection_heading_max_error"] == 0.0
        )
        item["is_exact_gt_copy"] = bool(
            item["agent_id_equal"] and item["position_max_error"] == 0.0 and item["heading_max_error"] == 0.0
            and mask_equal and timestamp_equal and shape_equal
        )
        per_agent[agent_id] = item
    return {
        "is_exact_gt_copy": bool(all(item["is_exact_gt_copy"] for item in per_agent.values())),
        "agent_id_sets_equal": True,
        "per_agent": per_agent,
    }

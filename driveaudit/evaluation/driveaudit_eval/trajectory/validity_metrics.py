from __future__ import annotations

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.common import longest_true_duration, wrap_angle


def _future_start(track):
    value = track.metadata.get("history_frames")
    return int(value) if value is not None else 0


def _run_mask(mask, minimum_points):
    output = np.zeros(len(mask), dtype=bool)
    start = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= minimum_points:
                output[start:i] = True
            start = None
    return output


def history_future_boundary(track):
    start = _future_start(track)
    if start <= 1 or start >= len(track.timestamps):
        return NOT_AVAILABLE
    history = np.flatnonzero(track.valid_mask[:start])
    future = np.flatnonzero(track.valid_mask[start:]) + start
    if len(history) < 2 or not len(future):
        return NOT_AVAILABLE
    last = int(history[-1])
    previous = int(history[-2])
    first = int(future[0])
    dt_history = track.timestamps[last] - track.timestamps[previous]
    dt_boundary = track.timestamps[first] - track.timestamps[last]
    if dt_history <= 0 or dt_boundary <= 0:
        return NOT_AVAILABLE
    velocity_history = (track.positions[last] - track.positions[previous]) / dt_history
    expected = track.positions[last] + velocity_history * dt_boundary
    result = {
        "history_last_index": last,
        "future_first_index": first,
        "boundary_position_error": float(np.linalg.norm(track.positions[first] - expected)),
        "boundary_heading_error": float(abs(wrap_angle(track.headings[first] - track.headings[last]))),
    }
    next_future = future[future > first]
    if len(next_future) and next_future[0] == first + 1:
        second = int(next_future[0])
        velocity_future = (track.positions[second] - track.positions[first]) / (track.timestamps[second] - track.timestamps[first])
        result["boundary_speed_error"] = float(abs(np.linalg.norm(velocity_future) - np.linalg.norm(velocity_history)))
        result["boundary_acceleration_jump"] = float(np.linalg.norm(velocity_future - velocity_history) / dt_boundary)
    else:
        result["boundary_speed_error"] = NOT_AVAILABLE
        result["boundary_acceleration_jump"] = NOT_AVAILABLE
    return result


def evaluate_track_validity(track, config):
    start = _future_start(track)
    timestamps = track.timestamps[start:]
    positions = track.positions[start:]
    valid = track.valid_mask[start:]
    expected = config.get("expected_future_frames") or len(valid)
    valid_count = int(np.sum(valid))
    valid_indices = np.flatnonzero(valid)
    gaps = 0
    if len(valid_indices):
        gaps = int(np.sum(~valid[valid_indices[0]:valid_indices[-1] + 1]))
        span = int(valid_indices[-1] - valid_indices[0] + 1)
    else:
        span = 0
    timestamp_diffs = np.diff(timestamps)
    expected_dt = track.metadata.get("dt")
    tolerance = float(config.get("dt_relative_tolerance", 0.05))
    timestamp_error = {
        "non_increasing_count": int(np.sum(timestamp_diffs < 0)),
        "duplicate_count": int(np.sum(timestamp_diffs == 0)),
        "dt_anomaly_count": int(np.sum(np.abs(timestamp_diffs - expected_dt) > expected_dt * tolerance)) if expected_dt else NOT_AVAILABLE,
        "duration_seconds": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "trajectory_horizon_shortfall_seconds": max(0.0, float((expected - 1) * expected_dt - (timestamps[-1] - timestamps[0]))) if expected_dt and len(timestamps) else NOT_AVAILABLE,
    }
    adjacent = valid[1:] & valid[:-1]
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = timestamp_diffs
    speed = np.divide(displacement, dt, out=np.full_like(displacement, np.nan), where=adjacent & (dt > 0))
    limit = float(config.get("teleport_speed_max", {}).get(track.agent_type, 70.0))
    teleport = adjacent & (speed > limit)
    duplicate = adjacent & (displacement <= float(config.get("duplicate_distance_epsilon", 0.001)))
    heading = track.headings[start + 1:]
    delta = np.diff(positions, axis=0)
    forward = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
    reverse = adjacent & (speed > float(config.get("reverse_speed_min", 0.5))) & (forward < 0)
    base_dt = expected_dt or (float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0)
    reverse_min = max(1, int(np.ceil(float(config.get("reverse_min_duration_seconds", 1.0)) / base_dt)))
    reverse_sustained = _run_mask(reverse, reverse_min)
    stationary = adjacent & (speed <= float(config.get("stationary_speed_epsilon", 0.1)))
    stationary_min = max(1, int(np.ceil(float(config.get("stationary_duration_seconds", 3.0)) / base_dt)))
    stationary_sustained = _run_mask(stationary, stationary_min)
    denominator = max(1, int(np.sum(adjacent)))
    termination_reason = track.metadata.get("termination_reason")
    return {
        "valid_point_ratio": float(valid_count / expected) if expected else NOT_AVAILABLE,
        "continuity_gap_ratio": float(gaps / span) if span else 1.0,
        "timestamp_error": timestamp_error,
        "teleportation_rate": float(np.sum(teleport) / denominator),
        "teleportation_count": int(np.sum(teleport)),
        "duplicate_point_ratio": float(np.sum(duplicate) / denominator),
        "stationary_collapse_ratio": float(np.sum(stationary_sustained) / denominator),
        "reverse_motion_ratio": float(np.sum(reverse_sustained) / denominator),
        "max_reverse_duration_seconds": longest_true_duration(reverse_sustained, timestamps[1:]),
        "history_future_boundary": history_future_boundary(track),
        "expected_future_points": int(expected),
        "valid_future_points": valid_count,
        "exited_map": track.metadata.get("exited_map", NOT_AVAILABLE),
        "exited_roi": track.metadata.get("exited_roi", NOT_AVAILABLE),
        "explicit_termination_reason": termination_reason or NOT_AVAILABLE,
        "unexplained_disappearance": bool((valid_count == 0 or (valid_count and not valid[-1])) and not termination_reason),
    }


def evaluate_scene_validity(tracks, config):
    per_agent = {track.agent_id: evaluate_track_validity(track, config) for track in tracks}
    expected = [track for track in tracks if track.metadata.get("predict", True)]
    missing = []
    for track in expected:
        start = _future_start(track)
        if np.any(track.valid_mask[:start]) and not np.any(track.valid_mask[start:]):
            missing.append(track.agent_id)
    identities = [track.agent_id for track in tracks]
    duplicate_ids = sorted({value for value in identities if identities.count(value) > 1})
    return {
        "per_agent": per_agent,
        "missing_agent_ratio": float(len(missing) / len(expected)) if expected else NOT_AVAILABLE,
        "missing_agent_ids": missing,
        "identity_persistence": float((len(expected) - len(missing)) / len(expected)) if expected else NOT_AVAILABLE,
        "duplicate_agent_ids": duplicate_ids,
    }


def compare_valid_masks(generated, ground_truth):
    """Compare future availability on declared timestamps without matching identities by position."""
    if generated.agent_id != ground_truth.agent_id:
        raise ValueError("agent_id mismatch: availability matching is strict")
    start_g = _future_start(generated)
    start_r = _future_start(ground_truth)
    gen = {round(float(t), 9): bool(generated.valid_mask[i]) for i, t in enumerate(generated.timestamps) if i >= start_g}
    ref = {round(float(t), 9): bool(ground_truth.valid_mask[i]) for i, t in enumerate(ground_truth.timestamps) if i >= start_r}
    timeline = sorted(set(gen) | set(ref))
    gen_mask = np.asarray([gen.get(t, False) for t in timeline], dtype=bool)
    ref_mask = np.asarray([ref.get(t, False) for t in timeline], dtype=bool)
    tp = int(np.sum(gen_mask & ref_mask))
    fp = int(np.sum(gen_mask & ~ref_mask))
    fn = int(np.sum(~gen_mask & ref_mask))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 1.0

    def boundary(mask, first):
        indices = np.flatnonzero(mask)
        return float(timeline[indices[0 if first else -1]]) if len(indices) else None

    gen_entry, ref_entry = boundary(gen_mask, True), boundary(ref_mask, True)
    gen_exit, ref_exit = boundary(gen_mask, False), boundary(ref_mask, False)
    return {
        "comparison_frames": len(timeline),
        "valid_mask_precision": ratio(tp, tp + fp),
        "valid_mask_recall": ratio(tp, tp + fn),
        "valid_mask_iou": ratio(tp, tp + fp + fn),
        "premature_disappearance_frames": fn,
        "over_persistence_frames": fp,
        "disappearance_time_error": float(gen_exit - ref_exit) if gen_exit is not None and ref_exit is not None else NOT_AVAILABLE,
        "entry_time_error": float(gen_entry - ref_entry) if gen_entry is not None and ref_entry is not None else NOT_AVAILABLE,
        # No valid reference future is unscorable, not 100% unexplained missing.
        "unexplained_missing_ratio": float(fn / np.sum(ref_mask)) if np.any(ref_mask) else 0.0,
        "generated_first_valid_time": gen_entry if gen_entry is not None else NOT_AVAILABLE,
        "reference_first_valid_time": ref_entry if ref_entry is not None else NOT_AVAILABLE,
        "generated_last_valid_time": gen_exit if gen_exit is not None else NOT_AVAILABLE,
        "reference_last_valid_time": ref_exit if ref_exit is not None else NOT_AVAILABLE,
        "valid_mask_equal": bool(np.array_equal(gen_mask, ref_mask)),
    }


def compare_scene_availability(generated, ground_truth):
    gen = {track.agent_id: track for track in generated}
    ref = {track.agent_id: track for track in ground_truth}
    if set(gen) != set(ref):
        raise ValueError("agent_id sets differ; availability matching is strict")
    per_agent = {agent_id: compare_valid_masks(gen[agent_id], ref[agent_id]) for agent_id in sorted(gen)}
    return {
        "per_agent": per_agent,
        "mean_valid_mask_precision": float(np.mean([value["valid_mask_precision"] for value in per_agent.values()])),
        "mean_valid_mask_recall": float(np.mean([value["valid_mask_recall"] for value in per_agent.values()])),
        "mean_valid_mask_iou": float(np.mean([value["valid_mask_iou"] for value in per_agent.values()])),
        "total_premature_disappearance_frames": int(sum(value["premature_disappearance_frames"] for value in per_agent.values())),
        "total_over_persistence_frames": int(sum(value["over_persistence_frames"] for value in per_agent.values())),
        "mean_unexplained_missing_ratio": float(np.mean([value["unexplained_missing_ratio"] for value in per_agent.values()])),
    }

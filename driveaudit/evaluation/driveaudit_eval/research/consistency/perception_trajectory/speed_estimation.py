from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


SPEED_SOURCES = ("bevformer_native", "kalman_state", "position_difference", "smoothed_position_difference")


def contiguous_segments(pairs: list[tuple[dict[str, Any], dict[str, Any]]],
                        maximum_gap_seconds: float) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    segments, current = [], []
    for pair in sorted(pairs, key=lambda value: value[0]["timestamp"]):
        if current:
            previous = current[-1][0]
            gap = float(pair[0]["timestamp"]) - float(previous["timestamp"])
            if int(pair[0]["frame_index"]) != int(previous["frame_index"]) + 1 or gap > maximum_gap_seconds:
                segments.append(current); current = []
        current.append(pair)
    if current:
        segments.append(current)
    return segments


def _local_linear_velocity(rows: list[dict[str, Any]], window: int) -> np.ndarray:
    times = np.asarray([row["timestamp"] for row in rows], dtype=float)
    xy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in rows], dtype=float)
    result = np.empty_like(xy); half = window // 2
    for index in range(len(rows)):
        left, right = max(0, index - half), min(len(rows), index + half + 1)
        local_t = times[left:right] - times[index]
        if len(local_t) < 2 or np.ptp(local_t) <= 0:
            result[index] = np.nan
        else:
            result[index] = [np.polyfit(local_t, xy[left:right, axis], 1)[0] for axis in range(2)]
    return result


def _bucket(length: int) -> str:
    if length <= 3: return "2-3"
    if length <= 9: return "4-9"
    if length <= 19: return "10-19"
    return ">=20"


def compare_speed_protocols(reference_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                            mapping: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    reference = {track["agent_id"]: track for track in reference_tracks}
    images = {track["image_track_id"]: track for track in image_tracks}
    minimum_length = int(config.get("minimum_track_length", 4)); maximum_gap = float(config.get("maximum_gap_seconds", 0.13))
    window = int(config.get("smoothing_window", 5)); errors: dict[str, list[float]] = defaultdict(list)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for match in mapping:
        sid, iid = match.get("submission_agent_id"), match.get("image_track_id")
        if not sid or not iid or sid not in reference or iid not in images:
            continue
        s = {int(row["frame_index"]): row for row in reference[sid]["observations"] if row.get("expected_visible", True)}
        i = {int(row["frame_index"]): row for row in images[iid]["observations"] if row.get("source") == "image_detection"}
        pairs = [(s[frame], i[frame]) for frame in sorted(set(s) & set(i))]
        for segment in contiguous_segments(pairs, maximum_gap):
            if len(segment) < minimum_length:
                continue
            bucket = _bucket(len(segment)); sr, ir = zip(*segment)
            for source, field in (("bevformer_native", "native_velocity"), ("kalman_state", "kalman_velocity")):
                if all(field in row or (source == "bevformer_native" and "velocity" in row) for row in ir):
                    sv = np.asarray([row.get("velocity", [np.nan, np.nan]) for row in sr], dtype=float)
                    iv = np.asarray([row.get(field, row.get("velocity", [np.nan, np.nan])) for row in ir], dtype=float)
                    values = np.abs(np.linalg.norm(iv, axis=1) - np.linalg.norm(sv, axis=1))
                    values = values[np.isfinite(values)]; errors[source].extend(values); buckets[source][bucket].extend(values)
            st = np.asarray([row["timestamp"] for row in sr], dtype=float)
            sxy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in sr], dtype=float)
            ixy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in ir], dtype=float)
            dt = np.diff(st); valid = dt > 0
            sv = np.diff(sxy, axis=0)[valid] / dt[valid, None]; iv = np.diff(ixy, axis=0)[valid] / dt[valid, None]
            values = np.abs(np.linalg.norm(iv, axis=1) - np.linalg.norm(sv, axis=1))
            errors["position_difference"].extend(values); buckets["position_difference"][bucket].extend(values)
            sv, iv = _local_linear_velocity(list(sr), window), _local_linear_velocity(list(ir), window)
            values = np.abs(np.linalg.norm(iv, axis=1) - np.linalg.norm(sv, axis=1)); values = values[np.isfinite(values)]
            errors["smoothed_position_difference"].extend(values); buckets["smoothed_position_difference"][bucket].extend(values)
    result = {}
    for source in SPEED_SOURCES:
        values = np.asarray(errors[source], dtype=float)
        result[source] = {"speed_mae": float(values.mean()) if len(values) else NOT_AVAILABLE,
                          "speed_rmse": float(np.sqrt(np.mean(values ** 2))) if len(values) else NOT_AVAILABLE,
                          "speed_p90": float(np.percentile(values, 90)) if len(values) else NOT_AVAILABLE,
                          "speed_p99": float(np.percentile(values, 99)) if len(values) else NOT_AVAILABLE,
                          "valid_speed_samples": int(len(values)),
                          "track_length_bucket": {bucket: {"speed_mae": float(np.mean(bucket_values)), "samples": len(bucket_values)}
                                                  for bucket, bucket_values in buckets[source].items()}}
    result["protocol"] = {"minimum_track_length": minimum_length, "maximum_gap_seconds": maximum_gap,
                          "smoothing": "local_linear_regression", "smoothing_window": window,
                          "crosses_track_gaps": False}
    return result

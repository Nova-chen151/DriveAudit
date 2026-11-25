from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy, ks_2samp, wasserstein_distance

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def _features(tracks: list[dict[str, Any]]) -> tuple[dict[str, list[float]], Counter, Counter]:
    from driveaudit_eval.research.consistency.perception_trajectory.metrics.actbench_metrics import _motion_class
    values: dict[str, list[float]] = defaultdict(list); classes, modes = Counter(), Counter()
    origin = min((row["timestamp"] for track in tracks for row in track["observations"]), default=0.0)
    by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        rows = sorted(track["observations"], key=lambda row: row["timestamp"])
        if not rows: continue
        classes[track.get("agent_type", track.get("category", "unknown"))] += 1
        modes[_motion_class(rows)] += 1
        for row in rows: by_time[round(float(row["timestamp"]), 3)].append(row)
        boxes = [row["box_bev"] for row in rows]
        t = np.asarray([row["timestamp"] for row in rows], dtype=float)
        xy = np.asarray([[box["x"], box["y"]] for box in boxes], dtype=float)
        yaw = np.unwrap(np.asarray([box["yaw"] for box in boxes], dtype=float))
        values["box_length"].extend(box["length"] for box in boxes); values["box_width"].extend(box["width"] for box in boxes)
        values["track_length"].append(len(rows)); values["visible_duration"].append(float(t[-1] - t[0]) if len(t) > 1 else 0.0)
        values["entry_time"].append(float(t[0] - origin)); values["exit_time"].append(float(t[-1] - origin))
        values["trajectory_length"].append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0)
        if len(rows) < 2: continue
        dt = np.diff(t); valid = dt > 0
        velocity = np.diff(xy, axis=0)[valid] / dt[valid, None]
        speed = np.linalg.norm(velocity, axis=1); values["speed"].extend(speed)
        values["stop_duration"].append(float(dt[valid][speed < .5].sum()))
        lateral = float(np.dot(xy[-1] - xy[0], [-np.sin(yaw[0]), np.cos(yaw[0])]))
        values["lane_change_duration"].append(float(t[-1] - t[0])) if abs(lateral) > 1.5 and abs(yaw[-1] - yaw[0]) < .35 else None
        values["longitudinal_velocity"].extend(np.sum(velocity * np.column_stack([np.cos(yaw[:-1][valid]), np.sin(yaw[:-1][valid])]), axis=1))
        values["lateral_velocity"].extend(np.sum(velocity * np.column_stack([-np.sin(yaw[:-1][valid]), np.cos(yaw[:-1][valid])]), axis=1))
        yaw_rate = np.diff(yaw)[valid] / dt[valid]; values["yaw_rate"].extend(yaw_rate)
        values["curvature"].extend(yaw_rate / np.maximum(speed, 1e-3))
        if len(speed) > 1:
            acceleration = np.diff(speed) / dt[valid][1:]; values["acceleration"].extend(acceleration)
            if len(acceleration) > 1: values["jerk"].extend(np.diff(acceleration) / dt[valid][2:])
    values["PET"] = values.get("PET", [])  # not derivable robustly without conflict-zone geometry.
    for rows in by_time.values():
        for index, a in enumerate(rows):
            for b in rows[index + 1:]:
                pa = np.asarray([a["box_bev"]["x"], a["box_bev"]["y"]]); pb = np.asarray([b["box_bev"]["x"], b["box_bev"]["y"]])
                va, vb = np.asarray(a.get("velocity", [0, 0]), dtype=float), np.asarray(b.get("velocity", [0, 0]), dtype=float)
                delta, relative = pb - pa, vb - va; distance = float(np.linalg.norm(delta))
                values["minimum_distance"].append(distance)
                closing = -float(np.dot(delta, relative)) / max(distance, 1e-6)
                if closing > 0:
                    values["TTC"].append(distance / closing); values["DRAC"].append(closing * closing / max(2 * distance, 1e-6))
    return values, classes, modes


def _mmd(a: np.ndarray, b: np.ndarray) -> float:
    # ponytail: quadratic MMD is capped deterministically; use linear-time MMD when multi-scene scale warrants it.
    if len(a) > 2000: a = a[np.linspace(0, len(a) - 1, 2000, dtype=int)]
    if len(b) > 2000: b = b[np.linspace(0, len(b) - 1, 2000, dtype=int)]
    pooled = np.concatenate([a, b])[:, None]
    distances = cdist(pooled, pooled, "sqeuclidean")
    positive = distances[distances > 0]; bandwidth = float(np.median(positive)) if len(positive) else 1.0
    k = np.exp(-distances / max(2.0 * bandwidth, 1e-12)); n = len(a)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


def _compare(a: list[float], b: list[float]) -> dict[str, Any]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b): return {"status": NOT_AVAILABLE, "unavailable_reason": "empty_distribution"}
    low, high = min(a.min(), b.min()), max(a.max(), b.max())
    if math.isclose(low, high): high = low + 1e-9
    edges = np.linspace(low, high, 21); ah, _ = np.histogram(a, edges); bh, _ = np.histogram(b, edges)
    ap, bp = (ah + 1e-12) / (ah.sum() + 20e-12), (bh + 1e-12) / (bh.sum() + 20e-12); middle = (ap + bp) / 2
    result = {"js_divergence": float(0.5 * (entropy(ap, middle) + entropy(bp, middle))),
              "wasserstein_distance": float(wasserstein_distance(a, b)), "ks_statistic": float(ks_2samp(a, b).statistic),
              "mmd": _mmd(a, b), "mean_difference": float(a.mean() - b.mean()),
              "std_difference": float(a.std() - b.std()), "common_bin_edges": edges.tolist(),
              "sample_count_a": len(a), "sample_count_b": len(b)}
    for percentile in (10, 50, 90, 99): result[f"p{percentile}_difference"] = float(np.percentile(a, percentile) - np.percentile(b, percentile))
    return result


def evaluate_distribution_reference(submitted: list[dict[str, Any]], image: list[dict[str, Any]],
                                    real: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {"submitted": _features(submitted), "image_extracted": _features(image), "real": _features(real)}
    pairs = (("submitted_vs_real", "submitted", "real"), ("image_extracted_vs_real", "image_extracted", "real"),
             ("image_extracted_vs_submitted", "image_extracted", "submitted"))
    result = {}
    for output, left, right in pairs:
        a, b = sources[left][0], sources[right][0]
        feature_names = sorted(set(a) | set(b))
        class_names = sorted(set(sources[left][1]) | set(sources[right][1]))
        mode_names = sorted(set(sources[left][2]) | set(sources[right][2]))
        at, bt = sum(sources[left][1].values()), sum(sources[right][1].values())
        real_modes = {key for key, value in sources[right][2].items() if value > 0}
        rare = {key for key, value in sources[right][2].items() if value <= max(1, .1 * sum(sources[right][2].values()))}
        result[output] = {"features": {key: _compare(a.get(key, []), b.get(key, [])) for key in feature_names},
                          "class_distribution_gap": {key: (sources[left][1][key] / at if at else 0) - (sources[right][1][key] / bt if bt else 0)
                                                     for key in class_names},
                          "motion_class_distribution_gap": {key: sources[left][2][key] / max(sum(sources[left][2].values()), 1) -
                                                            sources[right][2][key] / max(sum(sources[right][2].values()), 1) for key in mode_names},
                          "mode_coverage": len({key for key in real_modes if sources[left][2][key]}) / len(real_modes) if real_modes else NOT_AVAILABLE,
                          "rare_mode_coverage": len({key for key in rare if sources[left][2][key]}) / len(rare) if rare else NOT_AVAILABLE}
    return {"protocol": "distribution_reference", "status": "valid_low_sample_distribution",
            "paired_identity_used": False, "official_ftd": NOT_AVAILABLE,
            "official_ftd_unavailable_reason": "official_MTR_encoder_checkpoint_or_sample_size_unavailable",
            **result}


def trajectory_schema_to_tracks(trajectories: list[Any]) -> list[dict[str, Any]]:
    result = []
    for track in trajectories:
        rows = []
        for index in np.flatnonzero(track.valid_mask):
            rows.append({"frame_index": int(index), "timestamp": float(track.timestamps[index]),
                         "box_bev": {"x": float(track.positions[index, 0]), "y": float(track.positions[index, 1]),
                                     "z": 0.0, "yaw": float(track.headings[index]), "length": track.shape["length"],
                                     "width": track.shape["width"], "height": 1.5},
                         "velocity": [float(track.velocities[index]), 0.0], "source": "distribution_reference"})
        if rows: result.append({"agent_id": track.agent_id, "agent_type": track.agent_type, "category": track.agent_type,
                                "observations": rows})
    return result

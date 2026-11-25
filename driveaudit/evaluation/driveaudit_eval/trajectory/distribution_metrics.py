from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from driveaudit_eval.trajectory import NOT_AVAILABLE


def common_histograms(generated, ground_truth, bins=20):
    generated = np.asarray(generated, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    generated = generated[np.isfinite(generated)]
    ground_truth = ground_truth[np.isfinite(ground_truth)]
    if not len(generated) or not len(ground_truth):
        return NOT_AVAILABLE
    low, high = min(generated.min(), ground_truth.min()), max(generated.max(), ground_truth.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, int(bins) + 1)
    gen_hist, _ = np.histogram(generated, bins=edges)
    gt_hist, _ = np.histogram(ground_truth, bins=edges)
    return gen_hist.astype(float), gt_hist.astype(float), edges


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if not p.sum() or not q.sum():
        return NOT_AVAILABLE
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    left = np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
    right = np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0]))
    return float(0.5 * (left + right))


def compare_samples(generated, ground_truth, bins=20):
    histograms = common_histograms(generated, ground_truth, bins)
    if histograms == NOT_AVAILABLE:
        return NOT_AVAILABLE
    gen_hist, gt_hist, edges = histograms
    generated = np.asarray(generated, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    generated = generated[np.isfinite(generated)]
    ground_truth = ground_truth[np.isfinite(ground_truth)]
    result = {
        "Jensen_Shannon_divergence": js_divergence(gen_hist, gt_hist),
        "Wasserstein_distance": float(wasserstein_distance(generated, ground_truth)),
        "KS_statistic": float(ks_2samp(generated, ground_truth).statistic),
        "mean_difference": float(np.mean(generated) - np.mean(ground_truth)),
        "std_difference": float(np.std(generated) - np.std(ground_truth)),
        "common_bin_edges": edges.tolist(),
        "generated_histogram": gen_hist.tolist(), "ground_truth_histogram": gt_hist.tolist(),
    }
    for percentile in (10, 50, 90, 99):
        result["P%d_difference" % percentile] = float(np.percentile(generated, percentile) - np.percentile(ground_truth, percentile))
    return result


def _series(kinematics, name):
    values = []
    for metrics in kinematics.values():
        values.extend(metrics.get("series", {}).get(name, []))
    return np.asarray(values, dtype=float)


def _track_scalar(tracks, kind):
    values = []
    for track in tracks:
        start = int(track.metadata.get("history_frames") or 0)
        idx = np.flatnonzero(track.valid_mask[start:]) + start
        if not len(idx):
            continue
        if kind == "trajectory_length":
            adjacent = idx[1:][idx[1:] == idx[:-1] + 1]
            values.append(float(np.sum(np.linalg.norm(track.positions[adjacent] - track.positions[adjacent - 1], axis=1))))
        else:
            values.append(float(np.linalg.norm(track.positions[idx[-1]] - track.positions[idx[0]])))
    return values


def _interaction_samples(interaction, name):
    values = []
    for pair in interaction.get("pairs", []):
        value = pair.get(name)
        if isinstance(value, (int, float)):
            values.append(value)
        elif isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            values.append(value["mean"])
    return values


def evaluate_distributions(generated, ground_truth, gen_kin, gt_kin, gen_interaction, gt_interaction,
                           gen_map, gt_map, config, scene_count=1):
    samples = {
        "speed": (_series(gen_kin, "speed"), _series(gt_kin, "speed")),
        "longitudinal_acceleration": (_series(gen_kin, "longitudinal_acceleration"), _series(gt_kin, "longitudinal_acceleration")),
        "lateral_acceleration": (_series(gen_kin, "lateral_acceleration"), _series(gt_kin, "lateral_acceleration")),
        "jerk": (_series(gen_kin, "longitudinal_jerk"), _series(gt_kin, "longitudinal_jerk")),
        "yaw_rate": (_series(gen_kin, "yaw_rate"), _series(gt_kin, "yaw_rate")),
        "curvature": (_series(gen_kin, "curvature"), _series(gt_kin, "curvature")),
        "trajectory_length": (_track_scalar(generated, "trajectory_length"), _track_scalar(ground_truth, "trajectory_length")),
        "endpoint_displacement": (_track_scalar(generated, "endpoint_displacement"), _track_scalar(ground_truth, "endpoint_displacement")),
        "minimum_distance": (_interaction_samples(gen_interaction, "minimum_distance"), _interaction_samples(gt_interaction, "minimum_distance")),
        "TTC": (_interaction_samples(gen_interaction, "minimum_ttc"), _interaction_samples(gt_interaction, "minimum_ttc")),
        "THW": (_interaction_samples(gen_interaction, "thw"), _interaction_samples(gt_interaction, "thw")),
        "PET": (_interaction_samples(gen_interaction, "pet"), _interaction_samples(gt_interaction, "pet")),
        "accepted_gap": ([], []),
    }
    def lane_values(result):
        values = []
        for item in result.get("per_agent", {}).values():
            metric = item.get("lane_center_deviation")
            if isinstance(metric, dict):
                values.append(metric["mean"])
        return values
    samples["lane_deviation"] = (lane_values(gen_map), lane_values(gt_map))
    output = {name: compare_samples(a, b, config.get("bins", 20)) for name, (a, b) in samples.items()}
    output["sample_validity"] = "valid_low_sample" if scene_count < int(config.get("minimum_stable_scene_count", 30)) else "stable_sample_size"
    output["sample_scene_count"] = scene_count
    output["caveat"] = "A single scene provides sample statistics, not a stable distribution conclusion." if scene_count == 1 else None
    return output

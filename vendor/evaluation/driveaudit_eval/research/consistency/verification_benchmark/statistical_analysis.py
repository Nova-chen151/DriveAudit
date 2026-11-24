from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json


def scene_bootstrap(scene_values: dict[str, float], repeats: int, confidence: float, seed: int) -> dict[str, Any]:
    values = np.asarray(list(scene_values.values()), float); alpha = (1 - confidence) / 2
    if not len(values): return {"status": "not_available_empty", "bootstrap_unit": "scene"}
    rng = np.random.default_rng(seed)
    samples = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(repeats)])
    return {"status": "computed_low_sample" if len(values) < 20 else "computed", "bootstrap_unit": "scene",
            "scene_count": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "confidence_level": confidence, "bootstrap_repeats": repeats,
            "confidence_interval": [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))]}


def paired_comparison(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    scenes = sorted(set(left) & set(right)); differences = np.asarray([left[s] - right[s] for s in scenes], float)
    if len(scenes) < 2: return {"status": "not_available_insufficient_paired_scenes", "scene_count": len(scenes)}
    wilcoxon = stats.wilcoxon(differences) if np.any(differences) else None; ttest = stats.ttest_1samp(differences, 0)
    return {"status": "computed", "scene_count": len(scenes), "mean_difference": float(np.mean(differences)),
            "effect_size_d": float(np.mean(differences) / max(np.std(differences, ddof=1), 1e-9)),
            "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon else 1.0, "paired_t_p": float(ttest.pvalue),
            "win_tie_loss": [int((differences > 0).sum()), int((differences == 0).sum()), int((differences < 0).sum())]}


def summarize_scenes(scene_rows: list[dict[str, Any]], config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    metrics = ("projection_support_rate", "residual_flow_epe", "residual_flow_cosine", "camera_support_recall", "multiview_contradiction")
    result = {metric: scene_bootstrap({row["scene_id"]: row[metric] for row in scene_rows if isinstance(row.get(metric), (int, float))},
                                      int(config["statistics"]["bootstrap_repeats"]),
                                      float(config["statistics"]["confidence_level"]),
                                      int(config["benchmark"]["random_seed"])) for metric in metrics}
    write_json(Path(output) / "scene_statistics.json", result); return result


def severity_analysis(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    order = {"positive": 0, "mild": 1, "medium": 2, "severe": 3}; results = {}
    for family in sorted({row.get("negative_family") for row in rows if row.get("negative_family")}):
        items = [row for row in rows if row.get("negative_family") == family and row.get("severity") in order]
        means = {severity: float(np.mean([row[score_key] for row in items if row["severity"] == severity]))
                 for severity in ("mild", "medium", "severe") if any(row["severity"] == severity for row in items)}
        if len(means) < 3: continue
        x = np.asarray([1, 2, 3], float); y = np.asarray([means[s] for s in ("mild", "medium", "severe")])
        spearman, kendall = ((float("nan"), float("nan")) if np.ptp(y) < 1e-12 else
                             (stats.spearmanr(x, y).statistic, stats.kendalltau(x, y).statistic))
        results[family] = {"mild": means["mild"], "medium": means["medium"], "severe": means["severe"],
                           "spearman": float(spearman) if math.isfinite(spearman) else None,
                           "kendall": float(kendall) if math.isfinite(kendall) else None,
                           "monotonic_pair_accuracy": float(np.mean([y[0] > y[1], y[1] > y[2], y[0] > y[2]])),
                           "violation_count": int(not y[0] > y[1]) + int(not y[1] > y[2]),
                           "effect_size": float((y[0] - y[2]) / max(np.std(y), 1e-9))}
    return {"families": results, "family_count": len(results),
            "monotonic_pass_rate": float(np.mean([v["violation_count"] == 0 for v in results.values()])) if results else None}


def sensitivity_matrix(positive: list[dict[str, Any]], negatives: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    expected = {"world_x_shift": {"residual_flow_epe", "residual_flow_cosine"}, "world_y_shift": {"residual_flow_epe", "residual_flow_cosine"},
                "time_shift": {"residual_flow_epe", "residual_flow_cosine"}, "motion_direction_reverse": {"residual_flow_cosine"},
                "size_scale": {"scale_change_error"}, "repeated_static_frame": {"residual_flow_epe", "residual_flow_cosine"},
                "camera_stream_swap": {"residual_flow_epe", "residual_flow_cosine"}}
    positive_means = {key: np.mean([float(row.get(key, 0)) for row in positive]) for key in metric_keys}
    for family in sorted({row.get("negative_family") for row in negatives if row.get("negative_family")}):
        items = [row for row in negatives if row.get("negative_family") == family]
        for key in metric_keys:
            values = np.asarray([float(row.get(key, 0)) for row in items]); base = positive_means[key]
            effect = float((np.mean(values) - base) / max(np.std(values), 1e-6))
            wanted = key in expected.get(family, set())
            rows.append({"metric": key, "corruption": family, "normalized_effect_size": effect,
                         "expected_sensitivity": wanted, "observed_sensitivity": abs(effect) >= .5,
                         "false_sensitivity": bool(not wanted and abs(effect) >= .5),
                         "specificity_score": float(1.0 if wanted == (abs(effect) >= .5) else 0.0)})
    return rows

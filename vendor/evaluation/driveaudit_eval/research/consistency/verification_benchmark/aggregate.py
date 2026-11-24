from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import load_config as load_tcvv_config, load_pair

from driveaudit_eval.research.consistency.verification_benchmark.build_benchmark_manifest import discover
from driveaudit_eval.research.consistency.verification_benchmark.calibrated_verifier import discrimination, fit, predict
from driveaudit_eval.research.consistency.verification_benchmark.common import load_yaml, mean, read_json, write_csv, write_json, write_jsonl
from driveaudit_eval.research.consistency.verification_benchmark.false_support_filter import filter_rows
from driveaudit_eval.research.consistency.verification_benchmark.human_audit_export import export as export_human
from driveaudit_eval.research.consistency.verification_benchmark.human_audit_import import import_annotations
from driveaudit_eval.research.consistency.verification_benchmark.leaderboard_protocol import determine_status, write_protocol
from driveaudit_eval.research.consistency.verification_benchmark.multi_scene_runner import run_scenes
from driveaudit_eval.research.consistency.verification_benchmark.negative_pair_generator import (IMAGE_FAMILIES, TRAJECTORY_FAMILIES, build_negative_manifest,
                                      image_negative_rows, trajectory_negative_rows)
from driveaudit_eval.research.consistency.verification_benchmark.pair_builder import build_positive_pairs
from driveaudit_eval.research.consistency.verification_benchmark.report import write_report
from driveaudit_eval.research.consistency.verification_benchmark.residual_motion import enrich_rows
from driveaudit_eval.research.consistency.verification_benchmark.score_calibration import fit_percentile_mapping
from driveaudit_eval.research.consistency.verification_benchmark.statistical_analysis import sensitivity_matrix, severity_analysis, summarize_scenes
from driveaudit_eval.research.consistency.verification_benchmark.visualization import failure_cases, plots


def _support_increased(rule, raw):
    return rule > raw if isinstance(rule, (int, float)) and isinstance(raw, (int, float)) else None


class Runner:
    def __init__(self, config_path: str, force: bool = False, resume: bool = False):
        self.config = load_yaml(config_path); self.output = Path(self.config["output"]); self.output.mkdir(parents=True, exist_ok=True)
        self.tcvv_config = load_tcvv_config(self.config["inputs"]["tcvv_config"]); self.pair = load_pair(self.tcvv_config)
        self.force, self.resume = force, resume

    def audit(self) -> dict[str, Any]:
        result = discover(self.config)
        audit = {**result, "implementation_dependencies": {"opencv": True, "sklearn": True, "scipy": True},
                 "current_holdout": self.config["benchmark"]["current_holdout_scene"],
                 "future_gt_used": False, "planning_or_closed_loop_used": False}
        write_json(self.output / "manifest/input_audit.json", audit); return audit

    def pairs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        discovery = self.audit(); positive = build_positive_pairs(discovery, self.config)
        negatives = build_negative_manifest(discovery, positive["splits"], self.config)
        write_jsonl(self.output / "manifest/benchmark_pairs.jsonl", positive["rows"] + negatives["rows"])
        write_json(self.output / "calibration_pairs/status.json", {"calibration_scenes": [scene for scene, split in positive["splits"].items() if split == "calibration"],
                                                                    "submission_pairs_used_as_positive": False,
                                                                    "status": "available" if any(split == "calibration" for split in positive["splits"].values()) else "not_available_no_independent_calibration_scene"})
        return discovery, positive, negatives

    def background(self) -> tuple[dict[str, Any], dict[str, Any]]:
        residual = self.output / "residual_motion"
        history_path, future_path = residual / "history_residual_rows.json", residual / "future_residual_rows.json"
        if not self.force and history_path.is_file() and future_path.is_file():
            return {"rows": read_json(history_path), "summary": read_json(residual / "history_summary.json")}, {"rows": read_json(future_path), "summary": read_json(residual / "future_summary.json")}
        base = Path(self.config["inputs"]["tcvv_results"])
        history = enrich_rows(self.pair, read_json(base / "history_calibration/local_visual_evidence/per_window.json"),
                              read_json(base / "history_calibration/expected_motion/expected_motion.json"), self.config, residual, "history")
        future = enrich_rows(self.pair, read_json(base / "local_visual_evidence/per_window.json"),
                             read_json(base / "expected_motion/expected_motion.json"), self.config, residual, "future")
        return history, future

    def validate(self) -> dict[str, Any]:
        discovery, positive_manifest, negative_manifest = self.pairs(); history, future = self.background()
        positives = [{**row, "label": 1, "scene_id": row["pair_id"], "negative_family": None, "severity": "positive"}
                     for row in filter_rows(history["rows"], self.config)]
        trajectory = trajectory_negative_rows(history["rows"], self.config)
        base = Path(self.config["inputs"]["tcvv_results"])
        image = image_negative_rows(self.pair, history["rows"], read_json(base / "history_calibration/expected_motion/expected_motion.json"),
                                    self.tcvv_config, self.config, self.output / "negative_pairs/image_corruptions")
        negatives = filter_rows(trajectory + image, self.config)
        for row in negatives: row["scene_id"] = self.config["benchmark"]["current_holdout_scene"]
        negative_manifest["trajectory_negative_sample_count"] = len(trajectory)
        negative_manifest["image_negative_sample_count"] = len(image)
        negative_manifest["evaluated_negative_window_count"] = len(negatives)
        write_json(self.output / "positive_pairs/history_positive_features.json", positives)
        write_json(self.output / "negative_pairs/negative_features.json", negatives)
        labels = [1] * len(positives) + [0] * len(negatives); scores = [row["tcvv_rule_support"] for row in positives + negatives]
        rule_metrics = discrimination(labels, scores, float(self.config["verifiers"]["rule_based"]["support_threshold"]))
        rule_metrics["evaluation_scope"] = "synthetic_holdout_diagnostic"; rule_metrics["used_for_threshold_selection"] = False
        hard = [row for row in negatives if row.get("negative_family") in set(IMAGE_FAMILIES) | {"world_x_shift", "world_y_shift", "time_shift", "motion_direction_reverse"}]
        threshold = float(self.config["verifiers"]["rule_based"]["support_threshold"])
        rule_metrics["hard_negative_count"] = len(hard)
        rule_metrics["hard_negative_false_positive_rate"] = mean(float(row["tcvv_rule_support"] >= threshold) for row in hard)
        by_family = []
        for family in sorted(set(TRAJECTORY_FAMILIES + IMAGE_FAMILIES)):
            subset = [row for row in negatives if row.get("negative_family") == family]
            values = discrimination([1] * len(positives) + [0] * len(subset),
                                    [row["tcvv_rule_support"] for row in positives + subset],
                                    float(self.config["verifiers"]["rule_based"]["support_threshold"]))
            by_family.append({"negative_family": family, **values})
        write_json(self.output / "calibrated_verifier/discrimination_metrics.json", rule_metrics)
        write_csv(self.output / "calibrated_verifier/per_negative_family.csv", by_family)
        splits = positive_manifest["splits"]; calibration_scenes = {scene for scene, split in splits.items() if split == "calibration"}
        model = fit(positives + negatives, calibration_scenes, self.output / "calibrated_verifier",
                    int(self.config["verifiers"]["calibrated"]["minimum_calibration_scenes"]))
        calibrated_scores = predict(model, positives + negatives)
        calibrated_metrics = (discrimination(labels, calibrated_scores) if calibrated_scores is not None else
                              {key: value for key, value in model.items() if key != "model"})
        severity = severity_analysis(negatives, "tcvv_rule_support")
        matrix = sensitivity_matrix(positives, negatives, ["residual_flow_epe", "residual_flow_cosine", "scale_change_error", "tcvv_rule_support"])
        write_json(self.output / "statistics/severity_monotonicity.json", severity)
        write_csv(self.output / "statistics/metric_corruption_sensitivity_matrix.csv", matrix)
        mapping = fit_percentile_mapping([row["tcvv_rule_support"] for row in positives], [row["tcvv_rule_support"] for row in negatives],
                                         len(calibration_scenes), [self.config["inputs"]["pair_manifest"]], self.output / "leaderboard")
        holdout = self.holdout(future)
        future_filtered = filter_rows(future["rows"], self.config)
        reason_counts = defaultdict(int)
        for row in future_filtered:
            for reason in row["rejection_reason"]: reason_counts[reason] += 1
        false_support = {"raw_supported_rule_rejected": sum(row.get("visual_supported") and not row["support_decision"] for row in future_filtered),
                         "raw_rejected_rule_accepted": sum(not row.get("visual_supported") and row["support_decision"] for row in future_filtered),
                         "mean_false_support_probability": mean(row["false_support_probability"] for row in future_filtered),
                         "rejection_reason_counts": dict(reason_counts),
                         "future_entrant_rule_support_increase_warning": _support_increased(holdout["future_entrant"]["rule_support"], holdout["future_entrant"]["raw_conditional_support"])}
        write_json(self.output / "false_support_analysis/summary.json", false_support)
        scene_rows = [{"scene_id": holdout["scene_id"], "projection_support_rate": holdout["tcvv_rule"]["projection_support_rate"],
                       "residual_flow_epe": future["summary"]["residual_flow_epe"], "residual_flow_cosine": future["summary"]["residual_flow_cosine"],
                       "camera_support_recall": holdout["existing_tcvv"]["camera_support_recall"],
                       "multiview_contradiction": holdout["existing_tcvv"]["multiview_contradiction"]}]
        statistics = summarize_scenes(scene_rows, self.config, self.output / "statistics")
        multiscene = run_scenes(discovery["scenes"], lambda scene: scene_rows[0] if scene["pair_id"] == holdout["scene_id"] else
                                (_ for _ in ()).throw(FileNotFoundError("TCVV per-scene cache missing")),
                                self.output / "multi_scene_results", self.resume)
        human = export_human(self.pair, positives + negatives, self.output / "human_audit", 50,
                             render_media=self.config.get("retain_intermediates", True))
        human_import = import_annotations(self.output / "human_audit/annotations.csv", self.output / "human_audit")
        human.update(human_import)
        leaderboard = determine_status(discovery["strictly_pairable_scene_count"], sum(split == "validation" for split in splits.values()),
                                       rule_metrics, severity["monotonic_pass_rate"], human, self.config)
        write_protocol(leaderboard, self.output / "leaderboard")
        if self.config.get("retain_intermediates", True):
            plots(positives, negatives, self.output / "visualizations")
            failure_cases(self.pair, future_filtered, negatives, self.output / "visualizations/failure_cases")
        metrics = {"implementation_status": "partial" if discovery["strictly_pairable_scene_count"] < 2 else "completed",
                   "protocol": "tcvv_bench_v1", "discovery": discovery, "splits": splits,
                   "positive_pairs": {"history_positive_window_count": len(positives), "submission_as_positive": False,
                                      "paired_future_gt_positive_count": 0}, "negative_pairs": negative_manifest,
                   "background_compensation": future["summary"], "history_background_compensation": history["summary"],
                   "false_support_analysis": false_support,
                   "tcvv_rule_discrimination": rule_metrics, "per_negative_family": by_family,
                   "tcvv_calibrated": calibrated_metrics, "severity_monotonicity": severity,
                   "specificity_matrix_rows": len(matrix), "score_calibration": mapping, "multi_scene_results": multiscene,
                   "scene_statistics": statistics, "current_holdout": holdout, "human_validation": human,
                   "leaderboard": leaderboard,
                   "diagnostics": {"mtpc": "long_horizon_tracking_diagnostic_not_in_primary_score",
                                   "bevformer": "perception_utility_diagnostic_not_in_primary_score"},
                   "future_gt_used": False, "planning_or_closed_loop_used": False}
        write_json(self.output / "final_metrics.json", metrics); write_report(metrics, self.output / "final_report.md")
        return metrics

    def holdout(self, future: dict[str, Any] | None = None) -> dict[str, Any]:
        future = future or self.background()[1]; filtered = filter_rows(future["rows"], self.config)
        raw = read_json(Path(self.config["inputs"]["tcvv_results"]) / "final_metrics.json")
        original = raw["scorecard"]["primary_metrics"]
        supported_raw = mean(float(row.get("visual_supported", False)) for row in filtered)
        supported_rule = mean(float(row["support_decision"]) for row in filtered)
        by_camera = []
        for camera in sorted({row["camera_name"] for row in filtered}):
            selected = [row for row in filtered if row["camera_name"] == camera]
            by_camera.append({"camera": camera, "windows": len(selected), "raw_support": mean(float(row.get("visual_supported", False)) for row in selected),
                              "rule_support": mean(float(row["support_decision"]) for row in selected),
                              "residual_epe": mean(row["residual_flow_epe"] for row in selected),
                              "residual_cosine": mean(row["residual_flow_cosine"] for row in selected),
                              "background_leakage": mean(row["background_flow_leakage_ratio"] for row in selected)})
        result = {"scene_id": self.config["benchmark"]["current_holdout_scene"], "split": "test", "used_for_calibration": False,
                  "existing_tcvv": {"projection_support_rate": original["projection_support_rate"],
                                    "raw_flow_epe": original["local_flow_epe"], "raw_flow_cosine": original["flow_cosine_similarity"],
                                    "camera_support_recall": original["camera_support_recall"],
                                    "multiview_contradiction": original["multiview_contradiction"]},
                  "tcvv_rule": {"projection_support_rate": supported_rule, "raw_support_recomputed": supported_raw,
                                "residual_flow_epe": future["summary"]["residual_flow_epe"],
                                "residual_flow_cosine": future["summary"]["residual_flow_cosine"],
                                "background_leakage": future["summary"]["background_leakage"],
                                "mean_false_support_probability": mean(row["false_support_probability"] for row in filtered),
                                "false_support_reduction": float((supported_raw or 0) - (supported_rule or 0)),
                                "raw_supported_rejected_count": sum(row.get("visual_supported") and not row["support_decision"] for row in filtered)},
                  "tcvv_calibrated": {"status": "experimental_not_fitted_insufficient_calibration_scenes"},
                  "per_camera": by_camera, "future_entrant": {"raw_conditional_support": raw["future_entrant"]["conditional_entrant_support_rate"],
                                  "rule_support": mean(float(row["support_decision"]) for row in filtered if not row.get("history_anchored"))}}
        write_json(self.output / "multi_scene_results/current_sample_001/holdout.json", result)
        write_csv(self.output / "false_support_analysis/current_holdout_per_camera.csv", by_camera)
        return result




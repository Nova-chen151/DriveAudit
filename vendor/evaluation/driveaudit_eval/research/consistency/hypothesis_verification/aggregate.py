from __future__ import annotations

from driveaudit_eval.consistency.config import load_config as read_config, save_snapshot

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from driveaudit_eval.consistency.mtpc.common import box_iou
from driveaudit_eval.research.consistency.trajectory_verification.common import load_config as load_tcvv_config, load_geometry, load_pair, write_csv, write_json
from driveaudit_eval.research.consistency.verification_benchmark.calibrated_verifier import discrimination
from driveaudit_eval.research.consistency.verification_benchmark.statistical_analysis import severity_analysis

from driveaudit_eval.research.consistency.hypothesis_verification.calibration import calibrate
from driveaudit_eval.research.consistency.hypothesis_verification.contamination_detector import detect as detect_contamination
from driveaudit_eval.research.consistency.hypothesis_verification.hypothesis_competition import compete
from driveaudit_eval.research.consistency.hypothesis_verification.motion_verifier import verify as verify_motion
from driveaudit_eval.research.consistency.hypothesis_verification.observed_motion import extract
from driveaudit_eval.research.consistency.hypothesis_verification.presence_verifier import verify as verify_presence
from driveaudit_eval.research.consistency.hypothesis_verification.robust_likelihood import evaluate
from driveaudit_eval.research.consistency.hypothesis_verification.scene_discovery import discover
from driveaudit_eval.research.consistency.hypothesis_verification.scope_registry import PRIMARY_EXPECTED, scope


KEYS = ("agent_id", "camera_name", "frame_index", "target_frame_index")
PRESENCE_NEGATIVES = {
    "object_region_occlusion", "object_region_dropout", "wrong_view_replacement",
    "camera_stream_swap", "camera_order_permutation", "local_texture_copy",
}
HARD_MOTION_NEGATIVES = {
    "repeated_static_frame", "motion_freeze", "frozen_object_region", "motion_direction_flip",
    "motion_direction_reverse", "camera_stream_swap", "camera_order_permutation", "wrong_view_replacement",
    "local_texture_copy", "temporal_reverse", "frame_skip", "inconsistent_camera_delay",
    "roi_only_background", "road_texture_only", "building_edge_only", "strong_ego_motion_static_background",
}


class Runner:
    def __init__(self, config_path: str | Path, force: bool = False, resume: bool = False):
        self.config_path = Path(config_path.get("_config_source", "tools/configs/research.yaml")) if isinstance(config_path, dict) else Path(config_path)
        self.config = read_config(config_path)
        self.output = Path(self.config["output"]); self.output.mkdir(parents=True, exist_ok=True)
        self.tcvv_config = load_tcvv_config(self.config["inputs"]["tcvv_config"])
        self.pair = load_pair(self.tcvv_config)
        self.force, self.resume = force, resume
        geometry = load_geometry(self.pair["image_sequence"]["conditions_root"])
        self.calibrations = {(int(row["frame_index"]), row["camera_name"]): row for row in geometry["calibrations"]}

    def audit(self) -> dict[str, Any]:
        base = Path(self.config["inputs"]["tcvv_bench_results"]) / "baselines/tcvv_bench_v1_rule_based"
        required = ("final_metrics.json", "discrimination_metrics.json", "per_negative_family.csv",
                    "false_support_summary.json", "severity_monotonicity.json", "config_snapshot.yaml")
        files = {name: _sha256(base / name) if (base / name).is_file() else None for name in required}
        result = {
            "version": self.config["version"], "old_baseline": "tcvv_bench_v1_rule_based",
            "old_baseline_files": files, "old_baseline_frozen": all(files.values()),
            "simple_ring_residual_status": "diagnostic_invalid_for_ranking",
            "simple_ring_residual_for_ranking": False, "future_gt_used": False,
            "long_propagation_used": False, "planning_pdms_or_closed_loop_used": False,
            "holdout_scene": self.config["holdout"]["scene_id"], "holdout_excluded_from_calibration": True,
        }
        snapshot = self.output / "baseline_reference/tcvv_h_config_snapshot.yaml"; snapshot.parent.mkdir(parents=True, exist_ok=True)
        save_snapshot(self.config_path, snapshot, self.config.get("retain_intermediates", True)); result["tcvv_h_config_sha256"] = _sha256(snapshot)
        write_json(self.output / "baseline_reference/audit.json", result)
        return result

    def scene_discovery(self) -> dict[str, Any]:
        return discover(self.config, self.output / "scene_discovery")

    def observed(self, which: str) -> list[dict[str, Any]]:
        base = Path(self.config["inputs"]["tcvv_results"])
        if which == "history":
            evidence = _read(base / "history_calibration/local_visual_evidence/per_window.json")
            expectations = _read(base / "history_calibration/expected_motion/expected_motion.json")
            folder, scope_name = self.output / "history_positive", "history"
        else:
            evidence = _read(base / "local_visual_evidence/per_window.json")
            expectations = _read(base / "expected_motion/expected_motion.json")
            folder, scope_name = self.output / "current_holdout", "holdout"
        cached = folder / f"{scope_name}_observed_motion.json"
        if self.resume and not self.force and cached.is_file(): return _read(cached)
        return extract(self.pair, evidence, expectations, self.config, folder, scope_name)

    def hypotheses(self, which: str) -> list[dict[str, Any]]:
        folder = self.output / ("history_positive" if which == "history" else "current_holdout")
        cached = folder / f"{which}_hypothesis_rows.json"
        if self.resume and not self.force and cached.is_file(): return _read(cached)
        observed = self.observed(which)
        overlaps = _neighbor_overlaps(observed)
        rows = []
        for row in observed:
            key = _key(row)
            source = self.calibrations.get((int(row["frame_index"]), row["camera_name"]))
            target = self.calibrations.get((int(row["target_frame_index"]), row["camera_name"]))
            if source is None or target is None: continue
            contamination = detect_contamination(row, overlaps.get(key, 0.0), self.config["contamination"])
            competition = compete(row, source, target, self.config)
            presence = verify_presence(row, contamination, self.config["presence"])
            motion = verify_motion(row, competition, presence, contamination, self.config["motion"])
            rows.append({**row, **contamination, **competition, **presence, **motion,
                         "physical_scene_id": self.config["holdout"]["scene_id"]})
        write_json(cached, rows); write_csv(folder / f"{which}_hypothesis_rows.csv", _flat_rows(rows))
        return rows

    def negative_benchmark(self) -> dict[str, Any]:
        output = self.output / "negative_benchmark"
        cached = output / "negative_hypothesis_rows.json"
        if self.resume and not self.force and cached.is_file(): negatives = _read(cached)
        else:
            positives = self.hypotheses("history"); base_map = {_key(row): row for row in positives}
            old_residual = {_key(row): row for row in _read(Path(self.config["inputs"]["tcvv_bench_results"]) / "residual_motion/history_residual_rows.json")}
            source = _read(Path(self.config["inputs"]["tcvv_bench_results"]) / "negative_pairs/negative_features.json")
            negatives = []
            for negative in source:
                base = base_map.get(_key(negative)); residual = old_residual.get(_key(negative))
                if base is None or residual is None: continue
                negatives.append(self._negative_row(base, residual, negative))
            negatives.extend(self._background_only_rows(list(base_map.values())))
            write_json(cached, negatives); write_csv(output / "negative_hypothesis_rows.csv", _flat_rows(negatives))
        positives = self.hypotheses("history")
        metrics = self._discrimination(positives, negatives)
        write_json(output / "discrimination_metrics.json", metrics)
        families = self._per_family(negatives); write_csv(output / "per_negative_family.csv", families)
        monotonic = {
            "all_corruption_monotonicity": severity_analysis(negatives, "motion_probability"),
            "primary_scope_monotonicity": severity_analysis([r for r in negatives if r["expected_detection_scope"] == "PRIMARY_EXPECTED"], "motion_probability"),
        }
        write_json(output / "primary_scope_monotonicity.json", monotonic)
        write_csv(output / "metric_corruption_sensitivity_matrix.csv", _sensitivity(positives, negatives))
        anti = self._anti_false_support(families); write_json(output / "anti_false_support_summary.json", anti)
        return {"metrics": metrics, "families": families, "monotonicity": monotonic, "anti_false_support": anti,
                "negative_count": len(negatives), "negative_rows": negatives}

    def _negative_row(self, base: dict[str, Any], residual: dict[str, Any], negative: dict[str, Any]) -> dict[str, Any]:
        row = copy.deepcopy(base); family, severity = negative["negative_family"], negative["severity"]
        observed = np.asarray(row["observed_flow_vectors"], float)
        if negative.get("negative_generation") == "real_image_corruption_rerun" and len(observed):
            flow = np.asarray(negative.get("actual_flow", [0.0, 0.0]), float)
            row["observed_flow_vectors"] = np.repeat(flow[None], len(observed), axis=0).tolist()
            row["median_flow"] = flow.tolist()
        row = _quality_counterfactual(row, family, severity)
        competition = _counterfactual_competition(row, base, residual, negative, self.config)
        contamination = {key: base[key] for key in ("contamination_probability", "neighbor_agent_contamination", "background_edge_contamination",
                                                      "core_point_fraction", "dominant_cluster_location", "contamination_rejection_reason", "motion_cluster_count")}
        presence = verify_presence(row, contamination, self.config["presence"])
        motion = verify_motion(row, competition, presence, contamination, self.config["motion"])
        return {**row, **competition, **presence, **motion, "label": 0, "negative_family": family, "severity": severity,
                "negative_generation": negative.get("negative_generation"), "expected_detection_scope": scope(family),
                "allowed_unscorable": family in {"object_region_blur"}, "motion_probability": _motion_probability(competition, motion),
                "presence_probability": _presence_probability(presence)}

    def _background_only_rows(self, positives: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for base in positives[::max(1, len(positives) // 20)][:20]:
            best = base.get("best_background_stats", {})
            flows = best.get("expected_flow_per_point") or []
            if not flows: continue
            for family in ("roi_only_background", "road_texture_only", "building_edge_only", "strong_ego_motion_static_background"):
                row = copy.deepcopy(base); row["observed_flow_vectors"] = flows; row["median_flow"] = np.median(np.asarray(flows), axis=0).tolist()
                competition = compete(row, self.calibrations[(row["frame_index"], row["camera_name"])],
                                      self.calibrations[(row["target_frame_index"], row["camera_name"])], self.config)
                contamination = {"contamination_probability": .2, "neighbor_agent_contamination": 0.0, "background_edge_contamination": .5,
                                 "core_point_fraction": row.get("core_point_fraction", 0.0), "dominant_cluster_location": "core",
                                 "contamination_rejection_reason": ["counterfactual_background_only"], "motion_cluster_count": 1}
                presence = verify_presence(row, contamination, self.config["presence"])
                motion = verify_motion(row, competition, presence, contamination, self.config["motion"])
                rows.append({**row, **competition, **presence, **motion, "label": 0, "negative_family": family, "severity": "severe",
                             "negative_generation": "static_background_hypothesis_counterfactual", "expected_detection_scope": "PRIMARY_EXPECTED",
                             "allowed_unscorable": False, "motion_probability": _motion_probability(competition, motion),
                             "presence_probability": _presence_probability(presence)})
        return rows

    def _discrimination(self, positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
        for row in positives:
            row["motion_probability"] = _motion_probability(row, row); row["presence_probability"] = _presence_probability(row)
        presence_neg = [r for r in negatives if r["negative_family"] in PRESENCE_NEGATIVES]
        motion_neg = [r for r in negatives if r["expected_detection_scope"] == "PRIMARY_EXPECTED"]
        p_rows = [r for r in positives + presence_neg if not r.get("presence_unscorable")]
        m_rows = [r for r in positives + motion_neg if not r.get("motion_unscorable")]
        presence = discrimination([int(r.get("label", 1)) for r in p_rows], [r["presence_probability"] for r in p_rows])
        motion = discrimination([int(r.get("label", 1)) for r in m_rows], [r["motion_probability"] for r in m_rows])
        presence["unscorable_rate"] = _rate(positives + presence_neg, "presence_unscorable")
        motion["unscorable_rate"] = _rate(positives + motion_neg, "motion_unscorable")
        hard = [r for r in motion_neg if r["negative_family"] in HARD_MOTION_NEGATIVES and not r.get("motion_unscorable")]
        motion["hard_negative_fpr"] = _rate(hard, "motion_supported")
        motion["repeated_static_fpr"] = _family_rate(motion_neg, "repeated_static_frame")
        motion["background_only_fpr"] = _family_rate(motion_neg, "roi_only_background")
        motion["reverse_motion_fpr"] = _family_rate(motion_neg, "motion_direction_reverse")
        return {"presence": presence, "motion": motion,
                "hypothesis_wins_on_history_positive": _winner_rates(positives),
                "hypothesis_wins_on_primary_negatives": _winner_rates(motion_neg),
                "four_hypothesis_confusion": {"positive_expected_target": _winner_counts(positives),
                                               "negative_expected_non_target": _winner_counts(motion_neg)}}

    def _per_family(self, negatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for family in sorted({row["negative_family"] for row in negatives}):
            items = [row for row in negatives if row["negative_family"] == family]
            p = [r for r in items if not r.get("presence_unscorable")]; m = [r for r in items if not r.get("motion_unscorable")]
            wins = _winner_rates(items)
            rows.append({"negative_family": family, "expected_detection_scope": scope(family), "sample_count": len(items),
                         "presence_fpr": _rate(p, "presence_supported"), "motion_fpr": _rate(m, "motion_supported"),
                         "unscorable_rate": _rate(items, "motion_unscorable"), "target_margin_mean": _mean([r["target_hypothesis_margin"] for r in items]),
                         "target_win_rate": wins["target"], "background_win_rate": wins["background"], "zero_win_rate": wins["zero"],
                         "other_win_rate": wins["other"], "hard_rule_trigger_rate": _mean([bool(r.get("hard_rule_triggers")) for r in items])})
        return rows

    def _anti_false_support(self, families: list[dict[str, Any]]) -> dict[str, Any]:
        lookup = {row["negative_family"]: row for row in families}
        aliases = {"repeated_static_frame": "repeated_static_frame", "motion_freeze": "motion_freeze", "roi_only_background": "roi_only_background",
                   "road_texture_only": "road_texture_only", "building_edge_only": "building_edge_only", "shadow_motion": "not_available_no_real_shadow_corruption",
                   "local_texture_copy": "local_texture_copy", "neighboring_vehicle_enters_roi": "not_available_insufficient_neighbor_overlap_cases",
                   "wrong_scene_pairing": "not_available_insufficient_independent_scenes", "camera_stream_swap": "camera_stream_swap",
                   "motion_direction_reverse": "motion_direction_reverse", "object_blur": "object_region_blur",
                   "frame_hash_repeat": "repeated_static_frame", "strong_ego_motion_static_background": "strong_ego_motion_static_background",
                   "expected_static_but_observed_motion": "static_trajectory_replacement"}
        return {name: lookup.get(value, {"status": value}) for name, value in aliases.items()}

    def holdout(self) -> dict[str, Any]:
        rows = self.hypotheses("holdout"); persistent = [r for r in rows if r.get("history_anchored")]
        entrants = [r for r in rows if not r.get("history_anchored")]
        raw_supported = [r for r in persistent if r.get("old_tcvv_visual_supported")]
        summary = _state_summary(persistent)
        summary.update({"hypothesis_wins": _winner_rates(persistent), "target_margin_mean": _mean([r["target_hypothesis_margin"] for r in persistent]),
                        "target_residual_mean": _mean([r["target_stats"].get("robust_residual") for r in persistent]),
                        "target_inlier_ratio_mean": _mean([r.get("target_inlier_ratio") for r in persistent]),
                        "target_model_type_counts": dict(Counter(r.get("target_model_type") for r in persistent)),
                        "target_uncertain_window_count": sum(r.get("target_hypothesis_status") != "valid" for r in persistent),
                        "background_residual_mean": _mean([r["best_background_stats"].get("robust_residual") for r in persistent]),
                        "background_depth_win_counts": dict(Counter(r.get("best_background_depth_hypothesis") for r in persistent)),
                        "repeated_frame_detected_count": sum(float(r.get("repeated_frame_probability", 0)) >= .8 for r in persistent),
                        "frozen_region_detected_count": sum(float(r.get("frozen_region_probability", 0)) >= .8 for r in persistent),
                        "unexplained_motion_ratio": _winner_rates(persistent)["other"],
                        "contamination_rate": _mean([r["contamination_probability"] >= self.config["motion"]["maximum_contamination_probability"] for r in persistent]),
                        "static_contradiction_rate": _rate(persistent, "static_contradiction"),
                        "presence_failure_reasons": dict(Counter(r.get("presence_reason") for r in persistent)),
                        "motion_failure_reasons": dict(Counter(r.get("motion_reason") for r in persistent)),
                        "old_raw_supported_reanalysis": {"sample_count": len(raw_supported), "hypothesis_wins": _winner_rates(raw_supported),
                                                         "tcvv_h_motion_support_rate": _rate(raw_supported, "motion_supported"),
                                                         "tcvv_h_contamination_unscorable_rate": _rate(raw_supported, "motion_unscorable")},
                        "future_entrant": {"status": self.config["future_entrant"]["status"], "include_in_primary_score": False,
                                           **_state_summary(entrants), "hypothesis_wins": _winner_rates(entrants),
                                           "target_margin_mean": _mean([r["target_hypothesis_margin"] for r in entrants])},
                        "per_camera": {camera: {**_state_summary([r for r in persistent if r["camera_name"] == camera]),
                                                "hypothesis_wins": _winner_rates([r for r in persistent if r["camera_name"] == camera])}
                                       for camera in sorted({r["camera_name"] for r in persistent})}})
        old_raw = _read(Path(self.config["inputs"]["tcvv_results"]) / "final_metrics.json")
        old_rule = _read(Path(self.config["inputs"]["tcvv_bench_results"]) / "baselines/tcvv_bench_v1_rule_based/final_metrics.json")
        summary["baseline_comparison"] = {
            "existing_tcvv_raw": {"projection_support_rate": old_raw["primary_metrics"]["presence"]["projection_support_rate"],
                                  "flow_cosine": old_raw["primary_metrics"]["local_motion"]["flow_cosine_similarity"]},
            "old_tcvv_rule": {"projection_support_rate": old_rule["current_holdout"]["tcvv_rule"]["projection_support_rate"],
                              "residual_flow_cosine_invalid_for_ranking": old_rule["current_holdout"]["tcvv_rule"]["residual_flow_cosine"]},
            "tcvv_h_rule": {"presence_support_rate": summary["presence_support_rate"], "motion_support_rate": summary["motion_support_rate"]},
        }
        comparison = {"existing_tcvv_raw": old_raw, "old_tcvv_rule_based": old_rule, "tcvv_h_rule": summary,
                      "tcvv_h_calibrated": {"status": "experimental_not_fitted_insufficient_scenes"}}
        write_json(self.output / "current_holdout/metrics.json", summary); write_json(self.output / "current_holdout/comparison.json", comparison)
        return comparison

    def full(self) -> dict[str, Any]:
        audit = self.audit(); discovery = self.scene_discovery(); history = self.hypotheses("history")
        benchmark = self.negative_benchmark(); holdout = self.holdout()
        calibration_scenes = {row["physical_scene_id"] for row in discovery["accepted_scenes"]
                              if row["provenance"].get("pair_id") != self.config["holdout"]["scene_id"]}
        calibration = calibrate([], calibration_scenes, self.config, self.output / "calibration")
        readiness = _readiness(discovery, benchmark, self.config)
        result = {"implementation_status": "completed", "benchmark_readiness_status": readiness,
                  "protocol_version": self.config["version"], "audit": audit, "scene_discovery": discovery,
                  "history_positive_window_count": len(self.hypotheses("history")), "negative_benchmark": {k:v for k,v in benchmark.items() if k != "negative_rows"},
                  "calibration": calibration, "current_holdout": holdout["tcvv_h_rule"],
                  "tcvv_h_c": {"status": calibration["status"]}, "future_gt_used": False,
                  "future_entrant_in_primary_score": False, "simple_ring_residual_for_ranking": False,
                  "mtpc_role": "long_horizon_tracking_diagnostic", "bevformer_role": "perception_utility_diagnostic",
                  "enable_total_score": False, "human_validation_status": "pending_annotation"}
        write_json(self.output / "hypotheses/history_win_rates.json", _winner_rates(history))
        write_json(self.output / "hypotheses/holdout_win_rates.json", holdout["tcvv_h_rule"]["hypothesis_wins"])
        write_json(self.output / "contamination/holdout_summary.json", {"contamination_rate": holdout["tcvv_h_rule"]["contamination_rate"]})
        from driveaudit_eval.research.consistency.hypothesis_verification.visualization import export_human_audit
        human = export_human_audit(self.pair, self.hypotheses("holdout"), self.output / "human_audit",
                                   render_media=self.config.get("retain_intermediates", True))
        result["human_audit"] = human
        write_json(self.output / "final_metrics.json", result)
        return result


def _counterfactual_competition(row, base, residual, negative, config):
    source_cal = _GLOBAL_CALIBRATION[(row["frame_index"], row["camera_name"])] if _GLOBAL_CALIBRATION else None
    target_cal = _GLOBAL_CALIBRATION[(row["target_frame_index"], row["camera_name"])] if _GLOBAL_CALIBRATION else None
    if source_cal is None: raise RuntimeError("calibration context not initialized")
    result = compete(row, source_cal, target_cal, config)
    old_relative = np.asarray(residual.get("expected_relative_flow", [0.0, 0.0]), float)
    new_relative = np.asarray(negative.get("expected_relative_flow", old_relative), float)
    delta = new_relative - old_relative
    target_flow = np.asarray(result["target_expected_flow_per_point"], float)
    if len(target_flow): target_flow += delta
    stats = evaluate(row["observed_flow_vectors"], target_flow.tolist(), result["target_model_uncertainty"], config["robust_likelihood"],
                     prior=.5 if result["target_hypothesis_status"] != "valid" else 1.0)
    result["target_expected_flow_per_point"] = target_flow.tolist(); result["target_stats"] = stats; result["target_likelihood"] = stats["likelihood"]
    likelihoods = {"target": stats["likelihood"], "background": result["best_background_likelihood"],
                   "zero": result["zero_motion_likelihood"], "other": result["other_motion_likelihood"]}
    result["hypothesis_likelihoods"] = likelihoods; result["winning_hypothesis"] = max(likelihoods, key=likelihoods.get)
    result["target_hypothesis_margin"] = likelihoods["target"] - max(likelihoods[k] for k in ("background", "zero", "other"))
    result["target_vs_background_margin"] = likelihoods["target"] - likelihoods["background"]
    result["target_vs_zero_margin"] = likelihoods["target"] - likelihoods["zero"]
    result["target_vs_other_margin"] = likelihoods["target"] - likelihoods["other"]
    result["target_inlier_ratio"] = stats["inlier_ratio"]
    result["background_explanation_ratio"] = likelihoods["background"] / max(likelihoods["target"], 1e-6)
    return result


_GLOBAL_CALIBRATION: dict[tuple[int, str], dict[str, Any]] = {}


def run(config_path: str | Path, mode: str, force: bool = False, resume: bool = False) -> dict[str, Any]:
    runner = Runner(config_path, force, resume)
    _GLOBAL_CALIBRATION.clear(); _GLOBAL_CALIBRATION.update(runner.calibrations)
    if mode == "audit": return runner.audit()
    if mode == "discover_scenes": return runner.scene_discovery()
    if mode == "history_positive": return {"rows": len(runner.hypotheses("history"))}
    if mode == "hypotheses": return {"history": len(runner.hypotheses("history")), "holdout": len(runner.hypotheses("holdout"))}
    if mode == "negative_benchmark": return {k:v for k,v in runner.negative_benchmark().items() if k != "negative_rows"}
    if mode == "calibrate":
        discovery = runner.scene_discovery(); scenes = {r["physical_scene_id"] for r in discovery["accepted_scenes"] if r["provenance"].get("pair_id") != runner.config["holdout"]["scene_id"]}
        return calibrate([], scenes, runner.config, runner.output / "calibration")
    if mode == "holdout": return runner.holdout()
    if mode == "full": return runner.full()
    if mode == "human_export": return {"status": "deferred_to_visualization_export"}
    raise ValueError(f"unsupported mode: {mode}")


def _key(row): return tuple(row[k] if k in ("agent_id", "camera_name") else int(row[k]) for k in KEYS)
def _read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
from driveaudit_eval.utils.aggregation import mean as _mean
def _rate(rows, key): return _mean([bool(row.get(key)) for row in rows])
def _presence_probability(row):
    if row.get("presence_unscorable"): return 0.5
    return float(np.clip(row.get("presence_confidence", 0.0), 0, 1))
def _motion_probability(competition, motion):
    score = float(1 / (1 + np.exp(-6 * float(competition.get("target_hypothesis_margin", 0.0)))))
    if motion.get("motion_contradicted"): return min(score, .01)
    if motion.get("motion_unscorable"): return .5
    return score


def _quality_counterfactual(row, family, severity):
    level = {"mild": 1, "medium": 2, "severe": 3}.get(severity, 3)
    if family in {"repeated_static_frame", "motion_freeze", "frozen_object_region"}:
        row["frozen_region_probability"] = level / 3; row["repeated_frame_probability"] = level / 3
    if family == "object_region_blur" and level >= 2:
        row["image_quality_status"] = "unscorable_blur"; row["blur_score"] = 0.0
    return row


def _neighbor_overlaps(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["camera_name"], row["frame_index"])].append(row)
    result = {}
    for items in grouped.values():
        for row in items:
            result[_key(row)] = max([box_iou(row["source_projected_box"], other["source_projected_box"])
                                     for other in items if other["agent_id"] != row["agent_id"]] or [0.0])
    return result


def _winner_rates(rows):
    counts = Counter(row.get("winning_hypothesis") for row in rows); total = max(1, len(rows))
    return {name: counts[name] / total for name in ("target", "background", "zero", "other")}


def _winner_counts(rows):
    counts = Counter(row.get("winning_hypothesis") for row in rows)
    return {name: int(counts[name]) for name in ("target", "background", "zero", "other")}


def _family_rate(rows, family):
    items = [r for r in rows if r.get("negative_family") == family and not r.get("motion_unscorable")]
    return _rate(items, "motion_supported")


def _state_summary(rows):
    return {"sample_count": len(rows), "presence_scorable_coverage": 1 - (_rate(rows, "presence_unscorable") or 0),
            "presence_support_rate": _rate([r for r in rows if not r.get("presence_unscorable")], "presence_supported"),
            "presence_contradiction_rate": _rate([r for r in rows if not r.get("presence_unscorable")], "presence_contradicted"),
            "presence_ambiguous_rate": _rate([r for r in rows if not r.get("presence_unscorable")], "presence_ambiguous"),
            "presence_unscorable_rate": _rate(rows, "presence_unscorable"),
            "motion_scorable_coverage": 1 - (_rate(rows, "motion_unscorable") or 0),
            "motion_support_rate": _rate([r for r in rows if not r.get("motion_unscorable")], "motion_supported"),
            "motion_contradiction_rate": _rate([r for r in rows if not r.get("motion_unscorable")], "motion_contradicted"),
            "motion_ambiguous_rate": _rate([r for r in rows if not r.get("motion_unscorable")], "motion_ambiguous"),
            "motion_unscorable_rate": _rate(rows, "motion_unscorable")}


def _flat_rows(rows):
    keep = ("scope", "pair_id", "agent_id", "agent_type", "history_anchored", "camera_name", "frame_index", "target_frame_index",
            "image_quality_status", "valid_point_ratio", "core_point_fraction", "contamination_probability", "presence_state", "presence_reason",
            "motion_state", "motion_reason", "target_model_type", "target_model_uncertainty", "winning_hypothesis", "target_likelihood",
            "best_background_likelihood", "zero_motion_likelihood", "other_motion_likelihood", "target_hypothesis_margin", "target_inlier_ratio",
            "best_background_depth_hypothesis", "target_observed_cosine", "expected_motion_magnitude", "observed_motion_magnitude")
    return [{key: row.get(key) for key in keep} for row in rows]


def _sensitivity(positives, negatives):
    metrics = ("target_hypothesis_margin", "target_likelihood", "best_background_likelihood", "zero_motion_likelihood", "other_motion_likelihood",
               "target_observed_cosine", "presence_confidence", "contamination_probability")
    base = {metric: _mean([row.get(metric) for row in positives]) or 0.0 for metric in metrics}; rows = []
    expected = {"world_x_shift": {"target_hypothesis_margin", "target_likelihood"}, "world_y_shift": {"target_hypothesis_margin", "target_likelihood"},
                "time_shift": {"target_hypothesis_margin", "target_likelihood"}, "repeated_static_frame": {"zero_motion_likelihood"},
                "motion_direction_reverse": {"target_observed_cosine", "target_hypothesis_margin"}, "roi_only_background": {"best_background_likelihood"},
                "object_region_blur": {"presence_confidence"}}
    for family in sorted({r["negative_family"] for r in negatives}):
        items = [r for r in negatives if r["negative_family"] == family]
        for metric in metrics:
            values = [r.get(metric) for r in items]; value = _mean(values)
            effect = None if value is None else (value - base[metric]) / max(float(np.std([float(v) for v in values if isinstance(v, (int,float))])), 1e-6)
            wanted = metric in expected.get(family, set()); observed = effect is not None and abs(effect) >= .5
            rows.append({"metric": metric, "corruption": family, "normalized_effect_size": effect, "expected_sensitivity": wanted,
                         "observed_sensitivity": observed, "false_sensitivity": bool(observed and not wanted),
                         "specificity_score": float(wanted == observed)})
    return rows


def _readiness(discovery, benchmark, config):
    scenes = int(discovery["independent_physical_scene_count"])
    if scenes >= 20: return "benchmark_validated_multi_scene"
    families = {r["negative_family"]: r for r in benchmark["families"]}
    repeated = families.get("repeated_static_frame", {}).get("motion_fpr", 1.0)
    background = families.get("roi_only_background", {}).get("motion_fpr", 1.0)
    reverse = families.get("motion_direction_reverse", {}).get("motion_fpr", 1.0)
    monotonic = benchmark["monotonicity"]["primary_scope_monotonicity"].get("monotonic_pass_rate") or 0.0
    return "hypothesis_validated_single_scene" if scenes == 1 and repeated < .8889 and background < .5663 and reverse < .5 and monotonic > .5714 else "prototype_single_scene"

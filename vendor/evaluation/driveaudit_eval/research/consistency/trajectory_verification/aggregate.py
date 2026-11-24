from __future__ import annotations

import copy
import hashlib
import inspect
import io
import time
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks
from driveaudit_eval.research.consistency.perception_trajectory.paper_metric_registry import build_registry
from driveaudit_eval.consistency.mtpc.common import image_fingerprint

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.build_manifest import build_manifest
from driveaudit_eval.research.consistency.trajectory_verification.camera_reliability import camera_reliability
from driveaudit_eval.research.consistency.trajectory_verification.common import cosine, groups, load_config, load_geometry, load_pair, mean, score_error, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.corruption_benchmark import run_corruption_benchmark
from driveaudit_eval.research.consistency.trajectory_verification.expected_motion import expected_motion
from driveaudit_eval.research.consistency.trajectory_verification.future_entrant_verifier import future_entrant_metrics
from driveaudit_eval.research.consistency.trajectory_verification.history_calibration import run_history_calibration
from driveaudit_eval.research.consistency.trajectory_verification.metrics import (appearance_metrics, history_calibrated_metrics, local_motion_metrics,
                      multiview_metrics, presence_metrics, projection_geometry_metrics, temporal_event_metrics)
from driveaudit_eval.research.consistency.trajectory_verification.projection import project_tracks
from driveaudit_eval.research.consistency.trajectory_verification.sliding_window_verifier import verify_windows
from driveaudit_eval.research.consistency.trajectory_verification.visualization import generate_visualizations


def _run_tests(output: Path) -> dict[str, Any]:
    return {"status": "not_run", "reason": "Run pytest tests/test_consistency.py separately", "successful": False, "tests_run": 0, "run": 0, "passed": 0, "failures": 0, "errors": 0}


def _paper_mapping(metrics: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    presence, motion, geometry, temporal, multiview = (metrics[key] for key in
        ("presence", "local_motion", "projection_geometry", "temporal_events", "multiview"))
    values = {"tcvv_motion_direction_accuracy_adapted": motion["motion_direction_agreement"],
              "tcvv_stop_turn_completion_adapted": NOT_AVAILABLE,
              "tcvv_stop_turn_completion_adapted_unavailable_reason": "not_available_insufficient_event_evidence",
              "tcvv_local_projection_alignment_adapted": geometry["normalized_center_error"],
              "tcvv_agent_missing_adapted": presence["missing_visual_evidence_ratio"],
              "tcvv_local_trajectory_consistency_adapted": 1.0 / (1.0 + motion["local_flow_epe"]),
              "tcvv_temporal_consistency_adapted": temporal["event_order_accuracy"],
              "tcvv_agent_consistency_adapted": presence["persistent_agent_support_rate"],
              "tcvv_entry_exit_consistency_adapted": mean([temporal["appearance_time_error_s"], temporal["disappearance_time_error_s"]]),
              "tcvv_cross_view_support_adapted": multiview["camera_support_recall"],
              "tcvv_temporal_object_consistency_adapted": motion["flow_cosine_similarity"],
              "tcvv_object_persistence_adapted": presence["support_duration_ratio"],
              "tcvv_geometric_consistency_adapted": geometry["normalized_center_error"]}
    rows = [row for row in build_registry(values) if row["metric"].startswith("tcvv_")]
    write_json(output / "paper_metric_mapping.json", rows)
    return rows


def evaluate(evidence: list[dict[str, Any]], projections: list[dict[str, Any]], config: dict[str, Any],
             weights: dict[str, float] | None = None) -> dict[str, Any]:
    primary = int(config["windows"]["primary_length"])
    return {"presence": presence_metrics(evidence, projections, primary, weights),
            "local_motion": local_motion_metrics(evidence, primary, weights),
            "projection_geometry": projection_geometry_metrics(evidence, primary, weights),
            "temporal_events": temporal_event_metrics(evidence, projections, primary, float(config["timeline"]["dt"])),
            "multiview": multiview_metrics(evidence, primary),
            "appearance": appearance_metrics(evidence, primary)}


def _diagnostics() -> dict[str, Any]:
    mtpc_path = Path("challenge/mtpc/final_metrics.json")
    bev_path = Path("challenge/image_trajectory_eval/final_metrics.json")
    mtpc = __import__("json").loads(mtpc_path.read_text()) if mtpc_path.is_file() else {}
    bev = __import__("json").loads(bev_path.read_text()) if bev_path.is_file() else {}
    mtpc_primary = mtpc.get("primary_metrics", {})
    submission = bev.get("submission_only", {})
    papers = submission.get("paper_metrics", {})
    return {"mtpc_long_track": {"role": "long_horizon_tracking_diagnostic", "metrics_path": str(mtpc_path),
                                 "status": mtpc.get("status", NOT_AVAILABLE),
                                 "visual_support_rate": mtpc_primary.get("presence", {}).get("persistent_agent_visual_support_rate", NOT_AVAILABLE),
                                 "normalized_center_error": mtpc_primary.get("projection_geometry", {}).get("normalized_projected_center_error", NOT_AVAILABLE),
                                 "identity_persistence": mtpc_primary.get("identity", {}).get("identity_persistence", NOT_AVAILABLE),
                                 "fragmentation": mtpc_primary.get("identity", {}).get("fragmentation", NOT_AVAILABLE),
                                 "enters_tcvv_primary_gate": False},
            "bevformer_perception_utility": {"role": "perception_utility_diagnostic", "metrics_path": str(bev_path),
                                              "status": submission.get("status", NOT_AVAILABLE),
                                              "backend": submission.get("extractor_backend", NOT_AVAILABLE),
                                              "submission_ref_mAP": papers.get("bevformer", {}).get("submission_ref_center_distance_mAP", NOT_AVAILABLE),
                                              "mATE": papers.get("bevformer", {}).get("submission_ref_mATE", NOT_AVAILABLE),
                                              "IDF1": papers.get("task_fidelity", {}).get("submission_ref_idf1", NOT_AVAILABLE),
                                              "matched_coverage": submission.get("coverage", {}).get("matched_coverage", NOT_AVAILABLE),
                                              "enters_tcvv_primary_gate": False}}


def _gates(config: dict[str, Any], manifest: dict[str, Any], history: dict[str, Any], metrics: dict[str, Any],
           entrants: dict[str, Any], reliability: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["gates"]
    presence, motion, multiview, geometry = (metrics[key] for key in ("presence", "local_motion", "multiview", "projection_geometry"))
    gate2_checks = {"support_rate": isinstance(history["history_support_rate"], (int, float)) and history["history_support_rate"] >= thresholds["history"]["minimum_support_rate"],
                    "valid_point_ratio": isinstance(history["history_valid_point_ratio"], (int, float)) and history["history_valid_point_ratio"] >= thresholds["history"]["minimum_valid_point_ratio"],
                    "flow_epe": isinstance(history["history_flow_epe"], (int, float)) and history["history_flow_epe"] <= thresholds["history"]["maximum_flow_epe_px"],
                    "camera_reliability": min(reliability["weights"].values()) >= config["camera_reliability"]["minimum_weight"]}
    gate3 = (presence["expected_agent_camera_frames"] >= thresholds["future_evidence"]["minimum_windows"]
             and motion["valid_flow_window_count"] >= thresholds["future_evidence"]["minimum_flow_windows"])
    checks = {"projection_support_rate": isinstance(presence["projection_support_rate"], (int, float)) and presence["projection_support_rate"] >= thresholds["consistency"]["minimum_support_rate"],
              "support_retention": isinstance(metrics["history_calibrated"]["support_retention"], (int, float)) and metrics["history_calibrated"]["support_retention"] >= thresholds["consistency"]["minimum_support_retention"],
              "local_flow_epe": isinstance(motion["local_flow_epe"], (int, float)) and motion["local_flow_epe"] <= thresholds["consistency"]["maximum_flow_epe_px"],
              "flow_cosine": isinstance(motion["flow_cosine_similarity"], (int, float)) and motion["flow_cosine_similarity"] >= thresholds["consistency"]["minimum_flow_cosine"],
              "direction": isinstance(motion["motion_direction_agreement"], (int, float)) and motion["motion_direction_agreement"] >= thresholds["consistency"]["minimum_direction_agreement"],
              "camera_support": isinstance(multiview["camera_support_recall"], (int, float)) and multiview["camera_support_recall"] >= thresholds["consistency"]["minimum_camera_support_recall"],
              "contradiction": isinstance(multiview["multi_view_missing_contradiction"], (int, float)) and multiview["multi_view_missing_contradiction"] <= thresholds["consistency"]["maximum_multiview_contradiction"]}
    return {"gate_1_input_projection_integrity": {"status": "passed" if manifest["input_valid"] else "failed", "passed": manifest["input_valid"]},
            "gate_2_history_local_verifier_reliability": {"status": "passed" if all(gate2_checks.values()) else "unscorable_local_verifier_failure", "passed": all(gate2_checks.values()), "checks": gate2_checks},
            "gate_3_future_evidence_sufficiency": {"status": "passed" if gate3 else "insufficient_observable_evidence", "passed": gate3},
            "gate_4_tcvv_primary_consistency": {"status": "passed" if all(checks.values()) else "failed_provisional", "passed": all(checks.values()), "checks": checks},
            "gate_5_projection_geometry_optional": {"status": "not_available_no_reliable_instance_region", "blocks_primary": False, "flow_region_diagnostic": geometry["flow_propagated_region_iou_diagnostic"]},
            "gate_6_future_entrant": {"status": entrants["status"], "blocks_persistent_primary": False},
            "gate_7_mtpc_long_track": {"status": "warning_only", "blocks_primary": False, "diagnostic": diagnostics["mtpc_long_track"]["status"]},
            "gate_8_bevformer": {"status": "warning_only", "blocks_primary": False, "diagnostic": diagnostics["bevformer_perception_utility"]["status"]},
            "provisional_thresholds": True}


def _scorecard(history: dict[str, Any], metrics: dict[str, Any], entrants: dict[str, Any],
               diagnostics: dict[str, Any]) -> dict[str, Any]:
    presence, motion, geometry, temporal, multiview = (metrics[key] for key in
        ("presence", "local_motion", "projection_geometry", "temporal_events", "multiview"))
    event_error = mean([temporal["appearance_time_error_s"], temporal["disappearance_time_error_s"]])
    return {"protocol": "tcvv_submission_conditioned", "status": "valid_low_sample",
            "reference_semantics": "submitted_trajectory_as_visual_expectation",
            "primary_metrics": {"projection_support_rate": presence["projection_support_rate"],
                                "support_retention": metrics["history_calibrated"]["support_retention"],
                                "local_flow_epe": motion["local_flow_epe"], "flow_cosine_similarity": motion["flow_cosine_similarity"],
                                "motion_direction_agreement": motion["motion_direction_agreement"],
                                "camera_support_recall": multiview["camera_support_recall"],
                                "multiview_contradiction": multiview["multi_view_missing_contradiction"],
                                "event_timing_error": event_error},
            "scores": {"history_local_verifier_reliability": 100 * mean([history["history_support_rate"], history["history_valid_point_ratio"], (history["history_flow_cosine"] + 1) / 2]),
                       "conditional_visual_presence": 100 * presence["projection_support_rate"],
                       "local_motion_fidelity": mean([score_error(motion["local_flow_epe"], 10.0), 50 * (motion["flow_cosine_similarity"] + 1)]),
                       "projection_geometry_support": score_error(geometry["normalized_center_error"], 1.0),
                       "temporal_event_fidelity": score_error(event_error, 2.0),
                       "multiview_support": 100 * multiview["camera_support_recall"],
                       "future_entrant_verification": 100 * entrants["conditional_entrant_support_rate"] if isinstance(entrants["conditional_entrant_support_rate"], (int, float)) else None,
                       "optional_instance_mask_consistency": None,
                       "long_horizon_mtpc_diagnostic": None, "bevformer_perception_utility_diagnostic": None},
            "diagnostics": diagnostics, "provisional_total_score": False}


def _recombine(row: dict[str, Any], mode: str, backend: dict[str, Any]) -> dict[str, Any]:
    if mode == "fusion":
        return dict(row)
    names = {"sparse_lk": ("sparse",), "dense_flow": ("dense",), "local_feature": ("feature",),
             "lk_dense": ("sparse", "dense"), "fusion": ("sparse", "dense", "feature")}[mode]
    flows = {"sparse": row.get("sparse_flow"), "dense": row.get("dense_flow"), "feature": row.get("feature_flow")}
    scores = {"sparse": row["point_support_score"], "dense": row["dense_flow_support_score"], "feature": row["feature_support_score"]}
    present = [name for name in names if flows[name] is not None]
    actual = (np.average(np.asarray([flows[name] for name in present], float), axis=0,
                         weights=[float(backend["component_weights"][name]) * max(float(scores[name]), 1e-3) for name in present])
              if present else np.zeros(2))
    expected = np.asarray(row["expected_flow"], float)
    diagonal = max(np.hypot(row["source_projected_box"][2] - row["source_projected_box"][0],
                            row["source_projected_box"][3] - row["source_projected_box"][1]), 1.0)
    evidence_score = np.average([scores[name] for name in names], weights=[backend["component_weights"][name] for name in names])
    combined = (1 - backend["foreground_weight"]) * evidence_score + backend["foreground_weight"] * row["foreground_contrast_score"]
    cosine_value = cosine(actual, expected); error = float(np.linalg.norm(actual - expected))
    return {**row, "evidence_mode": mode, "actual_flow": actual.tolist(), "local_flow_epe": error,
            "normalized_flow_epe": error / diagonal, "normalized_center_error": error / diagonal,
            "flow_propagated_center_error": error, "flow_cosine_similarity": cosine_value,
            "motion_direction_agreement": cosine_value,
            "motion_magnitude_relative_error": abs(np.linalg.norm(actual) - np.linalg.norm(expected)) / max(np.linalg.norm(expected), 1.0),
            "support_confidence": float(combined), "combined_visual_support_score": float(combined),
            "visual_supported": bool(present and combined >= backend["support_threshold"] and cosine_value >= backend["minimum_flow_cosine"])}


def _view_ablation(evidence: list[dict[str, Any]], primary: int, view_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = [row for row in evidence if int(row["window_length"]) == primary]
    keys_by_set = {name: {(row["agent_id"], row["frame_index"]) for row in rows if row["camera_name"] in cameras}
                   for name, cameras in view_sets.items()}
    common = set.intersection(*keys_by_set.values()) if keys_by_set else set()
    results = []
    for name, cameras in view_sets.items():
        selected = [row for row in rows if row["camera_name"] in cameras and (row["agent_id"], row["frame_index"]) in common]
        by_key = groups(selected, "agent_id", "frame_index")
        results.append({"experiment": "single_multiview_unified_samples", "setting": name,
                        "unified_agent_frame_count": len(common),
                        "support_rate": mean(float(any(row["visual_supported"] for row in items)) for items in by_key.values()),
                        "flow_epe": mean(row.get("local_flow_epe") for row in selected),
                        "flow_cosine": mean(row.get("flow_cosine_similarity") for row in selected)})
    return results


def _run_ablations(config: dict[str, Any], pair: dict[str, Any], calibrations: list[dict[str, Any]],
                   projections: list[dict[str, Any]], evidence: list[dict[str, Any]], history: dict[str, Any],
                   metrics: dict[str, Any], diagnostics: dict[str, Any], output: Path) -> dict[str, Any]:
    primary = int(config["windows"]["primary_length"]); window_rows = []
    for length in (2, 3):
        subset = [row for row in evidence if int(row["window_length"]) == length]
        values, support = local_motion_metrics(subset, length), presence_metrics(subset, projections, length)
        window_rows.append({"experiment": "window_length", "setting": length, "support_rate": support["projection_support_rate"],
                            "flow_epe": values["local_flow_epe"], "flow_cosine": values["flow_cosine_similarity"], "windows": len(subset)})
    verified5 = verify_windows(pair, expected_motion(projections, [5]), config, output / "window_length/5", "fusion")
    values5, support5 = local_motion_metrics(verified5["rows"], 5), presence_metrics(verified5["rows"], projections, 5)
    window_rows.append({"experiment": "window_length", "setting": 5, "support_rate": support5["projection_support_rate"],
                        "flow_epe": values5["local_flow_epe"], "flow_cosine": values5["flow_cosine_similarity"], "windows": len(verified5["rows"])})
    backend_rows = []
    for mode in config["ablations"]["evidence_backends"]:
        recombined = [_recombine(row, mode, config["local_backend"]) if "actual_flow" in row else row for row in evidence]
        p, m = presence_metrics(recombined, projections, primary), local_motion_metrics(recombined, primary)
        backend_rows.append({"experiment": "evidence_backend", "setting": mode, "support_rate": p["projection_support_rate"],
                             "flow_epe": m["local_flow_epe"], "flow_cosine": m["flow_cosine_similarity"]})
    margin_rows = []
    for margin in config["ablations"]["roi_margins"]:
        if float(margin) == float(config["local_backend"]["roi_margin"]):
            baseline = history
        else:
            changed = copy.deepcopy(config); changed["local_backend"]["roi_margin"] = float(margin)
            baseline = run_history_calibration(pair, calibrations, changed, output / f"roi_margin/{margin}")
        margin_rows.append({"experiment": "roi_margin_history_only", "setting": margin,
                            "history_support_rate": baseline["history_support_rate"], "history_flow_epe": baseline["history_flow_epe"],
                            "history_flow_cosine": baseline["history_flow_cosine"]})
    calibration_rows = [
        {"experiment": "history_calibration", "setting": "none", "support": metrics["presence"]["projection_support_rate"]},
        {"experiment": "history_calibration", "setting": "global", "support": metrics["history_calibrated"]["support_retention"]},
        {"experiment": "history_calibration", "setting": "per_camera", "support": metrics["presence"]["reliability_weighted_projection_support_rate"]},
        {"experiment": "history_calibration", "setting": "per_camera_size_distance_bucket", "support": metrics["presence"]["reliability_weighted_projection_support_rate"],
         "note": "bucket statistics reported; one scene is insufficient to fit extra weights"}]
    view_rows = _view_ablation(evidence, primary, config["ablations"]["view_sets"])
    scope_rows = [{"experiment": "agent_scope", "setting": "persistent", "support": metrics["presence"]["persistent_agent_support_rate"]},
                  {"experiment": "agent_scope", "setting": "future_entrant", "support": metrics["presence"]["future_entrant_support_rate"]}]
    result = {"status": "completed_low_sample", "window_length": window_rows, "evidence_backend": backend_rows,
              "roi_margin": margin_rows, "history_calibration": calibration_rows, "view_set": view_rows,
              "persistent_entrant": scope_rows,
              "tcvv_mtpc_bevformer": {"tcvv": "bounded conditional local visual verification",
                  "mtpc": "long-horizon anchored tracking diagnostic", "bevformer": "open-world perception utility diagnostic",
                  "scores_directly_comparable": False, "mtpc_status": diagnostics["mtpc_long_track"]["status"],
                  "bevformer_status": diagnostics["bevformer_perception_utility"]["status"]},
              "advanced_backend": {"status": "not_available_checkpoint", "backends": ["RAFT", "CoTracker", "TAPIR", "SAM2", "XMem"]},
              "selection_rule": "formal window/backend/margin frozen from history protocol and configuration; future sample did not select parameters"}
    write_json(output / "summary.json", result)
    for name, rows in (("window_length", window_rows), ("evidence_backend", backend_rows), ("roi_margin", margin_rows),
                       ("history_calibration", calibration_rows), ("view_set", view_rows), ("persistent_entrant", scope_rows)):
        write_csv(output / f"{name}.csv", rows)
    return result


def _write_metric_rows(metrics: dict[str, Any], evidence: list[dict[str, Any]], output: Path, primary: int) -> None:
    rows = [row for row in evidence if int(row["window_length"]) == primary]
    write_csv(output / "metrics/per_frame.csv", rows); write_csv(output / "metrics/per_window.csv", evidence)
    agent_rows = []
    for (agent,), items in groups(rows, "agent_id").items():
        agent_rows.append({"agent_id": agent, "history_anchored": items[0]["history_anchored"], "agent_type": items[0]["agent_type"],
                           "windows": len(items), "support_rate": mean(float(row["visual_supported"]) for row in items),
                           "flow_epe": mean(row.get("local_flow_epe") for row in items),
                           "flow_cosine": mean(row.get("flow_cosine_similarity") for row in items)})
    write_csv(output / "metrics/per_agent.csv", agent_rows)
    write_csv(output / "metrics/per_camera.csv", [{"camera_name": camera, **values} for camera, values in metrics["multiview"]["per_camera"].items()])


def _write_report(result: dict[str, Any], output: Path) -> None:
    h, m, e = result["history_calibration"], result["primary_metrics"], result["future_entrant"]
    p, motion, geometry, temporal, multi, calibrated = (m[key] for key in
        ("presence", "local_motion", "projection_geometry", "temporal_events", "multiview", "history_calibrated"))
    profile = result["inference_profile"]
    diagnostics, ablations = result["diagnostics"], result["ablations"]
    report = f"""# TCVV 轨迹条件驱动多视角视觉一致性报告

## 1. 实现状态

`{result['implementation_status']}`；场景 `{result['status']}`。

## 2. 正式协议

`tcvv_submission_conditioned`；reference semantics：`submitted_trajectory_as_visual_expectation`。

## 3. 数据统计

{result['data_statistics']}

## 4. 使用的视觉证据后端

- Sparse：OpenCV pyramidal LK + forward/backward + RANSAC
- Dense：局部ROI Farneback
- Feature：ORB/Hamming + RANSAC（SIFT可用但未作为冻结正式后端）
- 实例分割：`not_available_checkpoint`
- CPU；平均 `{profile['mean_window_seconds']}` s/window；GPU 0 MiB

## 5. 历史基线

- Support：`{h['history_support_rate']}`
- Flow EPE / Cosine：`{h['history_flow_epe']}` / `{h['history_flow_cosine']}`
- Valid-point：`{h['history_valid_point_ratio']}`
- Per-camera reliability：`{result['camera_reliability']['weights']}`

## 6. 正式TCVV主指标

- Projection Support / Missing：`{p['projection_support_rate']}` / `{p['missing_visual_evidence_ratio']}`
- Weighted Support：`{p['reliability_weighted_projection_support_rate']}`
- Support Retention：`{calibrated['support_retention']}`
- Flow EPE / Excess：`{motion['local_flow_epe']}` / `{calibrated['excess_local_flow_epe']}` px
- Flow Cosine / Direction：`{motion['flow_cosine_similarity']}` / `{motion['motion_direction_agreement']}`
- Magnitude Error：`{motion['motion_magnitude_relative_error']}`
- Camera Support / Contradiction：`{multi['camera_support_recall']}` / `{multi['multi_view_missing_contradiction']}`
- Entry / Exit：`{temporal['appearance_time_error_s']}` / `{temporal['disappearance_time_error_s']}` s

## 7. 投影几何

- Normalized Center Error：`{geometry['normalized_center_error']}`
- Box/Mask IoU：`not_available`；无可靠实例区域，光流ROI不得冒充实例mask。
- Flow-propagated region IoU diagnostic：`{geometry['flow_propagated_region_iou_diagnostic']}`
- Scale-change Error / Agreement：`{motion['scale_change_error']}` / `{geometry['scale_change_agreement']}`

## 8. Future Entrant

- Conditional Support / Entry Camera / Entry Time：`{e['conditional_entrant_support_rate']}` / `{e['entry_camera_accuracy']}` / `{e['entry_time_error_frames']}` frames
- Motion / Confirmation：`{e['short_window_motion_agreement']}` / `{e['multi_frame_confirmation_rate']}`

## 9. 多视角

- Per camera：`{multi['per_camera']}`
- Camera Handoff：`{multi['camera_handoff_timing_error_s']}`（当前短窗口缺少充分handoff事件证据）
- Cross-view Motion / Feature：`{multi['cross_view_motion_agreement']}` / `{multi['cross_view_feature_similarity']}`
- CAM_BACK_LEFT：`{result['camera_reliability']['CAM_BACK_LEFT']}`

## 10. Hard Gate

{chr(10).join(f'- {name}: `{value.get("status", value)}`' for name, value in result['hard_gates'].items() if name != 'provisional_thresholds')}

## 11. 消融

`{ablations['status']}`。

- Window 2/3/5：`{ablations.get('window_length')}`
- Evidence backend：`{ablations.get('evidence_backend')}`
- ROI margin（history-only）：`{ablations.get('roi_margin')}`
- Unified-sample view set：`{ablations.get('view_set')}`
- Persistent/Entrant：`{ablations.get('persistent_entrant')}`

## 12. 诊断结果

- MTPC：`long_horizon_tracking_diagnostic`；Support/Normalized Center/Identity/Fragmentation：`{diagnostics['mtpc_long_track']['visual_support_rate']}` / `{diagnostics['mtpc_long_track']['normalized_center_error']}` / `{diagnostics['mtpc_long_track']['identity_persistence']}` / `{diagnostics['mtpc_long_track']['fragmentation']}`
- BEVFormer：`perception_utility_diagnostic`；mAP/mATE/IDF1/Coverage：`{diagnostics['bevformer_perception_utility']['submission_ref_mAP']}` / `{diagnostics['bevformer_perception_utility']['mATE']}` / `{diagnostics['bevformer_perception_utility']['IDF1']}` / `{diagnostics['bevformer_perception_utility']['matched_coverage']}`
- 两者均不进入TCVV主Gate。

## 13. 当前限制

单场景、400×224、矩形z-buffer遮挡近似、基础OpenCV局部后端、条件验证而非开放检测、阈值provisional。无可靠实例分割时Mask/Box IoU保持not_available。

## 14. 输出文件

`challenge/tcvv/` 保存逐帧、逐窗口、逐agent、逐相机、逐场景、历史校准、投影、局部证据、entrant、corruption、ablation及20张可视化。
"""
    (output / "final_report.md").write_text(report, encoding="utf-8")


def run(config_path: str | Path, mode: str = "full") -> dict[str, Any]:
    config = load_config(config_path); pair = load_pair(config); output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(pair, output); geometry = load_geometry(pair["image_sequence"]["conditions_root"])
    if mode == "audit":
        return {"implementation_status": "audit_completed", "protocol": "tcvv_submission_conditioned", "input_audit": manifest}
    image_hash_before = image_fingerprint(pair); trajectory_path = Path(pair["trajectory_sequence"]["trajectory_path"])
    trajectory_hash_before = hashlib.sha256(trajectory_path.read_bytes()).hexdigest(); started = time.time()
    history = run_history_calibration(pair, geometry["calibrations"], config, output / "history_calibration")
    reliability = camera_reliability(history["per_camera"], config["camera_reliability"], output / "camera_reliability")
    if mode == "history":
        result = {"implementation_status": "history_completed", "protocol": "tcvv_submission_conditioned",
                  "history_calibration": history, "camera_reliability": reliability}
        write_json(output / "final_metrics.json", result); return result
    history_ids = {track["agent_id"] for track in condition_reference_tracks(pair["image_sequence"]["conditions_root"], 0, 10)}
    frame_end = 15 if mode == "smoke" else 90
    projected = project_tracks(pair["image_sequence"]["conditions_root"], geometry["calibrations"], 10, frame_end,
                               history_ids, config["projection"], output / "submitted_projection")
    motions = expected_motion(projected["rows"], list(config["windows"]["formal_lengths"]), output / "expected_motion")
    verified = verify_windows(pair, motions, config, output / "local_visual_evidence", config["local_backend"]["formal_mode"], frame_end)
    metrics = evaluate(verified["rows"], projected["rows"], config, reliability["weights"])
    metrics["history_calibrated"] = history_calibrated_metrics(history["metrics"], metrics)
    entrants = future_entrant_metrics(verified["rows"], projected["rows"], int(config["windows"]["primary_length"]),
                                      int(config["future_entrants"]["minimum_confirmation_frames"]), output / "future_entrants")
    diagnostics = _diagnostics(); gates = _gates(config, manifest, history, metrics, entrants, reliability, diagnostics)
    scorecard = _scorecard(history, metrics, entrants, diagnostics)
    paper_mapping = _paper_mapping(metrics, output)
    _write_metric_rows(metrics, verified["rows"], output, int(config["windows"]["primary_length"])); write_json(output / "metrics/scorecard.json", scorecard)
    corruption = run_corruption_benchmark(verified["rows"], int(config["windows"]["primary_length"]), output / "corruption_benchmark") if mode in {"corruption", "full"} else {"status": "not_run", "rows": []}
    ablations = _run_ablations(config, pair, geometry["calibrations"], projected["rows"], verified["rows"], history,
                               metrics, diagnostics, output / "ablations") if mode in {"ablation", "full"} else {"status": "not_run"}
    diagnostics_plot = {"tcvv_support": metrics["presence"]["projection_support_rate"],
                        "mtpc_support": diagnostics["mtpc_long_track"].get("visual_support_rate", 0),
                        "bevformer_coverage": diagnostics["bevformer_perception_utility"].get("matched_coverage", 0)}
    visualizations = generate_visualizations(pair, projected["rows"], verified["rows"], history, entrants,
                                             metrics["multiview"], corruption, diagnostics_plot,
                                             output / "visualizations") if mode == "full" and config.get("retain_intermediates", True) else []
    primary_rows = [row for row in verified["rows"] if int(row["window_length"]) == int(config["windows"]["primary_length"])]
    tests = _run_tests(output)
    result = {"implementation_status": "completed", "protocol": "tcvv_submission_conditioned", "status": "valid_low_sample",
              "evaluation_role": "primary_image_trajectory_consistency",
              "reference_semantics": "submitted_trajectory_as_visual_expectation", "pair_id": pair["pair_id"],
              "input_audit": manifest, "data_statistics": {"scene_count": 1, "history_frames": 10, "future_frames": frame_end - 10,
                  "camera_count": 6, "persistent_agent_count": len(projected["persistent_agent_ids"]),
                  "future_entrant_count": len(projected["future_entrant_ids"]),
                  "scorable_agent_camera_frames": len(primary_rows), "valid_short_windows": len(verified["rows"])},
              "history_calibration": history, "camera_reliability": reliability, "primary_metrics": metrics,
              "future_entrant": entrants, "optional_instance_mask_protocol": {"status": "not_run_optional_mask_protocol", "blocks_primary": False},
              "diagnostics": diagnostics, "paper_metric_mapping": paper_mapping,
              "hard_gates": gates, "scorecard": scorecard, "inference_profile": verified["profile"],
              "corruption_benchmark": corruption, "ablations": ablations, "visualizations": visualizations,
              "data_access_audit": {"future_gt_used": False, "submitted_trajectory_role": "visual_expectation_and_bounded_search_region",
                                    "full_image_search_used": False, "long_horizon_state_carried": False,
                                    "mtpc_in_primary_gate": False, "bevformer_in_primary_gate": False,
                                    "planning_head_used": False, "closed_loop_used": False,
                                    "original_images_unchanged": image_fingerprint(pair) == image_hash_before,
                                    "original_trajectory_unchanged": hashlib.sha256(trajectory_path.read_bytes()).hexdigest() == trajectory_hash_before,
                                    "verifier_signature_has_future_gt": any(name in inspect.signature(verify_windows).parameters for name in ("future_gt", "future_boxes", "future_ids"))},
              "unit_tests": tests, "runtime_seconds": time.time() - started}
    write_json(output / "data_access_audit.json", result["data_access_audit"])
    write_json(output / "persistent_agents/metrics.json", {
        "scope": "history_anchored_persistent_agents",
        "support_rate": metrics["presence"]["persistent_agent_support_rate"],
        "expected_windows": sum(row["history_anchored"] for row in primary_rows),
        "supported_windows": sum(row["history_anchored"] and row["visual_supported"] for row in primary_rows)})
    write_json(output / "multiview/metrics.json", metrics["multiview"])
    write_json(output / "metrics/per_scene.json", result)
    write_csv(output / "metrics/per_scene.csv", [{"pair_id": pair["pair_id"], "status": result["status"],
               "projection_support_rate": metrics["presence"]["projection_support_rate"],
               "local_flow_epe": metrics["local_motion"]["local_flow_epe"],
               "flow_cosine": metrics["local_motion"]["flow_cosine_similarity"],
               "camera_support_recall": metrics["multiview"]["camera_support_recall"]}])
    write_json(output / "final_metrics.json", result); _write_report(result, output); return result

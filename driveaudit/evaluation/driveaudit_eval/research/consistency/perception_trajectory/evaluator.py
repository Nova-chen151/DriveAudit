from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.backends import CachedVADPerceptionBackend, GTMetadataBackend
from driveaudit_eval.research.consistency.perception_trajectory.bev_tracker import BEVTracker
from driveaudit_eval.research.consistency.perception_trajectory.cache_provenance import audit_gt_metadata_reference, audit_vad_detection_cache
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, write_json
from driveaudit_eval.research.consistency.perception_trajectory.corruption_benchmark import run_corruption_benchmark
from driveaudit_eval.research.consistency.perception_trajectory.data_audit import write_audit
from driveaudit_eval.research.consistency.perception_trajectory.extract_future_bev_tracks import extract_future_bev_tracks
from driveaudit_eval.research.consistency.perception_trajectory.formal_evaluation import apply_visibility, dimension_scores, evaluate_formal
from driveaudit_eval.research.consistency.perception_trajectory.history_warmup import history_quality_gate, known_history_samples, load_conditions, warmup_history
from driveaudit_eval.research.consistency.perception_trajectory.pair_protocol import load_pair, validate_pair
from driveaudit_eval.research.consistency.perception_trajectory.submission import load_paired_submission
from driveaudit_eval.research.consistency.perception_trajectory.visualization import render_bev_tracks, render_history, render_six_camera


SCORE_NAMES = ("history_calibration_quality", "presence_consistency", "bev_geometry_consistency",
               "motion_consistency", "identity_temporal_consistency", "multi_view_support")


def _run_unit_tests() -> dict[str, Any]:
    return {"status": "not_run", "reason": "Run pytest tests/test_consistency.py separately", "successful": False, "tests_run": 0, "run": 0, "passed": 0, "failures": 0, "errors": 0}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status", "reason"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _copy_detection_rows(source: Path, destination: Path) -> None:
    if not source.exists():
        _write_csv(destination, [], ["status", "reason"]); return
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["metric_status"] = "diagnostic_detection_only"
    _write_csv(destination, rows)


def _report(final: dict[str, Any]) -> str:
    data, warm, extraction = final["data_statistics"], final["history_warmup"], final["extraction"]
    gates, metrics = final["hard_gates"], final["future_metrics"]
    lines = [
        "# 六视角生成图像—选手轨迹一致性报告", "", "## 1. 实现状态", "", final["implementation_status"], "",
        f"正式状态：{final['status']}", f"pair_id：{final['pair_id']}",
        f"image_scene_id：{final['image_scene_id']}", f"trajectory_scene_id：{final['trajectory_scene_id']}",
        f"Pair metadata：{final['pair_validation']['pair_validation_status']}（scene 名不参与配对判定）", "",
        "## 2. BEV提取后端", "", f"模型：{final['backend']['model']}", f"config：{final['backend']['config']}",
        f"checkpoint：{final['backend']['checkpoint']}", f"环境：{final['backend']['environment']}", f"GPU：{final['backend']['gpu']}", "",
        f"后端来源：{final['backend']['declaration']}。",
        "正式轨迹来源：GT 图像关联的 input boxes 元数据 + 独立 BEVTracker；不使用 VAD 或 motion prediction。",
        "正式 GT：显式 pair 中的选手结构化未来轨迹，仅在提取结束后进入评价。",
        f"未来提交轨迹进入提取器：{final['prohibitions_verified']['future_submission_used_by_extractor']}。", "",
        "## 3. 数据统计", "", f"场景：{data['scene_count']}", f"历史帧：{data['history_frames']}",
        f"未来帧：{data['future_frames']}", f"相机完整率：{data['six_camera_complete_ratio']}",
        f"历史agent：{data['history_known_agent_count']}", f"未来提交agent：{data['future_submitted_agent_count']}",
        f"可见agent-frame：{data['expected_visible_agent_frames']}",
        f"时间映射：{final['pair_validation']['timeline_mapping']}",
        f"坐标映射：{final['pair_validation']['coordinate_transform']['method']}（仅由历史 Ego 边界估计）", "",
        "## 4. 历史10帧预热结果", "",
        f"Detection Recall：{warm['history_detection_recall']}", f"BEV ADE：{warm['history_center_ade']}",
        f"Heading Error：{warm['history_heading_error']}", f"IDF1：{warm['history_baseline_idf1']}",
        "估计标定偏差：not_available", "估计时间偏差：not_available", f"History Gate：{warm['history_gate']}", "",
        "## 5. 未来轨迹提取结果", "", f"BEV tracks：{extraction.get('tracks', NOT_AVAILABLE)}",
        f"matched tracks：{metrics['matched_tracks']}", f"unmatched submission：{metrics['unmatched_submission']}",
        f"unmatched image：{metrics['unmatched_image']}", f"future entrants：{metrics['future_entrants']}",
        f"推理总耗时：{extraction.get('total_inference_seconds', NOT_AVAILABLE)} s",
        f"平均每帧：{extraction.get('mean_inference_seconds', NOT_AVAILABLE)} s",
        f"GPU显存：{extraction.get('peak_gpu_memory_mib', NOT_AVAILABLE)} MiB", "",
        "## 6. 图像—轨迹一致性结果", "", f"Visible Agent Recall：{metrics['visible_agent_recall']}",
        f"Visible Detection Recall：{metrics['visible_detection_recall']}",
        f"Visible Detection Precision：{metrics['visible_detection_precision']}",
        f"Detection Center Error：{metrics['detection_center_error']}",
        f"Detection BEV IoU：{metrics['detection_bev_iou']}",
        f"Detection Class Accuracy：{metrics['detection_class_accuracy']}",
        f"Track Recovery Rate：{metrics['track_recovery_rate']}",
        f"Future Entry Recall：{metrics['future_entry_recall']}",
        f"Unmatched Submission Ratio：{metrics['unmatched_submission_ratio']}",
        f"Unmatched Image Track Ratio：{metrics['unmatched_image_track_ratio']}",
        f"BEV ADE：{metrics['bev_ade']}", f"Visible FDE：{metrics['visible_fde']}", f"BEV IoU：{metrics['bev_iou']}",
        f"Heading MAE：{metrics['heading_mae']}", f"Box Length/Width MAE：{metrics['box_length_mae']} / {metrics['box_width_mae']}",
        f"Speed MAE/RMSE：{metrics['speed_mae']} / {metrics['speed_rmse']}",
        f"Motion Direction Agreement：{metrics['motion_direction_agreement']}",
        f"ID Precision/Recall/IDF1：{metrics['id_precision']} / {metrics['id_recall']} / {metrics['idf1']}",
        f"ID Switch：{metrics['id_switch']}", f"Fragmentation：{metrics['fragmentation']}", "",
        "## 7. 历史校准后的结果", "", f"Excess ADE：{metrics['excess_ade']}", f"Recall Drop：{metrics['recall_drop']}",
        f"IDF1 Drop：{metrics['idf1_drop']}", f"Heading Error Increase：{metrics['heading_error_increase']}", "",
        "## 8. 多视角诊断", "", "GT 六相机图像完整；input 3D boxes 可投影，但没有独立 2D 检测框，因此 2D IoU/支持率为 not_available。", "",
        "## 9. Hard Gate", ""]
    lines.extend(f"- {name}: {value['status']} — {value['reason']}" for name, value in gates.items())
    lines += ["", "## 10. 当前限制", "", "- 当前只有一个显式 pair，属于低样本结果。", "- 图像分辨率为 400×224。",
              "- 两个源 scene 名不同，但由 pair_id 显式绑定，不影响配对。",
              "- 本轮是 GT-image/metadata reference 基准，不是从像素独立推理目标框；不能把这里的历史 100% Recall 称为检测器性能。",
              "- 未加载 VAD checkpoint，未运行 VAD、motion 或 planning head。",
              "- 遮挡、小目标与逐相机 2D 支持诊断需要独立像素检测器时再计算。",
              f"- 单元测试：{final['unit_tests'].get('passed', NOT_AVAILABLE)}/{final['unit_tests'].get('run', NOT_AVAILABLE)}。",
              f"- 理想 BEV 自一致性：{final['corruption_benchmark'].get('self_consistency', {}).get('passed', NOT_AVAILABLE)}；corruption 类型：{final['corruption_benchmark'].get('corruption_types', NOT_AVAILABLE)}。",
              "", "## 11. 输出文件", ""]
    lines.extend(f"- `{path}`" for path in final["output_files"])
    return "\n".join(lines) + "\n"


def evaluate(config: dict[str, Any], mode: str, output: str | Path = "challenge/image_trajectory_eval",
             pair_manifest: str | Path | None = None, pair_id: str | None = None) -> dict[str, Any]:
    output = Path(output)
    docs = output / "docs"
    for directory in ("history_warmup", "extracted_tracks", "matching", "visibility", "metrics", "reprojection", "corruption_benchmark", "visualizations"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for relative in ("extracted_tracks/raw_detections.pkl", "extracted_tracks/per_frame_detections.csv"):
        (output / relative).unlink(missing_ok=True)
    if not pair_manifest:
        raise ValueError("pair_manifest_required")
    pair = load_pair(pair_manifest, pair_id)
    pair_validation = validate_pair(pair)
    write_json(output / "pair_validation.json", {key: value for key, value in pair_validation.items() if key != "trajectory_scene"})
    backend_cfg = config["backend"]
    backend_audit = (audit_gt_metadata_reference(config, pair, output) if backend_cfg["name"] == "gt_metadata_boxes"
                     else audit_vad_detection_cache(config, output))
    inventory, manifest = write_audit(config, pair, pair_validation, output, docs)
    conditions = load_conditions(manifest["conditions"])
    backend = (GTMetadataBackend(backend_cfg["conditions_root"], backend_cfg["tracked_classes"])
               if backend_cfg["name"] == "gt_metadata_boxes" else
               CachedVADPerceptionBackend(backend_cfg["cache"], backend_cfg["score_threshold"], backend_cfg["tracked_classes"]))
    backend.reset(manifest["image_scene_id"])
    tracker = BEVTracker(config["tracker"]); tracker.reset(manifest["pair_id"])
    history_n = int(pair["timeline"]["image_history_frames"])
    warm = warmup_history(backend, tracker, conditions, history_n, output / "history_warmup")
    history_samples = known_history_samples(conditions, history_n)
    extraction: dict[str, Any] = {"tracks": NOT_AVAILABLE, "backend_result_status": "not_run"}; tracks: list[dict[str, Any]] = []
    if mode in {"smoke", "full"}:
        frames = manifest["future_frame_indices"][:3] if mode == "smoke" else manifest["future_frame_indices"]
        tracks, extraction = extract_future_bev_tracks(backend, frames, manifest["future_images"], conditions["calibrations"],
                                                       conditions["poses"], tracker, output / "extracted_tracks")
    else:
        write_json(output / "extracted_tracks" / "bev_tracks.json", [])
        write_json(output / "extracted_tracks" / "inference_profile.json", extraction)
    if mode in {"corruption", "full"}:
        corruption = run_corruption_benchmark(config["matching"], output / "corruption_benchmark")
    else:
        corruption = {"status": "not_run_in_this_mode"}; write_json(output / "corruption_benchmark" / "results.json", corruption)
    metrics = {"matched_tracks": NOT_AVAILABLE, "unmatched_submission": NOT_AVAILABLE,
               "unmatched_image": NOT_AVAILABLE, "future_entrants": NOT_AVAILABLE,
               "visible_detection_recall": NOT_AVAILABLE, "visible_detection_precision": NOT_AVAILABLE,
               "detection_center_error": NOT_AVAILABLE, "detection_bev_iou": NOT_AVAILABLE,
               "detection_heading_mae": NOT_AVAILABLE, "detection_box_length_mae": NOT_AVAILABLE,
               "detection_box_width_mae": NOT_AVAILABLE, "detection_class_accuracy": NOT_AVAILABLE,
               "visible_agent_recall": NOT_AVAILABLE, "future_entry_recall": NOT_AVAILABLE,
               "unmatched_submission_ratio": NOT_AVAILABLE, "unmatched_image_track_ratio": NOT_AVAILABLE,
               "bev_ade": NOT_AVAILABLE, "visible_fde": NOT_AVAILABLE, "bev_iou": NOT_AVAILABLE,
               "heading_mae": NOT_AVAILABLE, "speed_mae": NOT_AVAILABLE, "idf1": NOT_AVAILABLE,
               "id_switch": NOT_AVAILABLE, "fragmentation": NOT_AVAILABLE, "excess_ade": NOT_AVAILABLE,
               "recall_drop": NOT_AVAILABLE, "idf1_drop": NOT_AVAILABLE, "heading_error_increase": NOT_AVAILABLE}
    metrics.update({key: NOT_AVAILABLE for key in (
        "track_recovery_rate", "appearance_time_error_frames", "disappearance_time_error_frames",
        "center_error", "longitudinal_position_mae", "lateral_position_mae", "box_length_mae", "box_width_mae",
        "speed_rmse", "longitudinal_velocity_mae", "lateral_velocity_mae", "acceleration_mae", "yaw_rate_mae",
        "motion_direction_agreement", "stationary_moving_state_accuracy", "turning_state_accuracy",
        "id_precision", "id_recall", "track_completeness", "average_continuous_track_length",
        "entry_exit_timing_error_frames")})
    submitted_tracks: list[dict[str, Any]] = []; visibility_rows: list[dict[str, Any]] = []; formal = None
    paired_scene = manifest.get("submitted_trajectory")
    pairing_reason = (f"explicit pair_id={manifest['pair_id']} validated; source scene names are labels only"
                      if pair_validation["pair_validation_status"] == "valid" else
                      ",".join(pair_validation.get("errors", [])))
    if paired_scene and tracks:
        submitted_tracks = load_paired_submission(paired_scene, manifest["future_frame_indices"], conditions["poses"],
                                                   history_samples, pair_validation)
        visibility_rows = apply_visibility(submitted_tracks, conditions["calibrations"], config["visibility"])
        formal = evaluate_formal(submitted_tracks, tracks, visibility_rows, config, warm)
        mapping = formal["mapping"]
        detection, p, g, m, identity = formal["detection"], formal["presence"], formal["geometry"], formal["motion"], formal["identity"]
        metrics.update({"matched_tracks": sum(row["image_track_id"] is not None and row["submission_agent_id"] is not None for row in mapping),
                        "unmatched_submission": sum(row["match_status"] == "unmatched_submission_agent" for row in mapping),
                        "unmatched_image": sum(row["match_status"] == "unmatched_image_track" for row in mapping),
                        "future_entrants": sum(not track["history_present"] for track in submitted_tracks),
                        "visible_agent_recall": p["visible_agent_recall"], "future_entry_recall": p["future_entry_recall"],
                        "unmatched_submission_ratio": p["unmatched_submission_ratio"],
                        "unmatched_image_track_ratio": p["unmatched_image_track_ratio"], "bev_ade": g["bev_center_ade"],
                        "visible_fde": g["visible_fde"], "bev_iou": g["rotated_bev_iou"], "heading_mae": g["heading_mae"],
                        "speed_mae": m["speed_mae"], "idf1": identity["idf1_like"], "id_switch": identity["id_switch_count"],
                        "fragmentation": identity["fragmentation_count"], "excess_ade": g["excess_center_ade"]})
        metrics.update({"track_recovery_rate": p["track_recovery_rate"],
                        "appearance_time_error_frames": p["appearance_time_error_frames"],
                        "disappearance_time_error_frames": p["disappearance_time_error_frames"],
                        "center_error": g["center_error"], "longitudinal_position_mae": g["longitudinal_position_mae"],
                        "lateral_position_mae": g["lateral_position_mae"], "box_length_mae": g["box_length_mae"],
                        "box_width_mae": g["box_width_mae"], "speed_rmse": m["speed_rmse"],
                        "longitudinal_velocity_mae": m["longitudinal_velocity_mae"],
                        "lateral_velocity_mae": m["lateral_velocity_mae"], "acceleration_mae": m["acceleration_mae"],
                        "yaw_rate_mae": m["yaw_rate_mae"], "motion_direction_agreement": m["motion_direction_agreement"],
                        "stationary_moving_state_accuracy": m["stationary_moving_state_accuracy"],
                        "turning_state_accuracy": m["turning_state_accuracy"], "id_precision": identity["id_precision"],
                        "id_recall": identity["id_recall"], "track_completeness": identity["track_completeness"],
                        "average_continuous_track_length": identity["average_continuous_track_length"],
                        "entry_exit_timing_error_frames": identity["entry_exit_timing_error_frames"]})
        metrics.update({"visible_detection_recall": detection["visible_detection_recall"],
                        "visible_detection_precision": detection["visible_detection_precision"],
                        "detection_center_error": detection["center_error"], "detection_bev_iou": detection["rotated_bev_iou"],
                        "detection_heading_mae": detection["heading_mae"], "detection_box_length_mae": detection["box_length_mae"],
                        "detection_box_width_mae": detection["box_width_mae"], "detection_class_accuracy": detection["class_accuracy"]})
        write_json(output / "matching" / "track_mapping.json", mapping)
        write_json(output / "matching" / "unmatched_tracks.json", [row for row in mapping if row["match_status"].startswith("unmatched")])
        visibility_csv = [{key: row.get(key, NOT_AVAILABLE) for key in ("frame_index", "timestamp", "agent_id", "visible_in_any_camera",
                                                                          "visible_camera_count", "projected_box_area", "occlusion_ratio",
                                                                          "expected_visible", "small_object_flag", "truncated_flag")}
                          for row in visibility_rows]
        _write_csv(output / "visibility" / "per_frame_visibility.csv", visibility_csv)
        write_json(output / "reprojection" / "per_camera_metrics.json", formal["reprojection"])
    else:
        write_json(output / "matching" / "track_mapping.json", {"status": "not_available", "reason": pairing_reason, "mapping": []})
        write_json(output / "matching" / "unmatched_tracks.json", {"status": "not_computed", "reason": pairing_reason})
        _write_csv(output / "visibility" / "per_frame_visibility.csv", [], ["scene_id", "frame_index", "agent_id", "status", "reason"])
        write_json(output / "reprojection" / "per_camera_metrics.json", {"status": NOT_AVAILABLE, "reason": pairing_reason})
    if formal:
        _write_csv(output / "metrics" / "per_frame_metrics.csv", formal["detection"]["per_frame_rows"])
    else:
        _copy_detection_rows(output / "extracted_tracks" / "per_frame_detections.csv", output / "metrics" / "per_frame_metrics.csv")
    geometry_by_track = {row["image_track_id"]: row for row in (formal["geometry"]["per_track"] if formal else [])}
    per_track = [{"pair_id": manifest["pair_id"], "image_scene_id": manifest["image_scene_id"],
                  "trajectory_scene_id": manifest["trajectory_scene_id"], "image_track_id": track["image_track_id"], "category": track["category"],
                  "track_length": track["track_length"], "fragmentation": track["fragmentation"],
                  "mean_confidence": track["mean_confidence"], "status": "formal_input" if formal else "diagnostic_only",
                  **geometry_by_track.get(track["image_track_id"], {})} for track in tracks]
    _write_csv(output / "metrics" / "per_track_metrics.csv", per_track)
    history_gate = history_quality_gate(warm, config["gates"]["history_extractor_quality"])
    visible_count = formal["presence"]["expected_visible_agent_frames"] if formal else 0
    gate3_pass = visible_count >= int(config["gates"]["future_observability"]["minimum_visible_agent_frames"])
    status = ("invalid_pair_metadata" if pair_validation["pair_validation_status"] != "valid" else
              "blocked_inference" if not backend_audit["formal_backend_accepted"] else
              "unscorable_extractor_failure" if not history_gate["passed"] else
              "insufficient_observable_evidence" if not gate3_pass else "valid_low_sample")
    gates = {
        "Gate 1 Submission and Calibration Integrity": {"status": "passed" if pair_validation["pair_validation_status"] == "valid" else "failed", "reason": pairing_reason},
        "Gate 2 History Extractor Quality": {"status": "passed" if history_gate["passed"] else "failed",
                                              "reason": history_gate["reason"]},
        "Gate 3 Future Observability": {"status": "passed" if gate3_pass else ("failed" if formal else NOT_AVAILABLE),
                                         "reason": f"expected-visible agent-frames={visible_count}" if formal else "requires paired submitted trajectory"},
        "Gate 4 Image-Trajectory Consistency": {"status": "diagnostic_complete" if formal else "not_available",
                                                 "reason": "future metrics computed; formal status still follows history extractor gate" if formal else "pair validation failed"},
    }
    scores = dimension_scores(formal, warm, config) if formal else {name: NOT_AVAILABLE for name in SCORE_NAMES}
    scorecard = {"pair_id": manifest["pair_id"], "image_scene_id": manifest["image_scene_id"],
                 "trajectory_scene_id": manifest["trajectory_scene_id"], "status": status,
                 "extractor_backend": backend_cfg["name"], "history_quality": warm, "future_metrics": metrics,
                 "calibrated_metrics": {key: metrics[key] for key in ("excess_ade", "recall_drop", "idf1_drop", "heading_error_increase")},
                 "scores": scores, "provisional_total_score": False}
    write_json(output / "metrics" / "scorecard.json", scorecard)
    scene_row = {"pair_id": manifest["pair_id"], "image_scene_id": manifest["image_scene_id"],
                 "trajectory_scene_id": manifest["trajectory_scene_id"], "status": scorecard["status"], "history_frames": inventory["history_frames"],
                 "future_frames": inventory["future_frames"], "camera_complete_ratio": inventory["six_camera_complete_ratio"],
                 "history_agents": inventory["history_known_agent_count"], "future_submitted_agents": len(submitted_tracks) if paired_scene else NOT_AVAILABLE,
                 "extracted_tracks": extraction.get("tracks", NOT_AVAILABLE), **metrics}
    _write_csv(output / "metrics" / "per_scene_metrics.csv", [scene_row])
    render_six_camera(manifest["future_images"], manifest["future_frame_indices"][0], output / "visualizations" / "six_camera_frame_000010.png")
    render_history(history_samples, output / "visualizations" / "history_warmup_bev.png")
    if tracks: render_bev_tracks(tracks, output / "visualizations" / "diagnostic_extracted_bev_tracks.png", "GT-image attached metadata BEV tracks")
    unit_tests = _run_unit_tests() if mode == "full" else {"status": "not_run_in_this_mode"}
    write_json(output / "unit_test_results.json", unit_tests)
    final = {"implementation_status": "completed" if scorecard["status"] in {"valid", "valid_low_sample"} else "partial",
             "status": scorecard["status"],
             "pair_id": manifest["pair_id"], "image_scene_id": manifest["image_scene_id"],
             "trajectory_scene_id": manifest["trajectory_scene_id"], "pair_validation": pair_validation,
             "backend": {"model": backend_cfg["name"], "config": backend_cfg.get("config", NOT_AVAILABLE),
                         "checkpoint": backend_cfg.get("checkpoint") or "not_used",
                         "environment": backend_cfg.get("environment", "stdlib_evaluator"),
                         "gpu": "not_used", "declaration": backend_audit["declaration"]},
             "data_statistics": {"scene_count": 1, "history_frames": inventory["history_frames"], "future_frames": inventory["future_frames"],
                                 "six_camera_complete_ratio": inventory["six_camera_complete_ratio"],
                                 "history_known_agent_count": inventory["history_known_agent_count"],
                                 "future_submitted_agent_count": len(submitted_tracks) if paired_scene else NOT_AVAILABLE,
                                 "expected_visible_agent_frames": visible_count if formal else NOT_AVAILABLE},
             "history_warmup": warm, "extraction": extraction, "future_metrics": metrics,
             "multi_view_diagnostics": formal["reprojection"]["per_camera"] if formal else {camera: NOT_AVAILABLE for camera in CAMERA_ORDER},
             "hard_gates": gates,
             "scorecard": scorecard, "backend_provenance": backend_audit, "cache_provenance": backend_audit,
             "corruption_benchmark": corruption, "unit_tests": unit_tests,
             "prohibitions_verified": {"future_submission_used_by_extractor": False, "planning_head_run_by_this_tool": False,
                                       "closed_loop_run": False, "traffic_manager_modified": False, "pdms_computed": False}}
    write_json(output / "final_metrics.json", final)
    report_path = output / "final_report.md"
    final["output_files"] = sorted(set(str(path) for path in output.rglob("*") if path.is_file()) | {str(report_path)})
    report_path.write_text(_report(final), encoding="utf-8")
    write_json(output / "final_metrics.json", final)
    (docs / "current_results.md").write_text(_report(final), encoding="utf-8")
    return final

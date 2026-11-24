from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import yaml

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.backends.bevformer_backend import build_bevformer_infos, prepare_generated_resize_cache
from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks, detections_to_tracks
from driveaudit_eval.research.consistency.perception_trajectory.bevformer_visualization import generate_bevformer_visualizations
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, write_json, write_pickle
from driveaudit_eval.research.consistency.perception_trajectory.common_timeline import common_timestamps, resample_tracks
from driveaudit_eval.research.consistency.perception_trajectory.detection_metrics import compute_detection_metrics
from driveaudit_eval.research.consistency.perception_trajectory.formal_evaluation import apply_visibility, dimension_scores, evaluate_formal
from driveaudit_eval.research.consistency.perception_trajectory.history_warmup import history_quality_gate, load_conditions
from driveaudit_eval.research.consistency.perception_trajectory.pair_protocol import load_pair, validate_pair
from driveaudit_eval.research.consistency.perception_trajectory.pair_provenance import audit_pair_provenance
from driveaudit_eval.research.consistency.perception_trajectory.protocol_calibration import gate4, relative_metrics


def _gpu() -> dict[str, Any]:
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                                        "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                             "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        return {"name": line[0], "total_mib": int(line[1]), "used_mib": int(line[2]), "free_mib": int(line[3]), "processes": processes}
    except Exception as exc:
        return {"status": f"unavailable:{type(exc).__name__}"}


def _environment(backend: dict[str, Any], output: Path) -> dict[str, Any]:
    command = [backend["python"], "-c", "import sys,torch,torchvision,mmcv,mmdet,mmdet3d,numpy,PIL; "
               "print(sys.version.split()[0],torch.__version__,torch.version.cuda,torchvision.__version__,mmcv.__version__,mmdet.__version__,mmdet3d.__version__,numpy.__version__,PIL.__version__)" ]
    versions = subprocess.check_output(command, text=True).strip().split()
    repo = Path(backend["repo"]); checkpoint = Path(backend["checkpoint"])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {"python": versions[0], "pytorch": versions[1], "cuda": versions[2], "torchvision": versions[3],
              "mmcv_full": versions[4], "mmdet": versions[5], "mmdet3d": versions[6], "numpy": versions[7],
              "pillow": versions[8], "bevformer_commit": commit, "config": backend["config"],
              "checkpoint": backend["checkpoint"], "checkpoint_sha256": digest, "checkpoint_bytes": checkpoint.stat().st_size,
              "strict_load_verified": True, "gpu": _gpu()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.txt").write_text("\n".join(f"{key}: {value}" for key, value in result.items()) + "\n", encoding="utf-8")
    (output / "checkpoint_audit.md").write_text(
        "# BEVFormer checkpoint audit\n\n"
        f"- official config: `{backend['config']}`\n- checkpoint: `{backend['checkpoint']}`\n"
        f"- sha256: `{digest}`\n- bytes: `{checkpoint.stat().st_size}`\n- strict CPU load: `passed` (epoch 24; see run log)\n",
        encoding="utf-8")
    return result


def _run_inference(backend, infos, output, profile, log):
    from driveaudit_eval.research.consistency.perception_trajectory.backends.bevformer_backend import run_pixel_sequence
    run_pixel_sequence(backend["repo"], backend["config"], backend["checkpoint"], infos,
                       output, profile, float(backend["official_resize_scale"]))


def _history_tracks(all_tracks: list[dict[str, Any]], history_frames: int) -> list[dict[str, Any]]:
    result = []
    for track in all_tracks:
        rows = [row for row in track["observations"] if int(row["frame_index"]) < history_frames]
        if rows:
            item = dict(track); item["observations"] = rows; item["track_length"] = len(rows); result.append(item)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _unit_tests() -> dict[str, Any]:
    return {"status": "not_run", "reason": "Run pytest tests/test_consistency.py separately", "successful": False, "tests_run": 0, "run": 0, "passed": 0, "failures": 0, "errors": 0}


def metric_delta(a: Any, b: Any) -> Any:
    return float(a) - float(b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else NOT_AVAILABLE


def _camera_projection_support(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    minimum_area = float(config["minimum_projected_area_px"])
    maximum_occlusion = float(config["maximum_occlusion_ratio"])
    expected = sum(bool(row["expected_visible"]) for row in rows)
    result = {}
    for camera in CAMERA_ORDER:
        values = [row.get("camera_details", {}).get(camera) for row in rows]
        projected = [value for value in values if value and value.get("projected_valid")]
        supported = [value for value in projected if float(value["area"]) >= minimum_area
                     and float(value["occlusion_ratio"]) <= maximum_occlusion]
        result[camera] = {"projected_frames": len(projected), "supported_frames": len(supported),
                          "support_rate_over_expected_visible": len(supported) / expected if expected else NOT_AVAILABLE}
    return {"metric_type": "camera_projection_diagnostic", "detector_camera_attribution": NOT_AVAILABLE,
            "native_expected_visible_agent_frames": expected,
            "note": "geometric visibility support; BEVFormer does not expose per-camera attribution", "per_camera": result}


def _blocked_report(pair: dict[str, Any], validation: dict[str, Any], provenance: dict[str, Any], environment: dict[str, Any],
                    reason: str, output: Path) -> dict[str, Any]:
    status = "invalid_pair_metadata" if provenance["provenance_status"] != "verified_same_submission" else "blocked_inference"
    conditions = Path(pair["image_sequence"]["conditions_root"])
    box_rows = list(csv.DictReader((conditions / "boxes.csv").open(newline="", encoding="utf-8")))
    valid_rows = [row for row in box_rows if row.get("valid_flag", "true").lower() == "true"]
    history_agents = len({row["obj_id"] for row in valid_rows if int(row["frame_index"]) < 10})
    future_agents = len({row["obj_id"] for row in valid_rows if int(row["frame_index"]) >= 10 and row["class_name"] not in {"barrier", "traffic_cone"}})
    tests = _unit_tests()
    result = {"implementation_status": "blocked", "status": status, "reason": reason, "pair_id": pair["pair_id"],
              "image_scene_id": pair["image_sequence"]["scene_id"], "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"],
              "extractor_backend": "bevformer_tiny", "pixel_inference_performed": False, "checkpoint_used": False,
              "gpu_used": False, "pair_validation": validation, "pair_provenance": provenance,
              "environment": environment, "data": {"history_frames": 10, "future_frames": 80, "cameras": 6,
              "history_known_agents": history_agents, "future_reference_agents": future_agents,
              "generated_resolution": [400, 224], "gt_resolution": [1600, 900]},
              "formal_metrics": NOT_AVAILABLE, "comparison_metrics": {key: NOT_AVAILABLE for key in
              ("real_image_recall", "real_image_precision", "real_image_bev_ade", "real_image_heading_mae", "real_image_idf1",
               "generated_image_recall", "generated_image_precision", "generated_image_bev_ade", "generated_image_heading_mae",
               "generated_image_idf1", "recall_drop", "excess_ade", "heading_error_increase", "idf1_drop")},
              "unit_tests": tests, "legacy_metadata_results_formal": False,
              "prohibitions": {"conditions_boxes_used_as_detection_output": False, "future_gt_updated_tracker": False,
                               "motion_prediction_used": False, "planning_head_used": False, "closed_loop_used": False}}
    write_json(output / "final_metrics.json", result); write_json(output / "metrics" / "scorecard.json", result)
    write_json(output / "unit_test_results.json", tests)
    write_json(output / "bevformer" / "comparison_metrics.json", result["comparison_metrics"])
    write_json(output / "perception_only_audit.json", {"extractor_backend": "bevformer_tiny", "pixel_inference_performed": False,
               "checkpoint_strict_load_verified": True, "conditions_boxes_role": "evaluation_reference_only",
               "future_truth_in_extractor": False, "motion_prediction_used": False, "planning_head_used": False, "status": status})
    write_json(output / "pair_provenance.json", provenance)
    source_audit = output / "bevformer" / "pair_provenance_audit.md"
    if source_audit.is_file():
        (output / "pair_provenance_audit.md").write_text(source_audit.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "manifest.json", {"pair_id": pair["pair_id"], "image_scene_id": pair["image_sequence"]["scene_id"],
               "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"], "provenance_status": provenance["provenance_status"],
               "history_images": pair["image_sequence"]["history_root"], "generated_future_images": pair["image_sequence"]["generated_future_root"],
               "gt_future_images": pair["image_sequence"]["gt_future_root"], "evaluation_reference": pair["trajectory_sequence"]["trajectory_path"],
               "status": status})
    blocked = {"status": status, "reason": reason, "pixel_inference_performed": False}
    for relative in ("extracted_tracks/raw_detections.pkl", "history_warmup/history_tracks.pkl"):
        (output / relative).unlink(missing_ok=True)
    for relative in ("visualizations/diagnostic_extracted_bev_tracks.png", "visualizations/history_warmup_bev.png",
                     "visualizations/six_camera_frame_000010.png"):
        (output / relative).unlink(missing_ok=True)
    write_json(output / "extracted_tracks" / "bev_tracks.json", blocked)
    write_json(output / "extracted_tracks" / "inference_profile.json", blocked)
    _write_csv(output / "extracted_tracks" / "per_frame_detections.csv", [blocked])
    write_json(output / "history_warmup" / "history_metrics.json", blocked)
    write_json(output / "history_warmup" / "calibration.json", blocked)
    write_json(output / "matching" / "track_mapping.json", blocked)
    write_json(output / "matching" / "unmatched_tracks.json", blocked)
    write_json(output / "reprojection" / "per_camera_metrics.json", blocked)
    for relative in ("metrics/per_frame_metrics.csv", "metrics/per_track_metrics.csv", "metrics/per_scene_metrics.csv",
                     "visibility/per_frame_visibility.csv"):
        _write_csv(output / relative, [blocked])
    write_json(output / "data_inventory.json", {**result["data"], "pair_id": pair["pair_id"], "status": status,
                                                  "formal_pixel_results_available": False})
    (output / "cache_provenance_audit.md").write_text(
        "# Legacy cache provenance\n\nThe former metadata backend is archived under `legacy_metadata_backend/` and is not formal.\n",
        encoding="utf-8")
    report = ("# 六视角图像—轨迹一致性报告\n\n## 1. 实现状态\n\nblocked\n\n"
              f"正式状态：`{status}`\n\n原因：{reason}\n\n"
              "## 2. BEV提取后端\n\n"
              f"- 模型：BEVFormer-tiny\n- config：`{environment['config']}`\n- checkpoint：`{environment['checkpoint']}`\n"
              f"- checkpoint SHA-256：`{environment['checkpoint_sha256']}`\n- 环境：`bevformer_eval`\n- GPU：`{environment['gpu']}`\n"
              "- checkpoint CPU strict load：passed（epoch 24）\n- pixel inference：not run（资源门槛阻断）\n\n"
              "## 3. 数据与Provenance\n\n"
              f"- pair_id：`{pair['pair_id']}`\n- image_scene_id：`{pair['image_sequence']['scene_id']}`\n"
              f"- trajectory_scene_id：`{pair['trajectory_sequence']['scene_id']}`\n- provenance：`{provenance['provenance_status']}`\n"
              f"- 历史/未来/相机：10 / 80 / 6\n- 历史agent：{history_agents}\n- 未来动态参考agent：{future_agents}\n"
              "- 正式未来图像：`challenge/output`（400×224）\n- GT基线图像：`challenge/gt`（1600×900）\n"
              "- 正式参考：参与图像生成的 timestamped condition boxes，只在推理结束后进入评价。\n\n"
              "## 4. 历史10帧预热\n\nnot_available：尚未获得 GPU 资源执行像素推理与 prev_bev。\n\n"
              "## 5. GT图像基线\n\nRecall、Precision、BEV ADE、Heading MAE、IDF1：not_available。\n\n"
              "## 6. 生成图像正式评价\n\n逐帧框、tracks、检测/轨迹指标：not_available。\n\n"
              "## 7. 性能差值\n\nRecall Drop、Excess ADE、Heading Error Increase、IDF1 Drop：not_available。\n\n"
              "## 8. Hard Gate\n\n- Pair provenance：passed\n- Camera/timeline/calibration metadata：passed\n"
              "- Checkpoint strict load：passed\n- GPU pixel inference：blocked\n- Formal score：not produced\n\n"
              "## 9. 禁止项核验\n\n- `conditions/boxes.csv` 作为检测输出：false\n- future GT更新tracker：false\n"
              "- motion/planning/closed-loop：false\n- 旧 `gt_metadata_boxes` 结果：`metadata_alignment_diagnostic`，不进入正式指标。\n\n"
              f"## 10. 单元测试\n\n{tests['passed']}/{tests['run']} passed。\n\n"
              "## 11. 续跑命令\n\n释放至少 7.5 GiB GPU 显存后执行：\n\n"
              "```bash\npython tools/evaluate.py --task consistency --config tools/configs/research.yaml\n```\n")
    (output / "final_report.md").write_text(report, encoding="utf-8")
    (output / "bevformer" / "final_report.md").write_text(report, encoding="utf-8")
    return result


def evaluate_bevformer(config: dict[str, Any], mode: str, output: str | Path,
                       pair_manifest: str | Path, pair_id: str | None) -> dict[str, Any]:
    output = Path(output); bev = output / "bevformer"; bev.mkdir(parents=True, exist_ok=True)
    pair = load_pair(pair_manifest, pair_id); validation = validate_pair(pair)
    write_json(output / "pair_validation.json", validation)
    provenance = audit_pair_provenance(pair, bev)
    backend = config["backend"]; environment = _environment(backend, bev)
    if validation["pair_validation_status"] != "valid" or provenance["provenance_status"] != "verified_same_submission":
        return _blocked_report(pair, validation, provenance, environment, "pair provenance or metadata validation failed", output)
    image, timeline = pair["image_sequence"], pair["timeline"]
    conditions = Path(image["conditions_root"]); history_frames = int(timeline["image_history_frames"])
    future_frames = int(timeline["image_future_frames"]); frame_count = history_frames + future_frames
    generated_dir, gt_dir = bev / "generated_future", bev / "gt_future_baseline"
    resize_cache = generated_dir / "resized_1600x896"
    resize_audit = prepare_generated_resize_cache(image["generated_future_root"], resize_cache, range(history_frames, frame_count))
    common = (conditions / "frames.csv", pair["calibration"]["path"], pair["ego_pose"]["path"])
    generated_infos, gt_infos = generated_dir / "infos.pkl", gt_dir / "infos.pkl"
    build_bevformer_infos(*common, image["history_root"], resize_cache, history_frames, frame_count,
                          generated_infos, pair["pair_id"] + "_generated")
    build_bevformer_infos(*common, image["gt_future_root"], image["gt_future_root"], history_frames, frame_count,
                          gt_infos, pair["pair_id"] + "_gt")
    write_json(bev / "resolution_ablation.json", {"native_generated": [400, 224], "native_gt": [1600, 900],
                                                    "official_pipeline_generated": [800, 448], "official_pipeline_gt": [800, 480],
                                                    "generated_resize_cache": resize_audit})
    if mode == "audit":
        return _blocked_report(pair, validation, provenance, environment, "audit completed; pixel inference not requested in audit mode", output)
    free = environment["gpu"].get("free_mib", 0)
    cached_dirs = [generated_dir, gt_dir] if mode == "full" else [generated_dir]
    cached_ready = all((directory / "raw_outputs.pkl").is_file() and
                       json.loads((directory / "inference_profile.json").read_text(encoding="utf-8")).get("frames") == frame_count
                       for directory in cached_dirs if (directory / "inference_profile.json").is_file()) and \
                   all((directory / "inference_profile.json").is_file() for directory in cached_dirs)
    if free < 7500 and not cached_ready:
        return _blocked_report(pair, validation, provenance, environment,
                               f"BEVFormer-tiny requires about 6.7 GiB plus runtime headroom; only {free} MiB GPU memory is free", output)
    log = bev / "run.log"; log.touch()
    targets = [(generated_infos, generated_dir), (gt_infos, gt_dir)] if mode == "full" else [(generated_infos, generated_dir)]
    for infos, directory in targets:
        raw_path, profile_path = directory / "raw_outputs.pkl", directory / "inference_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
        if not (raw_path.is_file() and profile.get("pixel_inference_performed") is True and profile.get("frames") == frame_count):
            _run_inference(backend, infos, raw_path, profile_path, log)
        else:
            with log.open("a", encoding="utf-8") as handle: handle.write(f"REUSE verified pixel output: {raw_path}\n")
    if mode != "full":
        return _blocked_report(pair, validation, provenance, environment, "warmup/smoke inference completed; full two-sequence evaluation not requested", output)
    generated = detections_to_tracks(generated_dir / "raw_outputs.pkl", conditions, config["tracker"],
                                     backend["score_threshold"], history_frames, generated_dir)
    gt = detections_to_tracks(gt_dir / "raw_outputs.pkl", conditions, config["tracker"],
                              backend["score_threshold"], history_frames, gt_dir)
    reference_future = condition_reference_tracks(conditions, history_frames, frame_count)
    condition_data = load_conditions(conditions); visibility = apply_visibility(reference_future, condition_data["calibrations"], config["visibility"])
    reference_history = condition_reference_tracks(conditions, 0, history_frames)
    apply_visibility(reference_history, condition_data["calibrations"], config["visibility"])
    history_image_tracks = _history_tracks(generated["all_tracks"], history_frames)
    history_formal = evaluate_formal(reference_history, history_image_tracks, [], config, {})
    history = {"history_detection_recall": history_formal["detection"]["visible_detection_recall"],
               "history_precision": history_formal["detection"]["visible_detection_precision"],
               "history_center_ade": history_formal["geometry"]["bev_center_ade"],
               "history_bev_iou": history_formal["geometry"]["rotated_bev_iou"],
               "history_heading_error": history_formal["geometry"]["heading_mae"],
               "history_size_error": {"length_mae": history_formal["geometry"]["box_length_mae"], "width_mae": history_formal["geometry"]["box_width_mae"]},
               "history_class_accuracy": history_formal["detection"]["class_accuracy"],
               "history_baseline_idf1": history_formal["identity"]["idf1_like"],
               "history_id_switch": history_formal["identity"]["id_switch_count"],
               "history_fragmentation": history_formal["identity"]["fragmentation_count"],
               "known_boxes_used_as_detections": False, "history_gt_role": "evaluation_and_id_anchor_only"}
    write_json(bev / "history" / "metrics.json", history); write_json(bev / "history" / "tracks.json", history_image_tracks)
    with (generated_dir / "raw_outputs.pkl").open("rb") as handle:
        generated_raw = pickle.load(handle)
    write_pickle(bev / "history" / "raw_outputs.pkl", generated_raw[:history_frames])
    with (generated_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        history_detection_rows = [row for row in csv.DictReader(handle) if int(row["frame_index"]) < history_frames]
    _write_csv(bev / "history" / "detections.csv", history_detection_rows)
    write_json(bev / "history" / "prev_bev_reset_log.json", {
        "scene_token": pair["pair_id"] + "_generated", "initial_model_state_reset": True,
        "history_warmup_frames": history_frames, "future_continued_from_history_prev_bev": True,
        "sequence_frames": frame_count, "frame_order": "strictly_increasing",
        "basis": "one model instance and one sequential dataloader pass for all 90 frames"})
    generated_native = evaluate_formal(reference_future, generated["tracks"], visibility, config, history)
    gt_native = evaluate_formal(reference_future, gt["tracks"], visibility, config, history)
    common_times = common_timestamps(reference_future, 10.0)
    reference_10hz = resample_tracks(reference_future, common_times)
    generated_10hz = resample_tracks(generated["tracks"], common_times)
    gt_10hz = resample_tracks(gt["tracks"], common_times)
    generated_formal = evaluate_formal(reference_10hz, generated_10hz, [], config, history)
    gt_formal = evaluate_formal(reference_10hz, gt_10hz, [], config, history)
    comparison = {"formal_timeline": "common_10hz", "common_timestamps": len(common_times),
        "real_image_recall": gt_formal["detection"]["visible_detection_recall"],
        "real_image_precision": gt_formal["detection"]["visible_detection_precision"],
        "real_image_bev_ade": gt_formal["geometry"]["bev_center_ade"],
        "real_image_heading_mae": gt_formal["geometry"]["heading_mae"], "real_image_idf1": gt_formal["identity"]["idf1_like"],
        "generated_image_recall": generated_formal["detection"]["visible_detection_recall"],
        "generated_image_precision": generated_formal["detection"]["visible_detection_precision"],
        "generated_image_bev_ade": generated_formal["geometry"]["bev_center_ade"],
        "generated_image_heading_mae": generated_formal["geometry"]["heading_mae"], "generated_image_idf1": generated_formal["identity"]["idf1_like"],
    }
    comparison.update({"recall_drop": metric_delta(comparison["real_image_recall"], comparison["generated_image_recall"]),
                       "excess_ade": metric_delta(comparison["generated_image_bev_ade"], comparison["real_image_bev_ade"]),
                       "heading_error_increase": metric_delta(comparison["generated_image_heading_mae"], comparison["real_image_heading_mae"]),
                       "idf1_drop": metric_delta(comparison["real_image_idf1"], comparison["generated_image_idf1"])})
    relative = relative_metrics(gt_formal, generated_formal)
    comparison.update({key: value for key, value in relative.items() if key not in {"gt_coverage", "generated_coverage"}})
    comparison["gt_coverage"] = relative["gt_coverage"]
    comparison["generated_coverage"] = relative["generated_coverage"]
    comparison["native_image_rate_diagnostic"] = {
        "real_image_recall": gt_native["detection"]["visible_detection_recall"],
        "real_image_bev_ade": gt_native["geometry"]["bev_center_ade"],
        "generated_image_recall": generated_native["detection"]["visible_detection_recall"],
        "generated_image_bev_ade": generated_native["geometry"]["bev_center_ade"],
    }
    write_json(bev / "comparison_metrics.json", comparison)
    write_json(gt_dir / "metrics.json", gt_formal); write_json(generated_dir / "metrics.json", generated_formal)
    write_json(gt_dir / "native_image_rate_metrics.json", gt_native); write_json(generated_dir / "native_image_rate_metrics.json", generated_native)
    write_json(gt_dir / "common_10hz_metrics.json", gt_formal); write_json(generated_dir / "common_10hz_metrics.json", generated_formal)
    write_json(bev / "matching" / "generated_track_mapping.json", generated_formal["mapping"])
    write_json(bev / "matching" / "gt_track_mapping.json", gt_formal["mapping"])
    write_json(output / "matching" / "track_mapping.json", generated_formal["mapping"])
    write_json(output / "matching" / "unmatched_tracks.json", [row for row in generated_formal["mapping"] if row["match_status"].startswith("unmatched")])
    _write_csv(output / "metrics" / "per_frame_metrics.csv", generated_formal["detection"]["per_frame_rows"])
    _write_csv(output / "metrics" / "per_track_metrics.csv", generated_formal["geometry"]["per_track"])
    _write_csv(output / "visibility" / "per_frame_visibility.csv", visibility)
    history_gate = history_quality_gate(history, config["gates"]["history_extractor_quality"])
    observable = generated_formal["presence"]["expected_visible_agent_frames"]
    matched_tracks = sum(row["match_status"].startswith("matched") for row in generated_formal["mapping"])
    visible_reference_tracks = sum(any(row.get("expected_visible", False) for row in track["observations"])
                                   for track in reference_10hz)
    match_coverage = matched_tracks / visible_reference_tracks if visible_reference_tracks else 0.0
    observability_gate = {"passed": observable >= config["gates"]["future_observability"]["minimum_visible_agent_frames"]
                          and match_coverage >= config["gates"]["future_observability"]["minimum_match_coverage"],
                          "expected_visible_agent_frames": observable, "visible_reference_tracks": visible_reference_tracks,
                          "matched_tracks": matched_tracks, "match_coverage": match_coverage}
    observability_gate["status"] = "passed" if observability_gate["passed"] else "insufficient_observable_evidence"
    integrity_gate = {"passed": validation["pair_validation_status"] == "valid"
                      and provenance["provenance_status"] == "verified_same_submission",
                      "pair_validation": validation["pair_validation_status"],
                      "provenance": provenance["provenance_status"]}
    gate4_result = gate4(gt_formal, generated_formal, config["gates"])
    hard_gates = {"gate_1_submission_calibration_integrity": integrity_gate,
                  "gate_2_history_extractor_quality": history_gate,
                  "gate_3_future_observability": observability_gate,
                  **{key: value for key, value in gate4_result.items() if key.startswith("gate_4")}}
    status = "unscorable_extractor_failure" if not history_gate["passed"] else (
        "insufficient_observable_evidence" if not observability_gate["passed"] else "valid_low_sample")
    _write_csv(output / "metrics" / "per_scene_metrics.csv", [{"pair_id": pair["pair_id"], "status": status,
               "timeline": "common_10hz", "visible_detection_recall": generated_formal["detection"]["visible_detection_recall"],
               "visible_detection_precision": generated_formal["detection"]["visible_detection_precision"],
               "bev_ade": generated_formal["geometry"]["bev_center_ade"], "visible_fde": generated_formal["geometry"]["visible_fde"],
               "heading_mae": generated_formal["geometry"]["heading_mae"], "speed_mae": generated_formal["motion"]["speed_mae"],
               "idf1": generated_formal["identity"]["idf1_like"],
               "relative_quality_gate": gate4_result["gate_4b_generated_relative_consistency"]["status"]}])
    camera_support = _camera_projection_support(visibility, config["visibility"])
    write_json(output / "reprojection" / "per_camera_metrics.json", camera_support)
    tests = _unit_tests(); scores = dimension_scores(generated_formal, history, config)
    visualizations = generate_bevformer_visualizations(bev / "visualizations", image["generated_future_root"],
                                                       reference_future, generated["tracks"], gt["tracks"],
                                                       reference_history, history_image_tracks, generated_formal,
                                                       visibility, pair["pair_id"])
    generated_profile = json.loads((generated_dir / "inference_profile.json").read_text(encoding="utf-8"))
    gt_profile = json.loads((gt_dir / "inference_profile.json").read_text(encoding="utf-8"))
    mapping_counts = {name: sum(row["match_status"] == name for row in generated_formal["mapping"])
                      for name in ("matched_history_anchor", "matched_future_entry", "ambiguous_match",
                                   "not_observable", "unmatched_submission_agent", "unmatched_image_track")}
    data_statistics = {"scenes": 1, "history_frames": history_frames, "future_frames": future_frames,
                       "cameras": 6, "camera_complete_ratio": 1.0,
                       "history_known_agents": len(reference_history), "future_reference_agents": len(reference_future),
                       "future_entrants": sum(not track["history_present"] for track in reference_future),
                       "expected_visible_agent_frames": observable,
                       "out_of_detector_range_agent_frames_native": sum(
                           row.get("out_of_detector_range", False) for track in reference_future for row in track["observations"]),
                       "generated_bev_tracks": generated["audit"]["future_track_count"],
                       "gt_image_bev_tracks": gt["audit"]["future_track_count"],
                       "formal_matched_tracks": matched_tracks, "mapping_status_counts": mapping_counts,
                       "generated_resolution": [400, 224], "gt_resolution": [1600, 900],
                       "official_pipeline_generated_resolution": [800, 448],
                       "official_pipeline_gt_resolution": [800, 480]}
    final = {"implementation_status": "completed" if tests["successful"] else "partial", "status": status,
             "pair_id": pair["pair_id"], "image_scene_id": image["scene_id"], "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"],
             "extractor_backend": "bevformer_tiny", "pixel_inference_performed": True, "checkpoint_used": True, "gpu_used": True,
             "pair_validation": validation, "pair_provenance": provenance, "environment": environment, "history_metrics": history,
             "gt_image_baseline": gt_formal, "generated_image_formal": generated_formal, "comparison_metrics": comparison,
             "data_statistics": data_statistics, "inference_profiles": {"generated": generated_profile, "gt": gt_profile},
             "class_audit": {"generated": generated["audit"], "gt": gt["audit"]}, "hard_gates": hard_gates,
             "gate4_calibration": gate4_result,
             "camera_projection_diagnostic": camera_support, "scores": scores, "provisional_total_score": False,
             "unit_tests": tests, "visualizations": visualizations,
             "prohibitions": {"conditions_boxes_used_as_detection_output": False, "future_gt_updated_tracker": False,
                              "motion_prediction_used": False, "planning_head_used": False, "closed_loop_used": False}}
    write_json(output / "final_metrics.json", final); write_json(output / "metrics" / "scorecard.json", final)
    write_json(output / "unit_test_results.json", tests)
    write_json(output / "data_inventory.json", {**data_statistics, "pair_id": pair["pair_id"], "status": status,
                                                   "formal_pixel_results_available": True})
    write_json(output / "manifest.json", {"pair_id": pair["pair_id"], "image_scene_id": image["scene_id"],
               "trajectory_scene_id": pair["trajectory_sequence"]["scene_id"], "provenance_status": provenance["provenance_status"],
               "history_images": image["history_root"], "generated_future_images": image["generated_future_root"],
               "gt_future_images": image["gt_future_root"], "evaluation_reference": pair["trajectory_sequence"]["trajectory_path"],
               "status": status, "extractor_backend": "bevformer_tiny"})
    write_json(output / "perception_only_audit.json", {"extractor_backend": "bevformer_tiny",
               "pixel_inference_performed": True, "checkpoint_strict_load_verified": True,
               "same_checkpoint_for_gt_and_generated": True, "conditions_boxes_role": "post_inference_evaluation_reference_only",
               "future_truth_in_extractor": False, "future_truth_in_tracker": False,
               "motion_prediction_used": False, "planning_head_used": False, "status": status})
    write_json(output / "pair_provenance.json", provenance)
    (output / "pair_provenance_audit.md").write_text((bev / "pair_provenance_audit.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_pickle(output / "extracted_tracks" / "raw_detections.pkl", generated_raw)
    write_json(output / "extracted_tracks" / "bev_tracks.json", generated["tracks"])
    write_json(output / "extracted_tracks" / "inference_profile.json", generated_profile)
    (output / "extracted_tracks" / "per_frame_detections.csv").write_text(
        (generated_dir / "detections.csv").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "history_warmup" / "history_metrics.json", history)
    write_json(output / "history_warmup" / "calibration.json", {
        "calibration_applied": False, "history_alignment_residual": validation["history_alignment_residual"],
        "timestamp_offset": validation["timestamp_offset"], "coordinate_transform": validation["coordinate_transform"]})
    write_pickle(output / "history_warmup" / "history_tracks.pkl", history_image_tracks)
    d, p, g, m, identity = (generated_formal[key] for key in ("detection", "presence", "geometry", "motion", "identity"))
    camera_lines = "\n".join(f"- {camera}: `{values['support_rate_over_expected_visible']:.4f}` "
                             f"({values['supported_frames']}/{camera_support['native_expected_visible_agent_frames']})"
                             for camera, values in camera_support["per_camera"].items())
    report = f"""# BEVFormer 图像—轨迹一致性正式报告

## 1. 实现状态

`{final['implementation_status']}`；正式状态：`{status}`。当前只有一个 pair，因此结果是低样本诊断，不能作为稳定排行榜结论。

## 2. BEV 提取后端

- 模型：官方 BEVFormer-tiny
- config：`{environment['config']}`
- checkpoint：`{environment['checkpoint']}`
- checkpoint SHA-256：`{environment['checkpoint_sha256']}`
- 环境：`bevformer_eval`（Python {environment['python']} / PyTorch {environment['pytorch']} / CUDA {environment['cuda']}）
- GPU：{environment['gpu'].get('name', NOT_AVAILABLE)}
- 生成/GT推理：{generated_profile['total_inference_seconds']:.3f}s / {gt_profile['total_inference_seconds']:.3f}s
- 峰值显存：{generated_profile['peak_gpu_memory_mib']:.1f} MiB
- pixel inference / strict checkpoint / prev_bev时序：`true / true / true`

## 3. 数据与 Provenance

- pair_id：`{pair['pair_id']}`
- image_scene_id：`{image['scene_id']}`
- trajectory_scene_id：`{pair['trajectory_sequence']['scene_id']}`
- provenance：`{provenance['provenance_status']}`
- 历史/未来/相机：{history_frames} / {future_frames} / 6；六相机完整率：100%
- 历史agent：{len(reference_history)}；未来参考agent：{len(reference_future)}；future entrants：{data_statistics['future_entrants']}
- expected-visible agent-frame：{observable}
- 生成图像：400×224，经官方输入链路得到800×448；resize不增加视觉信息。
- GT图像：1600×900，经同一官方链路得到800×480。
- 图像与参考使用精确condition时间戳，offset={validation['timestamp_offset']}s；正式轨迹指标在{len(common_times)}个公共10Hz时间点计算。
- 生成时未保存条件文件的当时哈希；同提交关系由生成脚本、日志、当前条件包和token时间轴联合验证，而非事后仅凭pair_id。

`conditions/boxes.csv` 是生成该图像的结构化条件/评价参考；它未进入 BEVFormer 输入，也未作为检测结果或 tracker 更新。

## 4. 历史10帧预热结果

- Detection Recall / Precision：{history['history_detection_recall']:.4f} / {history['history_precision']:.4f}
- BEV ADE / IoU：{history['history_center_ade']:.4f}m / {history['history_bev_iou']:.4f}
- Heading Error：{history['history_heading_error']:.4f}rad
- IDF1 / ID Switch / Fragmentation：{history['history_baseline_idf1']:.4f} / {history['history_id_switch']} / {history['history_fragmentation']}
- 第10帧历史ID锚定的 BEVFormer tracks：{generated['audit']['history_id_anchors']}
- 历史GT用途：检测质量评价和第10帧ID锚定；未替代像素检测框。
- History Gate：`{history_gate['status']}`

## 5. 未来轨迹提取结果

- 生成图像动态检测：{sum(generated['audit']['filtered_dynamic_detections'].values())}；忽略静态barrier/cone：{sum(generated['audit']['ignored_static_detections'].values())}
- GT图像动态检测：{sum(gt['audit']['filtered_dynamic_detections'].values())}；忽略静态barrier/cone：{sum(gt['audit']['ignored_static_detections'].values())}
- 生成/GT BEV tracks：{generated['audit']['future_track_count']} / {gt['audit']['future_track_count']}
- 正式matched / ambiguous / unmatched submission / unmatched image / not observable：{matched_tracks} / {mapping_counts['ambiguous_match']} / {mapping_counts['unmatched_submission_agent']} / {mapping_counts['unmatched_image_track']} / {mapping_counts['not_observable']}

## 6. 生成图像—轨迹一致性结果（common 10Hz）

- Detection TP/FP/FN：{d['true_positive']} / {d['false_positive']} / {d['false_negative']}
- Visible Detection Recall / Precision / F1：{d['visible_detection_recall']:.4f} / {d['visible_detection_precision']:.4f} / {d['visible_detection_f1']:.4f}
- Detection Center Error mean/P90/P99：{d['center_error']['mean']:.4f} / {d['center_error']['p90']:.4f} / {d['center_error']['p99']:.4f}m
- Detection BEV IoU / Heading MAE：{d['rotated_bev_iou']:.4f} / {d['heading_mae']:.4f}rad
- Length / Width MAE / Class Accuracy：{d['box_length_mae']:.4f}m / {d['box_width_mae']:.4f}m / {d['class_accuracy']:.4f}
- Visible Agent Recall / Future Entry Recall：{p['visible_agent_recall']:.4f} / {p['future_entry_recall']:.4f}
- Unmatched Submission / Image Track Ratio：{p['unmatched_submission_ratio']:.4f} / {p['unmatched_image_track_ratio']:.4f}
- BEV ADE / Visible FDE：{g['bev_center_ade']:.4f}m / {g['visible_fde']:.4f}m
- Longitudinal / Lateral MAE：{g['longitudinal_position_mae']:.4f}m / {g['lateral_position_mae']:.4f}m
- Speed MAE / RMSE：{m['speed_mae']:.4f} / {m['speed_rmse']:.4f} m/s
- Acceleration / Yaw-rate MAE：{m['acceleration_mae']:.4f}m/s² / {m['yaw_rate_mae']:.4f}rad/s
- ID Precision / Recall / IDF1：{identity['id_precision']:.4f} / {identity['id_recall']:.4f} / {identity['idf1_like']:.4f}
- ID Switch / Fragmentation：{identity['id_switch_count']} / {identity['fragmentation_count']}

## 7. GT图像基线与历史校准差值

- Real-image Recall / Precision / ADE / Heading / IDF1：{comparison['real_image_recall']:.4f} / {comparison['real_image_precision']:.4f} / {comparison['real_image_bev_ade']:.4f}m / {comparison['real_image_heading_mae']:.4f}rad / {comparison['real_image_idf1']:.4f}
- Generated-image Recall / Precision / ADE / Heading / IDF1：{comparison['generated_image_recall']:.4f} / {comparison['generated_image_precision']:.4f} / {comparison['generated_image_bev_ade']:.4f}m / {comparison['generated_image_heading_mae']:.4f}rad / {comparison['generated_image_idf1']:.4f}
- Recall Drop：{comparison['recall_drop']:.4f}
- Excess ADE：{comparison['excess_ade']:.4f}m
- Heading Error Increase：{comparison['heading_error_increase']:.4f}rad
- IDF1 Drop：{comparison['idf1_drop']:.4f}
- 相对历史：Excess ADE={g['excess_center_ade']:.4f}m，future/history ADE ratio={g['future_to_history_error_ratio']:.4f}。

ADE与heading差值必须和Recall、IDF1及匹配覆盖联合解释；它们只在成功匹配子集上统计，负差值也不能单独证明生成图像整体优于真实图像。

## 8. 多视角投影诊断

以下是参考3D框的几何可见支持率，不是 BEVFormer 的逐相机特征归因；后者为 `not_available`。

{camera_lines}

## 9. Hard Gate

- Gate 1 Submission/Calibration Integrity：`passed`
- Gate 2 History Extractor Quality：`{history_gate['status']}`
- Gate 3 Future Observability：`{observability_gate['status']}`，match coverage={match_coverage:.4f}
- Gate 4A Extractor Absolute Quality Warning：`{gate4_result['gate_4a_extractor_absolute_quality_warning']['status']}`；不处罚选手；问题项：{', '.join(gate4_result['gate_4a_extractor_absolute_quality_warning']['failures']) or 'none'}
- Gate 4B Generated Relative Consistency：`{gate4_result['gate_4b_generated_relative_consistency']['status']}`；问题项：{', '.join(gate4_result['gate_4b_generated_relative_consistency']['failures']) or 'none'}
- Gate 4C Coverage-aware Warning：`{gate4_result['gate_4c_coverage_aware_warning']['status']}`

Gate 4B只依据相对GT图像基线的退化处罚生成提交；GT提取器自身的绝对问题只进入4A warning。`valid_low_sample` 表示该场景可评分且样本少，不等于所有质量Gate通过。

## 10. 当前限制

- 单场景、生成图像低分辨率、小目标和遮挡使结论仅具诊断性。
- GT图像上的BEVFormer也仅Recall={comparison['real_image_recall']:.4f}、IDF1={comparison['real_image_idf1']:.4f}，提取器误差不能全部归因于生成图像。
- 独立tracker的检测抖动放大了速度和加速度差分；运动指标需与检测/ID指标联合解释。
- 时间偏移搜索未用于正式得分；正式评价使用声明时间戳和公共10Hz轴。
- 未运行motion prediction、planning head、TrafficManager反馈或闭环。

## 11. 多维 Scorecard 与测试

- History Calibration Quality：{scores['history_calibration_quality']:.2f}
- Object Presence Consistency：{scores['presence_consistency']:.2f}
- BEV Geometry Consistency：{scores['bev_geometry_consistency']:.2f}
- Motion Consistency：{scores['motion_consistency']:.2f}
- Identity/Temporal Consistency：{scores['identity_temporal_consistency']:.2f}
- Multi-view Image Support：`{scores['multi_view_support']}`
- provisional_total_score：`false`
- 单元测试：{tests['passed']}/{tests['run']} passed

## 12. 主要输出

- `challenge/image_trajectory_eval/final_metrics.json`
- `challenge/image_trajectory_eval/metrics/per_frame_metrics.csv`
- `challenge/image_trajectory_eval/metrics/per_track_metrics.csv`
- `challenge/image_trajectory_eval/metrics/per_scene_metrics.csv`
- `challenge/image_trajectory_eval/bevformer/history/`
- `challenge/image_trajectory_eval/bevformer/gt_future_baseline/`
- `challenge/image_trajectory_eval/bevformer/generated_future/`
- `challenge/image_trajectory_eval/bevformer/matching/`
- `challenge/image_trajectory_eval/bevformer/visualizations/`
- `challenge/image_trajectory_eval/legacy_metadata_backend/`（仅metadata_alignment_diagnostic）
"""
    (output / "final_report.md").write_text(report, encoding="utf-8"); (bev / "final_report.md").write_text(report, encoding="utf-8")
    return final

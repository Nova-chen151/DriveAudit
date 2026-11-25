from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.config import load_config as read_config

from driveaudit_eval.trajectory.io import load_tracks

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.backends.bevformer_backend import build_bevformer_infos, prepare_generated_resize_cache
from driveaudit_eval.research.consistency.perception_trajectory.bevformer_evaluator import _environment, _history_tracks
from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks, detections_to_tracks
from driveaudit_eval.consistency.shared.common import write_json
from driveaudit_eval.research.consistency.perception_trajectory.common_timeline import common_timestamps, resample_tracks
from driveaudit_eval.research.consistency.perception_trajectory.distribution_reference_protocol import evaluate_distribution_reference, trajectory_schema_to_tracks
from driveaudit_eval.research.consistency.perception_trajectory.formal_evaluation import apply_visibility, evaluate_formal
from driveaudit_eval.research.consistency.perception_trajectory.history_warmup import history_quality_gate, load_conditions
from driveaudit_eval.research.consistency.perception_trajectory.metrics.actbench_metrics import compute_actbench_metrics
from driveaudit_eval.research.consistency.perception_trajectory.metrics.bevformer_metrics import compute_bevformer_submission_metrics, temporal_gain
from driveaudit_eval.research.consistency.perception_trajectory.metrics.drivinggen_metrics import compute_drivinggen_metrics
from driveaudit_eval.research.consistency.perception_trajectory.metrics.task_fidelity_metrics import compute_task_fidelity_metrics
from driveaudit_eval.research.consistency.perception_trajectory.metrics.worldlens_metrics import compute_worldlens_metrics
from driveaudit_eval.research.consistency.perception_trajectory.paper_metric_registry import build_registry
from driveaudit_eval.research.consistency.perception_trajectory.protocol_calibration import coverage_diagnostics


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_fingerprint(root: str | Path) -> str:
    digest = hashlib.sha256(); root = Path(root)
    for camera in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
        for frame in range(10, 90):
            path = root / camera / f"{frame:06d}.jpg"
            digest.update(f"{camera}/{frame:06d}.jpg\0".encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _pair(path: str | Path, pair_id: str) -> dict[str, Any]:
    data = read_config(path)
    matches = [row for row in data["pairs"] if row["pair_id"] == pair_id]
    if len(matches) != 1: raise ValueError(f"pair_id must resolve exactly once: {pair_id}")
    return matches[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _validate_submission_pair(pair: dict[str, Any]) -> dict[str, Any]:
    image, timeline = pair["image_sequence"], pair["timeline"]
    paths = [image["history_root"], image["generated_future_root"], image["conditions_root"],
             pair["calibration"]["path"], pair["ego_pose"]["path"], pair["trajectory_sequence"]["trajectory_path"]]
    missing = [path for path in paths if not Path(path).exists()]
    camera_count = sum((Path(image["generated_future_root"]) / camera).is_dir() for camera in
                       ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"))
    return {"pair_id": pair["pair_id"], "valid": not missing and camera_count == 6,
            "missing": missing, "camera_count": camera_count,
            "history_frames": int(timeline["image_history_frames"]), "future_frames": int(timeline["image_future_frames"]),
            "future_gt_required": False, "future_gt_paths_opened": []}


def _run_inference(config, infos, raw, profile, log, temporal):
    from driveaudit_eval.research.consistency.perception_trajectory.backends.bevformer_backend import run_pixel_sequence
    backend = config["backend"]
    run_pixel_sequence(backend["repo"], backend["config"], backend["checkpoint"], infos,
                       raw, profile, float(backend["official_resize_scale"]))
    data = _read_json(profile); data.update({"model_variant": backend["model_variant"],
                                             "checkpoint_sha256": hashlib.sha256(Path(backend["checkpoint"]).read_bytes()).hexdigest(),
                                             "temporal_prev_bev": temporal, "future_truth_fields_present": False})
    write_json(profile, data)


def _ensure_raw(config: dict[str, Any], pair: dict[str, Any], directory: Path, temporal: bool) -> tuple[Path, dict[str, Any]]:
    backend, image = config["backend"], pair["image_sequence"]
    directory.mkdir(parents=True, exist_ok=True); raw, profile = directory / "raw_outputs.pkl", directory / "inference_profile.json"
    expected_hash = hashlib.sha256(Path(backend["checkpoint"]).read_bytes()).hexdigest()
    future_root = str(Path(image["generated_future_root"]).resolve()); future_fingerprint = _image_fingerprint(future_root)
    if raw.is_file() and profile.is_file():
        audit = _read_json(profile)
        if (audit.get("frames") == 90 and audit.get("checkpoint_sha256") == expected_hash and
                audit.get("temporal_prev_bev") is temporal and audit.get("future_image_root") == future_root and
                audit.get("future_image_fingerprint") == future_fingerprint):
            return raw, audit
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    required_mib = int(backend.get("minimum_free_gpu_mib", 13000))
    if backend.get("require_exclusive_compute_gpu", False):
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip().splitlines()
        external = [row for row in processes if row and "chrome" not in row.lower()]
        if external: raise RuntimeError(f"blocked_external_gpu_process: {external}")
    if free_mib < required_mib:
        raise RuntimeError(f"blocked_gpu_memory: {backend['model_variant']} requires {required_mib} MiB free by policy; found {free_mib} MiB")
    resize_cache = directory / "resized_future_1600x896"
    prepare_generated_resize_cache(image["generated_future_root"], resize_cache, range(10, 90))
    conditions = Path(image["conditions_root"]); infos = directory / "infos.pkl"
    build_bevformer_infos(conditions / "frames.csv", pair["calibration"]["path"], pair["ego_pose"]["path"],
                          image["history_root"], resize_cache, 10, 90, infos,
                          pair["pair_id"] + ("_temporal" if temporal else "_frame_independent"),
                          frame_independent=not temporal)
    _run_inference(config, infos, raw, profile, directory / "run.log", temporal)
    audit = _read_json(profile); audit.update({"future_image_root": future_root,
                                               "future_image_fingerprint": future_fingerprint})
    write_json(profile, audit)
    return raw, audit


def _evaluate_raw(config: dict[str, Any], pair: dict[str, Any], raw: Path, directory: Path) -> dict[str, Any]:
    conditions = Path(pair["image_sequence"]["conditions_root"])
    extracted = detections_to_tracks(raw, conditions, config["tracker"], config["backend"]["score_threshold"], 10, directory)
    detection_tracks = []
    for track in extracted["all_tracks"]:
        observations = [row for row in track["observations"] if int(row["frame_index"]) >= 10 and row.get("source") == "image_detection"]
        if observations: detection_tracks.append({**track, "observations": observations, "track_length": len(observations)})
    condition_data = load_conditions(conditions)
    future_reference = condition_reference_tracks(conditions, 10, 90)
    visibility = apply_visibility(future_reference, condition_data["calibrations"], config["visibility"])
    history_reference = condition_reference_tracks(conditions, 0, 10)
    apply_visibility(history_reference, condition_data["calibrations"], config["visibility"])
    history_tracks = _history_tracks(extracted["all_tracks"], 10)
    history_formal = evaluate_formal(history_reference, history_tracks, [], config, {})
    history = {"history_detection_recall": history_formal["detection"]["visible_detection_recall"],
               "history_detection_precision": history_formal["detection"]["visible_detection_precision"],
               "history_precision": history_formal["detection"]["visible_detection_precision"],
               "history_center_ade": history_formal["geometry"]["bev_center_ade"],
               "history_bev_iou": history_formal["geometry"]["rotated_bev_iou"],
               "history_heading_error": history_formal["geometry"]["heading_mae"],
               "history_scale_error": NOT_AVAILABLE, "history_velocity_error": history_formal["motion"]["speed_mae"],
               "history_baseline_idf1": history_formal["identity"]["idf1_like"],
               "history_fragmentation": history_formal["identity"]["fragmentation_count"]}
    history_paper = compute_bevformer_submission_metrics(history_reference, history_tracks)
    history["history_center_distance_ap"] = history_paper["submission_ref_center_distance_mAP"]
    history["history_scale_error"] = history_paper["submission_ref_mASE"]
    history.update({"history_mATE": history_paper["submission_ref_mATE"],
                    "history_mASE": history_paper["submission_ref_mASE"],
                    "history_mAOE": history_paper["submission_ref_mAOE"],
                    "history_mAVE": history_paper["submission_ref_mAVE"]})
    native = evaluate_formal(future_reference, extracted["tracks"], visibility, config, history)
    times = common_timestamps(future_reference, 10.0)
    reference_10hz = resample_tracks(future_reference, times, config["speed_estimation"]["maximum_gap_seconds"])
    tracks_10hz = resample_tracks(extracted["tracks"], times, config["speed_estimation"]["maximum_gap_seconds"])
    common = evaluate_formal(reference_10hz, tracks_10hz, visibility, config, history)
    return {"extracted": extracted, "detection_tracks": detection_tracks,
            "future_reference": future_reference, "reference_10hz": reference_10hz,
            "tracks_10hz": tracks_10hz, "visibility": visibility, "history_reference": history_reference,
            "history_tracks": history_tracks, "history": history, "native": native, "common": common,
            "common_timestamps": times.tolist()}


def _gate4(config: dict[str, Any], act: dict[str, Any], drivinggen: dict[str, Any], bev: dict[str, Any],
           task: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    limits = config["submission_only_gates"]; failures = []
    checks = {
        "motion_accuracy": (act["act_agent_motion_accuracy_adapted"], limits["minimum_motion_accuracy"], "min"),
        "agent_missing_ratio": (drivinggen["agent_missing_ratio"], limits["maximum_agent_missing_ratio"], "max"),
        "matched_coverage": (coverage["matched_coverage"], limits["minimum_matched_coverage"], "min"),
        "trajectory_ade": (drivinggen["drivinggen_alignment_ade"], limits["maximum_ade_m"], "max"),
        "submission_ref_mATE": (bev["submission_ref_mATE"], limits["maximum_mate_m"], "max"),
        "submission_ref_mAOE": (bev["submission_ref_mAOE"], limits["maximum_maoe_rad"], "max"),
        "submission_ref_mAVE": (bev["submission_ref_mAVE"], limits["maximum_mave_mps"], "max"),
        "submission_ref_idf1": (task["submission_ref_idf1"], limits["minimum_idf1"], "min")}
    for name, (value, threshold, direction) in checks.items():
        if isinstance(value, (int, float)) and ((direction == "min" and value < threshold) or (direction == "max" and value > threshold)):
            failures.append(name)
    return {"status": "passed" if not failures else "failed_provisional", "passed": not failures,
            "provisional_thresholds": True, "failures": failures, "checks": checks}


def _scores(history: dict[str, Any], act: dict[str, Any], drivinggen: dict[str, Any], bev: dict[str, Any],
            task: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    def mean(*values: Any) -> Any:
        usable = [min(1.0, max(0.0, float(value))) for value in values if isinstance(value, (int, float))]
        return 100.0 * sum(usable) / len(usable) if usable else None
    visibility = [row["recall"] for row in bev["visibility_stratified"].values() if isinstance(row["recall"], (int, float))]
    return {"history_extractor_reliability": mean(history["history_detection_recall"], history["history_detection_precision"], history["history_baseline_idf1"]),
            "submission_detection_fidelity": mean(bev["submission_ref_center_distance_mAP"], task["submission_ref_precision"], task["submission_ref_recall"]),
            "instruction_execution_alignment": mean(act["act_agent_motion_accuracy_adapted"], 1 / (1 + act["act_agent_ade_adapted"]) if isinstance(act["act_agent_ade_adapted"], (int, float)) else None),
            "presence_consistency": mean(1 - drivinggen["agent_missing_ratio"] if isinstance(drivinggen["agent_missing_ratio"], (int, float)) else None, coverage["matched_coverage"]),
            "bev_geometry_consistency": mean(1 / (1 + bev["submission_ref_mATE"]) if isinstance(bev["submission_ref_mATE"], (int, float)) else None, bev["submission_ref_dynamic_occupancy_iou"]),
            "motion_trajectory_consistency": mean(drivinggen["trajectory_consistency"], 1 / (1 + bev["submission_ref_mAVE"]) if isinstance(bev["submission_ref_mAVE"], (int, float)) else None),
            "identity_temporal_consistency": mean(task["submission_ref_idf1"], task["submission_ref_hota_adapted"], task["track_completeness"]),
            "visibility_temporal_robustness": mean(*visibility),
            "optional_distribution_fidelity": None, "optional_paired_gt_calibration": None,
            "provisional_normalization": True}


def run_submission_only(config: dict[str, Any], output: Path, pair_manifest: str, pair_id: str,
                        run_inference: bool = True) -> dict[str, Any]:
    pair = _pair(pair_manifest, pair_id); validation = _validate_submission_pair(pair)
    if not validation["valid"]: raise ValueError(f"invalid Protocol S pair: {validation}")
    s = output / "protocol_s_submission_only"; temporal_dir = s / "extracted_tracks"
    provenance_path = output / "pair_provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {"provenance_status": "not_available"}
    environment = _environment(config["backend"], s / "bevformer")
    access = {"protocol": "submission_only", "future_gt_allowed": False,
              "future_gt_root_present_during_run": Path(pair["image_sequence"].get("gt_future_root", "__absent__")).exists(),
              "opened_future_gt_paths": [], "extractor_inputs": [pair["image_sequence"]["history_root"],
              pair["image_sequence"]["generated_future_root"], pair["calibration"]["path"], pair["ego_pose"]["path"]],
              "instruction_reference_opened_after_tracking": True}
    write_json(s / "future_gt_access_audit.json", access)
    if not access["future_gt_root_present_during_run"]:
        write_json(s / "future_gt_physical_isolation_audit.json", access)
    if not run_inference:
        return {"protocol": "submission_only", "status": "audit_ready", "validation": validation, "environment": environment}
    raw, profile = _ensure_raw(config, pair, temporal_dir, True)
    evaluated = _evaluate_raw(config, pair, raw, temporal_dir)
    formal, native = evaluated["common"], evaluated["native"]
    coverage = coverage_diagnostics(formal)
    bev = compute_bevformer_submission_metrics(evaluated["future_reference"], evaluated["detection_tracks"])
    bev["submission_ref_detection_recall"] = native["detection"]["visible_detection_recall"]
    act = compute_actbench_metrics(evaluated["reference_10hz"], evaluated["tracks_10hz"], formal["mapping"])
    bev["matched_agent_frame_coverage"] = coverage["matched_coverage"]
    act["matched_agent_frame_coverage"] = coverage["matched_coverage"]
    drivinggen = compute_drivinggen_metrics(formal, coverage)
    task = compute_task_fidelity_metrics(formal)
    worldlens = compute_worldlens_metrics(formal, act, drivinggen)
    independent_dir = s / "temporal_ablation" / "reset_prev_bev"
    independent_raw, independent_profile = _ensure_raw(config, pair, independent_dir, False)
    independent_eval = _evaluate_raw(config, pair, independent_raw, independent_dir)
    independent_bev = compute_bevformer_submission_metrics(independent_eval["future_reference"], independent_eval["detection_tracks"])
    independent_bev["submission_ref_detection_recall"] = independent_eval["native"]["detection"]["visible_detection_recall"]
    temporal = temporal_gain(bev, independent_bev, formal["identity"]["idf1_like"],
                             independent_eval["common"]["identity"]["idf1_like"], formal["identity"]["fragmentation_count"],
                             independent_eval["common"]["identity"]["fragmentation_count"])
    bev["temporal_diagnostics"] = temporal
    gate1 = {"status": "passed", "passed": validation["valid"], "future_gt_required": False}
    gate2 = history_quality_gate(evaluated["history"], config["gates"]["history_extractor_quality"])
    gate3_passed = (formal["presence"]["expected_visible_agent_frames"] >= config["gates"]["future_observability"]["minimum_visible_agent_frames"]
                    and coverage["matched_coverage"] >= config["gates"]["future_observability"]["minimum_match_coverage"])
    gate3 = {"status": "passed" if gate3_passed else "insufficient_observable_evidence", "passed": gate3_passed,
             "expected_visible_agent_frames": formal["presence"]["expected_visible_agent_frames"],
             "matched_coverage": coverage["matched_coverage"]}
    gate4 = _gate4(config, act, drivinggen, bev, task, coverage)
    status = "unscorable_extractor_failure" if not gate2["passed"] else ("insufficient_observable_evidence" if not gate3_passed else "valid_low_sample")
    paper = {"actbench": act, "drivinggen": drivinggen, "worldlens": worldlens, "bevformer": bev, "task_fidelity": task}
    for name, data in paper.items(): write_json(s / "paper_metrics" / f"{name}_metrics.json", data)
    write_json(s / "visibility_metrics" / "visibility_stratified.json", bev["visibility_stratified"])
    write_json(s / "temporal_ablation" / "metrics.json", {"temporal": temporal, "temporal_profile": profile,
                                                             "frame_independent_profile": independent_profile})
    _write_csv(s / "per_frame_metrics.csv", formal["detection"]["per_frame_rows"])
    _write_csv(s / "per_track_metrics.csv", formal["geometry"]["per_track"])
    write_json(s / "matching" / "track_mapping.json", formal["mapping"])
    write_json(s / "extracted_tracks" / "bev_tracks.json", evaluated["extracted"]["tracks"])
    metrics = {"protocol": "submission_only", "status": status, "protocol_completion_status": "valid_submission_only",
               "reference_semantics": "submitted_trajectory_as_instruction_reference", "pair_id": pair_id,
               "extractor_backend": config["backend"]["model_variant"], "checkpoint": config["backend"]["checkpoint"],
               "pixel_inference_performed": True, "future_gt_used": False, "history": evaluated["history"],
               "formal_common_10hz": formal, "coverage": coverage, "paper_metrics": paper,
               "hard_gates": {"gate_1_submission_integrity": gate1, "gate_2_history_extractor_reliability": gate2,
                              "gate_3_future_observability": gate3, "gate_4_submission_image_trajectory_consistency": gate4,
                              "gate_5_distribution_reference": {"status": "not_run_optional_distribution_protocol"},
                              "gate_6_paired_gt": {"status": "not_run_optional_gt_protocol"}},
               "scores": _scores(evaluated["history"], act, drivinggen, bev, task, coverage),
               "provisional_total_score": False, "inference_profile": profile, "access_audit": access,
               "prohibitions": {"planning_head_used": False, "motion_prediction_used": False,
                                "future_gt_updated_tracker": False, "closed_loop_used": False}}
    write_json(s / "metrics.json", metrics)
    (s / "report.md").write_text(_submission_report(metrics), encoding="utf-8")
    return metrics


def _submission_report(metrics: dict[str, Any]) -> str:
    b, a, d, t = (metrics["paper_metrics"][key] for key in ("bevformer", "actbench", "drivinggen", "task_fidelity"))
    c = metrics["coverage"]
    return f"""# Protocol S: Submission-only consistency

- status: `{metrics['status']}` (`{metrics['protocol_completion_status']}`)
- extractor: `{metrics['extractor_backend']}`
- reference semantics: `submitted_trajectory_as_instruction_reference`
- future GT used: `false`
- center-distance mAP: `{b['submission_ref_center_distance_mAP']}`
- mATE / mASE / mAOE / mAVE: `{b['submission_ref_mATE']}` / `{b['submission_ref_mASE']}` / `{b['submission_ref_mAOE']}` / `{b['submission_ref_mAVE']}`
- mAAE / NDS adapted: `{b['submission_ref_mAAE']}` / `{b['submission_ref_NDS_adapted']}`
- AP/APH IoU 0.5: `{b['submission_ref_3d_AP_iou_0_5']}` / `{b['submission_ref_3d_APH_iou_0_5']}`
- AP/APH IoU 0.7: `{b['submission_ref_3d_AP_iou_0_7']}` / `{b['submission_ref_3d_APH_iou_0_7']}`
- ACT motion accuracy / ADE / FDE: `{a['act_agent_motion_accuracy_adapted']}` / `{a['act_agent_ade_adapted']}` / `{a['act_agent_fde_adapted']}`
- DrivingGen DTW / Agent Missing / Agent Consistency / Trajectory Consistency: `{d['drivinggen_alignment_dtw']}` / `{d['agent_missing_ratio']}` / `{d['agent_consistency']}` / `{d['trajectory_consistency']}`
- submission-ref Precision / Recall / IDF1 / HOTA adapted: `{t['submission_ref_precision']}` / `{t['submission_ref_recall']}` / `{t['submission_ref_idf1']}` / `{t['submission_ref_hota_adapted']}`
- matched agent-frame / track coverage: `{c['matched_coverage']}` / `{c['track_match_coverage']}`

All detection names are submission-reference adaptations. They are not standard dataset-GT accuracy.
"""


def run_distribution(config: dict[str, Any], output: Path, pair_manifest: str, pair_id: str) -> dict[str, Any]:
    s = output / "protocol_s_submission_only"; s_metrics = _read_json(s / "metrics.json")
    pair = _pair(pair_manifest, pair_id); conditions = Path(pair["image_sequence"]["conditions_root"])
    submitted = condition_reference_tracks(conditions, 10, 90)
    image = _read_json(s / "extracted_tracks" / "bev_tracks.json")
    reference_path = config["protocols"]["distribution_reference"].get("reference_dataset")
    declarations = {"coordinate_system": "world", "heading_unit": "rad", "history_frames": 31, "ego_ids": ["Ego"]}
    real = trajectory_schema_to_tracks(load_tracks(reference_path, "ground_truth", declarations)) if reference_path and Path(reference_path).is_file() else []
    result = evaluate_distribution_reference(submitted, image, real)
    result.update({"reference_dataset": reference_path or NOT_AVAILABLE, "reference_track_count": len(real),
                   "current_scene_identity_matching_used": False, "source_submission_protocol_status": s_metrics["status"]})
    directory = output / "protocol_d_distribution_reference"; write_json(directory / "metrics.json", result)
    for name in ("submitted_vs_real", "image_extracted_vs_real", "image_extracted_vs_submitted"):
        write_json(directory / "feature_distributions" / f"{name}.json", result[name])
    speed = result["submitted_vs_real"]["features"].get("speed", {})
    (directory / "report.md").write_text(f"""# Protocol D: Distribution reference

- status: `{result['status']}`
- reference dataset: `{result['reference_dataset']}`
- reference tracks: `{len(real)}`
- official FTD: `{result['official_ftd']}`
- submitted-vs-real speed JS / Wasserstein / KS / MMD: `{speed.get('js_divergence', NOT_AVAILABLE)}` / `{speed.get('wasserstein_distance', NOT_AVAILABLE)}` / `{speed.get('ks_statistic', NOT_AVAILABLE)}` / `{speed.get('mmd', NOT_AVAILABLE)}`
- mode coverage / rare-mode coverage: `{result['submitted_vs_real'].get('mode_coverage', NOT_AVAILABLE)}` / `{result['submitted_vs_real'].get('rare_mode_coverage', NOT_AVAILABLE)}`

This low-sample distribution protocol never supplies identities or boxes to the current-scene extractor or tracker.
""", encoding="utf-8")
    return result


def run_paired_gt(config: dict[str, Any], output: Path, pair_manifest: str, pair_id: str) -> dict[str, Any]:
    pair = _pair(pair_manifest, pair_id); gt_root = Path(pair["image_sequence"].get("gt_future_root", "__absent__"))
    directory = output / "protocol_g_paired_gt"
    if not gt_root.is_dir():
        result = {"protocol": "paired_gt", "status": "not_run_optional_gt_protocol",
                  "reason": "paired_future_images_not_available", "blocks_protocol_s": False}
    else:
        gt_pair = copy.deepcopy(pair); gt_pair["image_sequence"]["generated_future_root"] = str(gt_root)
        raw, profile = _ensure_raw(config, gt_pair, directory / "extracted_tracks", True)
        evaluated = _evaluate_raw(config, gt_pair, raw, directory / "extracted_tracks")
        formal = evaluated["common"]
        adapted = compute_bevformer_submission_metrics(evaluated["future_reference"], evaluated["detection_tracks"])
        submission = _read_json(output / "protocol_s_submission_only" / "metrics.json")
        s_bev = submission["paper_metrics"]["bevformer"]; s_task = submission["paper_metrics"]["task_fidelity"]
        gap = lambda a, b: float(a) - float(b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else NOT_AVAILABLE
        result = {"protocol": "paired_gt", "status": "valid_low_sample_paired_gt_calibration",
                  "reference_semantics": "strictly_paired_condition_boxes_without_official_dataset_attributes",
                  "paired_image_frames": profile["frames"] - 10, "pixel_inference_performed": True,
                  "paired_ref_center_distance_mAP": adapted["submission_ref_center_distance_mAP"],
                  "paired_ref_mATE": adapted["submission_ref_mATE"], "paired_ref_mASE": adapted["submission_ref_mASE"],
                  "paired_ref_mAOE": adapted["submission_ref_mAOE"], "paired_ref_mAVE": adapted["submission_ref_mAVE"],
                  "paired_ref_idf1": formal["identity"]["idf1_like"],
                  "generated_minus_real_image": {
                      "center_distance_mAP": gap(s_bev["submission_ref_center_distance_mAP"], adapted["submission_ref_center_distance_mAP"]),
                      "mATE": gap(s_bev["submission_ref_mATE"], adapted["submission_ref_mATE"]),
                      "mAOE": gap(s_bev["submission_ref_mAOE"], adapted["submission_ref_mAOE"]),
                      "mAVE": gap(s_bev["submission_ref_mAVE"], adapted["submission_ref_mAVE"]),
                      "IDF1": gap(s_task["submission_ref_idf1"], formal["identity"]["idf1_like"])},
                  "standard_nuscenes_mAP": NOT_AVAILABLE, "standard_NDS": NOT_AVAILABLE,
                  "standard_waymo_AP": NOT_AVAILABLE,
                  "standard_unavailable_reason": "incompatible_dataset_format_missing_official_attributes_and_difficulty_labels",
                  "future_gt_updated_extractor_or_tracker": False, "blocks_protocol_s": False,
                  "inference_profile": profile}
    write_json(directory / "metrics.json", result)
    gaps = result.get("generated_minus_real_image", {})
    (directory / "report.md").write_text(f"""# Protocol G: Paired-GT calibration

- status: `{result['status']}`
- paired future frames: `{result.get('paired_image_frames', NOT_AVAILABLE)}`
- real-image center-distance mAP / mATE / mAOE / mAVE / IDF1: `{result.get('paired_ref_center_distance_mAP', NOT_AVAILABLE)}` / `{result.get('paired_ref_mATE', NOT_AVAILABLE)}` / `{result.get('paired_ref_mAOE', NOT_AVAILABLE)}` / `{result.get('paired_ref_mAVE', NOT_AVAILABLE)}` / `{result.get('paired_ref_idf1', NOT_AVAILABLE)}`
- generated-minus-real mAP / mATE / mAOE / mAVE / IDF1: `{gaps.get('center_distance_mAP', NOT_AVAILABLE)}` / `{gaps.get('mATE', NOT_AVAILABLE)}` / `{gaps.get('mAOE', NOT_AVAILABLE)}` / `{gaps.get('mAVE', NOT_AVAILABLE)}` / `{gaps.get('IDF1', NOT_AVAILABLE)}`
- standard metrics unavailable reason: `{result.get('reason', result.get('standard_unavailable_reason', '-'))}`

Protocol G is optional and never blocks Protocol S.
""", encoding="utf-8")
    return result


def _flatten_values(data: Any, output: dict[str, Any]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)): _flatten_values(value, output)
            else: output.setdefault(key, value)
    elif isinstance(data, list):
        for value in data: _flatten_values(value, output)


def _paper_outputs(output: Path, submission: dict[str, Any], distribution: dict[str, Any], paired: dict[str, Any]) -> None:
    paper = output / "paper_metrics"; values: dict[str, Any] = {}
    _flatten_values(submission.get("paper_metrics", {}), values); _flatten_values(distribution, values); _flatten_values(paired, values)
    speed_distribution = distribution.get("submitted_vs_real", {}).get("features", {}).get("speed", {})
    values.update({"trajectory_distribution_js": speed_distribution.get("js_divergence", NOT_AVAILABLE),
                   "trajectory_distribution_wasserstein": speed_distribution.get("wasserstein_distance", NOT_AVAILABLE),
                   "trajectory_distribution_ks": speed_distribution.get("ks_statistic", NOT_AVAILABLE),
                   "trajectory_distribution_mmd": speed_distribution.get("mmd", NOT_AVAILABLE)})
    bev = submission.get("paper_metrics", {}).get("bevformer", {})
    if bev.get("visibility_stratified"):
        values["visibility_stratified"] = "available"
        values["visibility_stratified_sample_count"] = bev.get("sample_count", NOT_AVAILABLE)
    if paired.get("standard_unavailable_reason"):
        for name in ("standard_nuscenes_mAP", "standard_NDS", "standard_waymo_AP"):
            values[f"{name}_unavailable_reason"] = paired["standard_unavailable_reason"]
    registry = build_registry(values); write_json(paper / "metric_availability.json", registry)
    rows = [{"metric": row["metric_name"], "paper": row["paper"], "original_definition": row["paper_definition"],
             "current_name": row["metric_name"], "protocol_s_status": "available" if row["protocol_s_available"] else "not_available",
             "protocol_d_status": "available" if row["protocol_d_available"] else "not_available",
             "protocol_g_status": "available" if row["protocol_g_available"] else "not_available",
             "implementation_level": row["implementation_level"], "reference_semantics": row["reference_semantics"],
             "current_value": row["current_value"], "sample_count": row["sample_count"],
             "requires_paired_gt": row["requires_paired_gt"], "requires_distribution_reference": row["requires_reference_distribution"],
             "unavailable_reason": row["unavailable_reason"]} for row in registry]
    _write_csv(paper / "metric_crosswalk.csv", rows)
    table = ["| 论文 | 原指标 | 当前指标 | Protocol | 当前值 | 复现级别 | 限制 |", "|---|---|---|---|---:|---|---|"]
    for row in registry:
        protocol = "G" if row["protocol_g_available"] else "D" if row["protocol_d_available"] else "S"
        table.append(f"| {row['paper']} | {row['paper_original_name']} | {row['metric_name']} | {protocol} | {row['current_value']} | {row['implementation_level']} | {row['unavailable_reason'] or '-'} |")
    (paper / "metric_crosswalk.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    for name, value in submission.get("paper_metrics", {}).items(): write_json(paper / f"{name}_metrics.json", value)


def run_protocols(config: dict[str, Any], mode: str, output_path: str | Path, pair_manifest: str, pair_id: str) -> dict[str, Any]:
    output = Path(output_path); output.mkdir(parents=True, exist_ok=True)
    if mode == "audit":
        submission = run_submission_only(config, output, pair_manifest, pair_id, False)
        distribution = {"status": "not_run_audit_mode"}; paired = {"status": "not_run_audit_mode"}
    else:
        submission = run_submission_only(config, output, pair_manifest, pair_id, True)
        distribution = run_distribution(config, output, pair_manifest, pair_id) if mode in {"full", "distribution_reference"} else {"status": "not_run_optional_distribution_protocol"}
        paired = run_paired_gt(config, output, pair_manifest, pair_id) if mode in {"full", "paired_gt"} else {"status": "not_run_optional_gt_protocol"}
    if submission.get("protocol_completion_status") == "valid_submission_only":
        submission["hard_gates"]["gate_5_distribution_reference"] = {
            "status": distribution["status"], "blocks_protocol_s": False}
        submission["hard_gates"]["gate_6_paired_gt"] = {
            "status": paired["status"], "blocks_protocol_s": False}
        write_json(output / "protocol_s_submission_only" / "metrics.json", submission)
        (output / "protocol_s_submission_only" / "report.md").write_text(_submission_report(submission), encoding="utf-8")
    _paper_outputs(output, submission, distribution, paired)
    final = {"implementation_status": "completed" if submission.get("protocol_completion_status") == "valid_submission_only" else "partial",
             "evaluation_role": "perception_utility_diagnostic",
             "status": submission["status"], "current_formal_protocol": "submission_only",
             "submission_only": submission, "distribution_reference": distribution, "paired_gt_calibration": paired,
             "provisional_total_score": False}
    write_json(output / "final_metrics.json", final)
    (output / "final_report.md").write_text(_final_report(final), encoding="utf-8")
    return final


def _final_report(final: dict[str, Any]) -> str:
    s, d, g = final["submission_only"], final["distribution_reference"], final["paired_gt_calibration"]
    if s.get("protocol_completion_status") != "valid_submission_only":
        return f"# Protocol evaluation\n\nstatus: `{s['status']}`\n"
    b, a, dg, t = (s["paper_metrics"][key] for key in ("bevformer", "actbench", "drivinggen", "task_fidelity"))
    gates = s["hard_gates"]; coverage = s["coverage"]; profile = s["inference_profile"]
    d_speed = d.get("submitted_vs_real", {}).get("features", {}).get("speed", {})
    g_gap = g.get("generated_minus_real_image", {})
    gate4 = gates["gate_4_submission_image_trajectory_consistency"]
    return f"""# 六视角生成图像—结构化轨迹一致性报告

## 1. 实现状态

`{final['implementation_status']}`；正式协议 `submission_only`；场景状态 `{s['status']}`。

评价角色：`perception_utility_diagnostic`。MTPC启用后，本报告不再是图像—轨迹一致性唯一主分。

## 2. Protocol S 无未来 GT

- reference: `submitted_trajectory_as_instruction_reference`
- extractor: `{s['extractor_backend']}`
- checkpoint: `{s['checkpoint']}`
- inference: `{profile.get('frames', NOT_AVAILABLE)}` frames, mean `{profile.get('mean_inference_seconds', NOT_AVAILABLE)}` s/frame, peak `{profile.get('peak_gpu_memory_mib', NOT_AVAILABLE)}` MiB
- future GT accessed: `{s['future_gt_used']}`
- physical future-GT isolation audit: `protocol_s_submission_only/future_gt_physical_isolation_audit.json`
- history Recall / Precision / IDF1: `{s['history']['history_detection_recall']}` / `{s['history']['history_detection_precision']}` / `{s['history']['history_baseline_idf1']}`
- submission-ref Precision / Recall / IDF1 / HOTA adapted: `{t['submission_ref_precision']}` / `{t['submission_ref_recall']}` / `{t['submission_ref_idf1']}` / `{t['submission_ref_hota_adapted']}`
- ACT motion Accuracy / ADE / FDE: `{a['act_agent_motion_accuracy_adapted']}` / `{a['act_agent_ade_adapted']}` / `{a['act_agent_fde_adapted']}`
- DrivingGen DTW / Missing / Agent Consistency / Trajectory Consistency: `{dg['drivinggen_alignment_dtw']}` / `{dg['agent_missing_ratio']}` / `{dg['agent_consistency']}` / `{dg['trajectory_consistency']}`
- matched agent-frame coverage: `{coverage['matched_visible_agent_frames']}` / `{coverage['expected_visible_agent_frames']}` = `{coverage['matched_coverage']}`
- matched track coverage: `{coverage['matched_submitted_tracks']}` / `{coverage['visible_submitted_tracks']}` = `{coverage['track_match_coverage']}`

## 3. BEVFormer submission-reference metrics

- center-distance mAP: `{b['submission_ref_center_distance_mAP']}`
- mATE / mASE / mAOE / mAVE: `{b['submission_ref_mATE']}` / `{b['submission_ref_mASE']}` / `{b['submission_ref_mAOE']}` / `{b['submission_ref_mAVE']}`
- mAAE: `{b['submission_ref_mAAE']}` (`missing_attributes`)
- NDS adapted: `{b['submission_ref_NDS_adapted']}` (not computed without mAAE)
- AP/APH@3D-IoU 0.5: `{b['submission_ref_3d_AP_iou_0_5']}` / `{b['submission_ref_3d_APH_iou_0_5']}`
- AP/APH@3D-IoU 0.7: `{b['submission_ref_3d_AP_iou_0_7']}` / `{b['submission_ref_3d_APH_iou_0_7']}`
- temporal diagnostic: `{b['temporal_diagnostics']}`

These are image/submission consistency metrics, not standard dataset-GT mAP/NDS/Waymo AP.

## 4. Optional protocols

- Protocol D: `{d['status']}`; reference tracks: `{d.get('reference_track_count', NOT_AVAILABLE)}`; official FTD: `{d.get('official_ftd', NOT_AVAILABLE)}`
- submitted-vs-real speed JS / Wasserstein / KS / MMD: `{d_speed.get('js_divergence', NOT_AVAILABLE)}` / `{d_speed.get('wasserstein_distance', NOT_AVAILABLE)}` / `{d_speed.get('ks_statistic', NOT_AVAILABLE)}` / `{d_speed.get('mmd', NOT_AVAILABLE)}`
- submitted-vs-real mode coverage / rare-mode coverage: `{d.get('submitted_vs_real', {}).get('mode_coverage', NOT_AVAILABLE)}` / `{d.get('submitted_vs_real', {}).get('rare_mode_coverage', NOT_AVAILABLE)}`
- Protocol G: `{g['status']}`; real-image mAP / mATE / mAOE / mAVE / IDF1: `{g.get('paired_ref_center_distance_mAP', NOT_AVAILABLE)}` / `{g.get('paired_ref_mATE', NOT_AVAILABLE)}` / `{g.get('paired_ref_mAOE', NOT_AVAILABLE)}` / `{g.get('paired_ref_mAVE', NOT_AVAILABLE)}` / `{g.get('paired_ref_idf1', NOT_AVAILABLE)}`
- generated-minus-real mAP / mATE / mAOE / mAVE / IDF1: `{g_gap.get('center_distance_mAP', NOT_AVAILABLE)}` / `{g_gap.get('mATE', NOT_AVAILABLE)}` / `{g_gap.get('mAOE', NOT_AVAILABLE)}` / `{g_gap.get('mAVE', NOT_AVAILABLE)}` / `{g_gap.get('IDF1', NOT_AVAILABLE)}`
- standard nuScenes/Waymo status: `not_available` (`{g.get('standard_unavailable_reason', 'requires_paired_gt')}`)

## 5. Hard Gates

{chr(10).join(f'- {name}: `{value["status"]}`' for name, value in gates.items())}

Gate 4 provisional failures: `{gate4.get('failures', [])}`. Optional Protocol D/G outcomes do not block Protocol S.

All thresholds are provisional; one scene is insufficient for a stable leaderboard conclusion. No planning head or closed loop was run.
"""

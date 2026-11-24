from __future__ import annotations

from driveaudit_eval.consistency.mtpc.common import load_geometry as _geometry

import inspect
import json
import hashlib
import time
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.shared.conditions import condition_reference_tracks
from driveaudit_eval.consistency.shared.common import json_cell, read_csv

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.backends import OpenCVFlowTracker
from driveaudit_eval.consistency.mtpc.common import CAMERAS, image_fingerprint, image_path, load_config, load_pair, mean, read_json, score_from_error, write_csv, write_json
from driveaudit_eval.consistency.mtpc.expected_image_motion import compute_expected_motion
from driveaudit_eval.consistency.mtpc.future_entrant_verification import verify_future_entrants
from driveaudit_eval.consistency.mtpc.history_anchor import build_history_projections
from driveaudit_eval.consistency.mtpc.metrics import (compute_history_calibrated, compute_identity_metrics, compute_image_motion_metrics,
                      compute_multiview_metrics, compute_presence_metrics, compute_projection_geometry_metrics,
                      compute_temporal_event_metrics)
from driveaudit_eval.consistency.mtpc.project_submitted_trajectory import project_submitted_trajectory
from driveaudit_eval.consistency.mtpc.track_future_images import track_future_images
from driveaudit_eval.consistency.mtpc.visualization import generate_visualizations


def _history_stage(config: dict[str, Any], pair: dict[str, Any], output: Path, history_length: int = 10,
                   mode: str = "fusion", track_end: int = 90, result_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    conditions = pair["image_sequence"]["conditions_root"]; geometry = _geometry(conditions)
    history_tracks = condition_reference_tracks(conditions, 0, 10); history_ids = {track["agent_id"] for track in history_tracks}
    projections = build_history_projections(history_tracks, geometry["calibrations"],
        (int(config["projection"]["image_width"]), int(config["projection"]["image_height"])),
        float(config["projection"]["minimum_projected_area_px"]), float(config["projection"]["maximum_occlusion_ratio"]))
    start = 10 - history_length; projections = [row for row in projections if row["frame_index"] >= start]
    write_json(output / "history_anchor/history_reference_projection.json", projections)
    history_images = {camera: [str(image_path(pair, camera, frame)) for frame in range(start, 10)] for camera in CAMERAS}
    timestamps = [geometry["timestamps"][frame] for frame in range(start, 10)]
    backend = OpenCVFlowTracker(config["tracking"], mode); backend.reset(pair["pair_id"])
    # Reindex the history subset for the backend while preserving physical timestamps.
    subset = [{**row, "frame_index": row["frame_index"] - start} for row in projections]
    history = backend.initialize(history_images, subset, sorted(history_ids), geometry["calibrations"], geometry["poses"], timestamps)
    directory = result_dir or output / "extracted_visual_tracks"
    tracked = track_future_images(backend, pair["image_sequence"]["generated_future_root"], geometry["calibrations"], geometry["poses"],
                                  geometry["timestamps"], 10, track_end, directory)
    return history, tracked, projections


def _evaluate(config: dict[str, Any], projections: list[dict[str, Any]], motion_rows: list[dict[str, Any]],
              visual: list[dict[str, Any]], history: dict[str, Any]) -> dict[str, Any]:
    presence = compute_presence_metrics(projections, visual); geometry = compute_projection_geometry_metrics(projections, visual)
    motion = compute_image_motion_metrics(motion_rows, visual); temporal = compute_temporal_event_metrics(projections, visual)
    identity = compute_identity_metrics(projections, visual); multiview = compute_multiview_metrics(projections, visual, motion["per_frame"])
    calibrated = compute_history_calibrated(history, geometry, motion, presence, identity, multiview)
    return {"presence": presence, "projection_geometry": geometry, "image_motion": motion, "temporal_events": temporal,
            "identity": identity, "multiview": multiview, "history_calibrated": calibrated}


def _bevformer_diagnostic() -> dict[str, Any]:
    path = Path("challenge/image_trajectory_eval/final_metrics.json")
    if not path.is_file(): return {"role": "perception_utility_diagnostic", "status": "not_available"}
    source = read_json(path); submission = source.get("submission_only", {}); papers = submission.get("paper_metrics", {})
    bev, task, act, driving = (papers.get(key, {}) for key in ("bevformer", "task_fidelity", "actbench", "drivinggen"))
    return {"role": "perception_utility_diagnostic", "backend": submission.get("extractor_backend", NOT_AVAILABLE), "status": submission.get("status", NOT_AVAILABLE),
            "metrics_path": str(path), "submission_ref_mAP": bev.get("submission_ref_center_distance_mAP", NOT_AVAILABLE),
            "mATE": bev.get("submission_ref_mATE", NOT_AVAILABLE), "mAOE": bev.get("submission_ref_mAOE", NOT_AVAILABLE),
            "mAVE": bev.get("submission_ref_mAVE", NOT_AVAILABLE), "IDF1": task.get("submission_ref_idf1", NOT_AVAILABLE),
            "agent_missing": driving.get("agent_missing_ratio", NOT_AVAILABLE), "matched_coverage": submission.get("coverage", {}).get("matched_coverage", NOT_AVAILABLE),
            "motion_accuracy": act.get("act_agent_motion_accuracy_adapted", NOT_AVAILABLE), "warning_only": True, "enters_mtpc_primary_gate": False}


def _gates(config: dict[str, Any], history: dict[str, Any], metrics: dict[str, Any], entrants: dict[str, Any], bev: dict[str, Any], input_valid: bool) -> dict[str, Any]:
    h, f, c = config["gates"]["history_anchor"], config["gates"]["future_observability"], config["gates"]["consistency"]
    gate2_checks = {"visible_samples": history["history_projected_visible_agent_frames"] >= h["minimum_visible_samples"],
                    "point_survival": isinstance(history["history_point_survival"], (int, float)) and history["history_point_survival"] >= h["minimum_point_survival"],
                    "center_error": isinstance(history["history_center_tracking_error_px"], (int, float)) and history["history_center_tracking_error_px"] <= h["maximum_center_error_px"],
                    "identity": isinstance(history["history_identity_persistence"], (int, float)) and history["history_identity_persistence"] >= h["minimum_identity_persistence"]}
    p, g, m, i, v = (metrics[key] for key in ("presence", "projection_geometry", "image_motion", "identity", "multiview"))
    gate3 = p["expected_visible_agent_camera_frames"] >= f["minimum_expected_visible_agent_camera_frames"] and p["supported_agent_camera_frames"] >= f["minimum_supported_agent_camera_frames"]
    checks = {"support_rate": p["persistent_agent_visual_support_rate"] >= c["minimum_support_rate"],
              "normalized_center_error": isinstance(g["normalized_projected_center_error"], (int, float)) and g["normalized_projected_center_error"] <= c["maximum_normalized_center_error"],
              "box_iou": isinstance(g["box_iou_2d"], (int, float)) and g["box_iou_2d"] >= c["minimum_box_iou"],
              "flow_epe": isinstance(m["flow_endpoint_error_px"], (int, float)) and m["flow_endpoint_error_px"] <= c["maximum_flow_epe_px"],
              "direction": isinstance(m["motion_direction_agreement"], (int, float)) and m["motion_direction_agreement"] >= c["minimum_direction_agreement"],
              "identity": i["identity_persistence"] >= c["minimum_identity_persistence"],
              "camera_support": v["camera_support_recall"] >= c["minimum_camera_support_recall"]}
    return {"gate_1_input_calibration_integrity": {"status": "passed" if input_valid else "failed", "passed": input_valid},
            "gate_2_history_anchor_quality": {"status": "passed" if all(gate2_checks.values()) else "unscorable_anchor_failure", "passed": all(gate2_checks.values()), "checks": gate2_checks},
            "gate_3_future_observable_evidence": {"status": "passed" if gate3 else "insufficient_observable_evidence", "passed": gate3},
            "gate_4_mtpc_consistency": {"status": "passed" if all(checks.values()) else "failed_provisional", "passed": all(checks.values()), "checks": checks},
            "gate_5_future_entrant_optional": {"status": entrants["status"], "blocks_primary": False},
            "gate_6_bevformer_diagnostic": {"status": "warning_only", "blocks_primary": False, "diagnostic_status": bev["status"]},
            "provisional_thresholds": True}


def _scorecard(history: dict[str, Any], metrics: dict[str, Any], entrants: dict[str, Any], bev: dict[str, Any]) -> dict[str, Any]:
    p, g, m, t, i, v = (metrics[key] for key in ("presence", "projection_geometry", "image_motion", "temporal_events", "identity", "multiview"))
    return {"protocol": "mtpc_history_anchored", "primary_scope": "history_anchored_agents", "scores": {
        "history_anchor_reliability": 100 * mean([history["history_point_survival"], history["history_identity_persistence"], history["history_cross_view_consistency"]]),
        "persistent_object_presence": 100 * p["persistent_agent_visual_support_rate"],
        "projection_geometry_consistency": mean([score_from_error(g["normalized_projected_center_error"], 1.0), 100 * g["box_iou_2d"], 100 * g["mask_iou"]]),
        "image_motion_fidelity": mean([score_from_error(m["normalized_flow_endpoint_error"], 2.0), 50 * (m["motion_direction_agreement"] + 1)]),
        "temporal_event_fidelity": score_from_error(mean([t["entry_time_error_s"], t["exit_time_error_s"]]), 2.0),
        "identity_persistence": 100 * i["identity_persistence"], "multi_view_support": 100 * v["camera_support_recall"],
        "conditional_future_entrant_consistency": 100 * entrants["future_entrant_support_rate"] if isinstance(entrants["future_entrant_support_rate"], (int, float)) else None,
        "bevformer_perception_utility_diagnostic": None, "optional_mask_consistency": None},
        "future_entrant_consistency": entrants, "perception_utility_diagnostic": bev, "provisional_total_score": False}


def _rows(metrics: dict[str, Any], output: Path) -> None:
    frame_rows = []
    for family in ("presence", "projection_geometry", "image_motion"):
        for row in metrics[family].get("per_frame", []): frame_rows.append({"metric_family": family, **row})
    write_csv(output / "metrics/per_frame.csv", frame_rows)
    agents = sorted({row.get("agent_id") for row in frame_rows if row.get("agent_id")}); write_csv(output / "metrics/per_agent.csv", [{"agent_id": agent, "samples": sum(row.get("agent_id") == agent for row in frame_rows)} for agent in agents])
    per_camera = metrics["multiview"]["per_camera"]; write_csv(output / "metrics/per_camera.csv", [{"camera_name": camera, **values} for camera, values in per_camera.items()])






def run(config_path: str | Path, mode: str = "full") -> dict[str, Any]:
    config = load_config(config_path); pair = load_pair(config); output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    geometry = _geometry(pair["image_sequence"]["conditions_root"]); input_valid = all(image_path(pair, camera, frame).is_file() for camera in CAMERAS for frame in range(90)) and len(geometry["calibrations"]) == 540
    audit = {"pair_id": pair["pair_id"], "scene_count": 1, "history_frames": 10, "future_frames": 80, "camera_count": 6,
             "camera_order": list(CAMERAS), "image_size": [400, 224], "input_valid": input_valid,
             "stage_a_forbidden_parameters_absent": not any(name in inspect.signature(track_future_images).parameters for name in ("submitted_future_trajectory", "future_gt_boxes", "future_gt_ids")),
             "formal_backend": "opencv_lk_farneback_ransac_fusion", "future_gt_required": False}
    write_json(output / "input_audit.json", audit); write_json(output / "manifest.json", {"protocol": "mtpc_history_anchored", "pair_id": pair["pair_id"]})
    if mode == "audit": return {"implementation_status": "audit_completed", "input_audit": audit}
    track_end = 10 if mode == "history" else 15 if mode == "smoke" else 90
    image_hash_before = image_fingerprint(pair); trajectory_path = Path(pair["trajectory_sequence"]["trajectory_path"])
    trajectory_hash_before = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    start = time.time(); history, tracked, history_projection = _history_stage(config, pair, output, 10, config["tracking"]["formal_mode"], track_end)
    write_json(output / "history_anchor/history_reference_projection.json", history_projection); write_json(output / "history_anchor/metrics.json", history)
    stage_a_finished = time.time(); access = {"stage_a_future_submission_trajectory_accessed": False, "stage_a_future_gt_accessed": False,
        "stage_a_extractor_inputs": [pair["image_sequence"]["history_root"], pair["image_sequence"]["generated_future_root"], pair["calibration"]["path"], pair["ego_pose"]["path"], "history_anchor/history_reference_projection.json"],
        "visual_tracks_persisted_before_stage_b": (output / "extracted_visual_tracks/per_agent_tracks.json").is_file(), "stage_a_finished_unix": stage_a_finished,
        "input_image_sha256": image_hash_before, "trajectory_sha256_before": trajectory_hash_before}
    write_json(output / "data_isolation_audit.json", access)
    if mode == "history": return {"implementation_status": "history_completed", "history": history}
    projected = project_submitted_trajectory(pair["image_sequence"]["conditions_root"], geometry["calibrations"],
        {track["agent_id"] for track in condition_reference_tracks(pair["image_sequence"]["conditions_root"], 0, 10)}, config["projection"], output / "submitted_projection")
    access.update({"stage_b_submission_reference_read": True, "stage_b_started_unix": time.time(), "stage_b_after_stage_a": time.time() >= stage_a_finished,
                   "original_images_unchanged": image_fingerprint(pair) == image_hash_before,
                   "original_trajectory_unchanged": hashlib.sha256(trajectory_path.read_bytes()).hexdigest() == trajectory_hash_before}); write_json(output / "data_isolation_audit.json", access)
    motion_rows = compute_expected_motion(projected["rows"], output / "expected_motion"); metrics = _evaluate(config, projected["rows"], motion_rows, tracked["observations"], history)
    entrants = verify_future_entrants(projected["rows"], pair["image_sequence"]["generated_future_root"], config["future_entrants"], output / "future_entrants") if config["future_entrants"]["enabled"] else {"status": "disabled", "future_entrant_support_rate": NOT_AVAILABLE}
    bev = _bevformer_diagnostic(); gates = _gates(config, history, metrics, entrants, bev, input_valid); scorecard = _scorecard(history, metrics, entrants, bev)
    _rows(metrics, output); write_json(output / "metrics/scorecard.json", scorecard)
    research_diagnostics = config.get("research_diagnostics", True)
    if research_diagnostics:
        from driveaudit_eval.research.consistency.mtpc_corruption import run_corruption_benchmark
        from driveaudit_eval.research.consistency.mtpc_ablation import _run_ablations
    corruption = run_corruption_benchmark(projected["rows"], tracked["observations"], output / "corruption_benchmark") if research_diagnostics and mode in {"corruption", "full"} else {"status": "not_run"}
    ablations = _run_ablations(config, pair, output, projected["rows"], motion_rows, history, tracked, metrics) if research_diagnostics and mode in {"ablation", "full"} else {"status": "not_run"}
    visualizations = generate_visualizations(pair, history_projection, projected["rows"], tracked["observations"], metrics, entrants, bev, corruption, output / "visualizations") if research_diagnostics and mode == "full" and config.get("retain_intermediates", True) else []
    result = {"implementation_status": "completed", "evaluation_role": "long_horizon_tracking_diagnostic",
              "protocol": "mtpc_history_anchored", "status": "valid_low_sample",
              "primary_scope": "history_anchored_agents", "pair_id": pair["pair_id"], "input_audit": audit, "history_anchor": history,
              "data_statistics": {"scene_count": 1, "history_frames": 10, "future_frames": 80, "camera_count": 6,
                  "history_agent_count": len({row["agent_id"] for row in history_projection}), "history_anchored_agent_count": len(projected["persistent_agent_ids"]),
                  "future_entrant_count": len(projected["future_entrant_ids"]), "expected_visible_persistent_agent_camera_frames": metrics["presence"]["expected_visible_agent_camera_frames"]},
              "primary_metrics": metrics, "future_entrant_consistency": entrants, "perception_utility_diagnostic": bev,
              "optional_mask_protocol": {"status": "not_run_optional_no_submitted_instance_masks", "blocks_primary": False},
              "hard_gates": gates, "scorecard": scorecard, "inference_profile": {key: tracked[key] for key in tracked if key != "observations" and key != "history_observations"},
              "corruption_benchmark": corruption, "ablations": ablations, "visualizations": visualizations,
              "prohibitions": {"future_submission_updated_tracker": False, "future_gt_used": False, "bevformer_in_primary_gate": False,
                               "planning_head_used": False, "closed_loop_used": False}, "runtime_seconds": time.time() - start}
    write_json(output / "metrics/per_scene.json", result); write_csv(output / "metrics/per_scene.csv", [{"pair_id": pair["pair_id"], "status": result["status"], **{key: value for key, value in metrics["presence"].items() if not isinstance(value, (dict, list))}}])
    write_json(output / "final_metrics.json", result); return result

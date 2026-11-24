from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks

from driveaudit_eval.research.consistency.trajectory_verification.common import mean, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.expected_motion import expected_motion
from driveaudit_eval.research.consistency.trajectory_verification.metrics import (appearance_metrics, local_motion_metrics, multiview_metrics, presence_metrics,
                      projection_geometry_metrics, temporal_event_metrics)
from driveaudit_eval.research.consistency.trajectory_verification.projection import project_tracks
from driveaudit_eval.research.consistency.trajectory_verification.sliding_window_verifier import verify_windows


def _metrics(evidence: list[dict[str, Any]], projections: list[dict[str, Any]], primary_window: int,
             dt: float, weights: dict[str, float] | None = None) -> dict[str, Any]:
    return {"presence": presence_metrics(evidence, projections, primary_window, weights),
            "local_motion": local_motion_metrics(evidence, primary_window, weights),
            "projection_geometry": projection_geometry_metrics(evidence, primary_window, weights),
            "temporal_events": temporal_event_metrics(evidence, projections, primary_window, dt),
            "multiview": multiview_metrics(evidence, primary_window),
            "appearance": appearance_metrics(evidence, primary_window)}


def run_history_calibration(pair: dict[str, Any], calibrations: list[dict[str, Any]], config: dict[str, Any],
                            output: str | Path, evidence_mode: str = "fusion",
                            window_lengths: list[int] | None = None) -> dict[str, Any]:
    output = Path(output)
    history_tracks = condition_reference_tracks(pair["image_sequence"]["conditions_root"], 0, 10)
    history_ids = {track["agent_id"] for track in history_tracks}
    projected = project_tracks(pair["image_sequence"]["conditions_root"], calibrations, 0, 10, history_ids,
                               config["projection"], output / "projection")
    lengths = window_lengths or list(config["windows"]["formal_lengths"])
    motion = expected_motion(projected["rows"], lengths, output / "expected_motion")
    verified = verify_windows(pair, motion, config, output / "local_visual_evidence", evidence_mode, frame_end=10)
    primary = int(config["windows"]["primary_length"])
    metrics = _metrics(verified["rows"], projected["rows"], primary, float(config["timeline"]["dt"]), None)
    rows = [row for row in verified["rows"] if int(row["window_length"]) == primary]
    by_camera = {}
    for camera in config["camera_order"]:
        items = [row for row in rows if row["camera_name"] == camera]
        by_camera[camera] = {"samples": len(items),
                             "support_rate": mean(float(row["visual_supported"]) for row in items),
                             "valid_point_ratio": mean(row.get("valid_point_ratio") for row in items),
                             "flow_epe": mean(row.get("local_flow_epe") for row in items),
                             "flow_cosine": mean(row.get("flow_cosine_similarity") for row in items),
                             "feature_support": mean(row.get("feature_support_score") for row in items),
                             "foreground_contrast": mean(row.get("foreground_contrast_score") for row in items)}
    buckets = {}
    for label, predicate in {
        "small_64_256": lambda row: row["projected_area"] < 256,
        "medium_256_1024": lambda row: 256 <= row["projected_area"] < 1024,
        "large_gt_1024": lambda row: row["projected_area"] >= 1024,
        "near_0_20m": lambda row: row["projected_depth"] < 20,
        "mid_20_40m": lambda row: 20 <= row["projected_depth"] < 40,
        "far_ge_40m": lambda row: row["projected_depth"] >= 40,
    }.items():
        items = [row for row in rows if predicate(row)]
        buckets[label] = {"samples": len(items), "support_rate": mean(float(row["visual_supported"]) for row in items),
                          "flow_epe": mean(row.get("local_flow_epe") for row in items)}
    baseline = {"history_frames_used": 10, "history_projected_agent_camera_windows": len(rows),
                "history_support_rate": metrics["presence"]["projection_support_rate"],
                "history_flow_epe": metrics["local_motion"]["local_flow_epe"],
                "history_flow_cosine": metrics["local_motion"]["flow_cosine_similarity"],
                "history_direction_agreement": metrics["local_motion"]["motion_direction_agreement"],
                "history_valid_point_ratio": metrics["local_motion"]["valid_point_ratio"],
                "history_feature_support": metrics["appearance"]["local_feature_support"],
                "history_cross_view_support": metrics["multiview"]["camera_support_recall"],
                "metrics": metrics, "per_camera": by_camera, "per_bucket": buckets,
                "profile": verified["profile"], "visual_evidence_from_real_history_images": True,
                "history_gt_used_only_as_projection_expectation": True}
    write_csv(output / "per_frame.csv", rows)
    write_json(output / "per_camera.json", by_camera)
    write_json(output / "per_bucket.json", buckets)
    write_json(output / "baseline_distribution.json", baseline)
    return baseline

from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.bev_geometry_metrics import compute_bev_geometry
from driveaudit_eval.research.consistency.perception_trajectory.detection_metrics import compute_detection_metrics
from driveaudit_eval.research.consistency.perception_trajectory.identity_metrics import compute_identity_metrics
from driveaudit_eval.research.consistency.perception_trajectory.motion_metrics import compute_motion_metrics
from driveaudit_eval.research.consistency.perception_trajectory.presence_metrics import compute_presence_metrics
from driveaudit_eval.research.consistency.perception_trajectory.reprojection_metrics import compute_reprojection_metrics
from driveaudit_eval.research.consistency.perception_trajectory.track_matcher import match_tracks
from driveaudit_eval.consistency.shared.visibility import evaluate_frame_visibility


def apply_visibility(submission_tracks: list[dict[str, Any]], calibrations: list[dict[str, Any]],
                     config: dict[str, Any]) -> list[dict[str, Any]]:
    by_frame: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for track in submission_tracks:
        for observation in track["observations"]:
            sample = {**observation, "agent_id": track["agent_id"]}
            by_frame[int(observation["frame_index"])].append((observation, sample))
    calibration_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for calibration in calibrations: calibration_by_frame[int(calibration["frame_index"])].append(calibration)
    flat = []
    for frame, pairs in by_frame.items():
        results = evaluate_frame_visibility([sample for _, sample in pairs], calibration_by_frame[frame], config)
        for (observation, _), result in zip(pairs, results):
            observation.update({key: result[key] for key in ("expected_visible", "visible_camera_count", "projected_box_area",
                                                               "occlusion_ratio", "small_object_flag", "truncated_flag")})
            observation["visibility_status"] = "expected_visible" if result["expected_visible"] else "not_observable"
            if observation.get("out_of_detector_range", False):
                observation["expected_visible"] = False
                observation["visibility_status"] = "out_of_detector_range"
            flat.append(result)
    return flat


def evaluate_formal(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                    visibility_rows: list[dict[str, Any]], config: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    mapping = match_tracks(submission_tracks, image_tracks, config["matching"])
    detection = compute_detection_metrics(submission_tracks, image_tracks, config["detection_evaluation"])
    presence = compute_presence_metrics(submission_tracks, image_tracks, mapping)
    geometry = compute_bev_geometry(submission_tracks, image_tracks, mapping, history.get("history_center_ade", NOT_AVAILABLE))
    motion = compute_motion_metrics(submission_tracks, image_tracks, mapping, config.get("speed_estimation"))
    identity = compute_identity_metrics(submission_tracks, image_tracks, mapping, config["detection_evaluation"])
    reprojection = compute_reprojection_metrics(visibility_rows, image_tracks)
    return {"mapping": mapping, "detection": detection, "presence": presence, "geometry": geometry, "motion": motion,
            "identity": identity, "reprojection": reprojection}


def dimension_scores(metrics: dict[str, Any], history: dict[str, Any], config: dict[str, Any]) -> dict[str, float | str]:
    def upper(value: Any, threshold: float) -> float | str:
        return max(0.0, 100.0 * (1.0 - float(value) / threshold)) if isinstance(value, (float, int)) else NOT_AVAILABLE
    p, g, m, i = metrics["presence"], metrics["geometry"], metrics["motion"], metrics["identity"]
    history_values = [history.get("history_detection_recall"), history.get("history_baseline_idf1")]
    return {
        "history_calibration_quality": 100.0 * sum(history_values) / len(history_values) if all(isinstance(x, (float, int)) for x in history_values) else NOT_AVAILABLE,
        "presence_consistency": 100.0 * p["visible_agent_recall"] if isinstance(p["visible_agent_recall"], (float, int)) else NOT_AVAILABLE,
        "bev_geometry_consistency": upper(g["excess_center_ade"], config["gates"]["generated_relative_consistency"]["maximum_center_error_increase_m"]),
        "motion_consistency": upper(m["speed_mae"], config["gates"]["extractor_absolute_quality_warning"]["maximum_gt_speed_mae_mps"]),
        "identity_temporal_consistency": 100.0 * i["idf1_like"] if isinstance(i["idf1_like"], (float, int)) else NOT_AVAILABLE,
        "multi_view_support": NOT_AVAILABLE,
    }

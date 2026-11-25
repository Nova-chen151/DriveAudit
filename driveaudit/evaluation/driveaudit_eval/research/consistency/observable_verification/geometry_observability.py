from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.hypothesis_verification.background_hypothesis import build as background_build
from driveaudit_eval.research.consistency.hypothesis_verification.target_hypothesis import build as target_build

from driveaudit_eval.research.consistency.observable_verification.historical_noise import select_sigma


def sample_core_points(projection: dict[str, Any], count: int = 25) -> list[list[float]]:
    polygon = np.asarray(projection.get("projected_polygon", []), np.float32)
    if len(polygon) < 3: return []
    x, y, w, h = cv2.boundingRect(polygon); points = []
    for v in np.linspace(y + .2 * h, y + .8 * h, int(math.sqrt(count))):
        for u in np.linspace(x + .2 * w, x + .8 * w, int(math.sqrt(count))):
            if cv2.pointPolygonTest(polygon, (float(u), float(v)), False) >= 0: points.append([float(u), float(v)])
    return points


def evaluate(expectation: dict[str, Any], intermediate: list[dict[str, Any]], source_calibration: dict[str, Any],
             target_calibration: dict[str, Any], noise_model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source, target = expectation["source_projection"], expectation["target_projection"]
    points = sample_core_points(source); observed = {"observed_points": points, "source_projection": source,
        "target_projection": target, "projected_depth": source.get("projected_depth")}
    h_target = target_build(observed, config["target_hypothesis"])
    h_background = background_build(observed, source_calibration, target_calibration, config["background_hypothesis"], tuple(config["image"]["size"]))
    target_flow = np.asarray(h_target["target_expected_flow_per_point"], float)
    candidates = h_background["background_hypotheses"]
    separation_rows = []
    for candidate in candidates:
        background_flow = np.asarray(candidate["expected_flow_per_point"], float); n = min(len(target_flow), len(background_flow))
        if not n: continue
        delta = target_flow[:n] - background_flow[:n]
        separation = float(np.median(np.linalg.norm(delta, axis=1)))
        tv, bv = np.median(target_flow[:n], axis=0), np.median(background_flow[:n], axis=0)
        norm = np.linalg.norm(tv) * np.linalg.norm(bv)
        angle = float(np.arccos(np.clip(np.dot(tv, bv) / norm, -1, 1))) if norm > 1e-6 else 0.0
        magnitude = float(abs(np.linalg.norm(tv) - np.linalg.norm(bv)))
        separation_rows.append({"depth_source": candidate["depth_source"], "separation_px": separation,
                                "angle_separation_rad": angle, "magnitude_separation_px": magnitude,
                                "expected_flow_per_point": candidate["expected_flow_per_point"]})
    closest = min(separation_rows, key=lambda r: r["separation_px"]) if separation_rows else None
    zero = float(np.median(np.linalg.norm(target_flow, axis=1))) if len(target_flow) else 0.0
    medians = np.asarray([np.median(np.asarray(c["expected_flow_per_point"], float), axis=0) for c in candidates if c["expected_flow_per_point"]], float)
    spread = float(np.median(np.linalg.norm(medians - np.median(medians, axis=0), axis=1))) if len(medians) else 0.0
    area = min(float(source.get("projected_area") or 0), float(target.get("projected_area") or 0))
    distance = float(source.get("projected_depth") or 1e9); sigma_history, fallback = select_sigma(noise_model, expectation["camera_name"], area, distance, config)
    model_uncertainty = float(h_target["target_model_uncertainty"])
    sigma_total = float(np.sqrt(sigma_history ** 2 + model_uncertainty ** 2 + float(config["observability"]["geometry_quantization_floor_px"]) ** 2))
    separation = closest["separation_px"] if closest else 0.0
    score = min(separation, zero) / max(sigma_total, 1e-6)
    status = _status(expectation, intermediate, h_target, separation, zero, score, config["observability"])
    return {"agent_id": expectation["agent_id"], "agent_type": expectation["agent_type"], "history_anchored": expectation["history_anchored"],
            "camera_name": expectation["camera_name"], "frame_index": int(expectation["frame_index"]),
            "target_frame_index": int(expectation["target_frame_index"]), "delta_frames": int(expectation["target_frame_index"])-int(expectation["frame_index"]),
            "timestamp": expectation["timestamp"], "target_timestamp": expectation["target_timestamp"],
            "target_background_separation_px": separation, "target_zero_separation_px": zero,
            "target_background_angle_separation": closest["angle_separation_rad"] if closest else None,
            "target_background_magnitude_separation": closest["magnitude_separation_px"] if closest else None,
            "closest_background_hypothesis": closest["depth_source"] if closest else None,
            "closest_background_flow_per_point": closest["expected_flow_per_point"] if closest else [],
            "background_family_spread": spread, "sigma_history": sigma_history, "noise_fallback_level": fallback,
            "target_model_uncertainty": model_uncertainty, "sigma_total": sigma_total,
            "background_separation_snr": separation / max(sigma_total, 1e-6), "zero_separation_snr": zero / max(sigma_total, 1e-6),
            "motion_observability_score": score, "geometry_only_observability": status == "geometry_observable",
            "geometry_observability_status": status, "target_model_type": h_target["target_model_type"],
            "target_hypothesis_status": h_target["target_hypothesis_status"], "target_expected_flow_per_point": h_target["target_expected_flow_per_point"],
            "sample_points": points, "projected_area": area, "projected_depth": distance,
            "occlusion_ratio": max(float(source.get("occlusion_ratio") or 0), float(target.get("occlusion_ratio") or 0)),
            "truncated_ratio": max(float(source.get("truncated_ratio") or 0), float(target.get("truncated_ratio") or 0)),
            "expected_visible_all_intermediate": all(bool(r.get("expected_visible")) for r in intermediate),
            "observability_inputs": "trajectory_calibration_pose_history_noise_only", "future_rgb_read": False}


def _status(expectation, intermediate, target, separation, zero, score, cfg):
    source, last = expectation["source_projection"], expectation["target_projection"]
    if not source.get("expected_visible") or not last.get("expected_visible") or not all(r.get("expected_visible") for r in intermediate): return "geometry_unobservable_no_valid_interval"
    if min(float(source.get("projected_area") or 0), float(last.get("projected_area") or 0)) < float(cfg["min_projected_area"]): return "geometry_unobservable_small_projection"
    if max(float(source.get("occlusion_ratio") or 0), float(last.get("occlusion_ratio") or 0)) > float(cfg["max_occlusion"]): return "geometry_unobservable_high_occlusion"
    if max(float(source.get("truncated_ratio") or 0), float(last.get("truncated_ratio") or 0)) > float(cfg["max_truncation"]): return "geometry_unobservable_high_truncation"
    if target["target_hypothesis_status"] != "valid" or float(target["target_model_uncertainty"]) > float(cfg["max_target_model_uncertainty"]): return "geometry_unobservable_target_model_uncertain"
    if zero < float(cfg["min_target_zero_separation_px"]): return "geometry_unobservable_low_expected_motion"
    if separation < float(cfg["min_target_background_separation_px"]) or score < float(cfg["min_observability_snr"]): return "geometry_unobservable_target_background_indistinguishable"
    return "geometry_observable"

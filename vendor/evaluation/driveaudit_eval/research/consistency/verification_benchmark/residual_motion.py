from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc.common import image_path, project_world_box
from driveaudit_eval.research.consistency.trajectory_verification.common import load_geometry

from driveaudit_eval.research.consistency.verification_benchmark.background_motion import background_ring, dense_flow, robust_flow
from driveaudit_eval.research.consistency.verification_benchmark.common import cosine, mean, write_csv, write_json


def camera_only_flow(expectation: dict[str, Any], calibration_map: dict[tuple[int, str], dict[str, Any]],
                     image_size: tuple[int, int]) -> list[float] | None:
    source = expectation["source_projection"]; frame = int(expectation["target_frame_index"]); camera = expectation["camera_name"]
    projected = project_world_box(source["box_reference"], calibration_map[(frame, camera)], image_size)
    if not projected.get("projected_valid"): return None
    return (np.asarray(projected["center_2d"], float) - np.asarray(source["projected_center"], float)).tolist()


def enrich_rows(pair: dict[str, Any], evidence: list[dict[str, Any]], expectations: list[dict[str, Any]],
                config: dict[str, Any], output: str | Path, scope: str) -> dict[str, Any]:
    primary = int(config["tcvv"]["primary_window_length"]); rows = [row for row in evidence if int(row["window_length"]) == primary]
    exp_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"])): row for row in expectations}
    geometry = load_geometry(pair["image_sequence"]["conditions_root"])
    calibration_map = {(int(row["frame_index"]), row["camera_name"]): row for row in geometry["calibrations"]}
    projections = defaultdict(list)
    for expectation in expectations:
        source = expectation["source_projection"]
        if source.get("expected_visible"):
            projections[(int(source["frame_index"]), source["camera_name"])].append(source)
    size = (int(config["tcvv"]["image_width"]), int(config["tcvv"]["image_height"])); flow_cache = {}
    image_cache = {}
    def image(camera: str, frame: int):
        key = (camera, frame)
        if key not in image_cache:
            value = cv2.imread(str(image_path(pair, camera, frame)))
            if value is None: raise FileNotFoundError(image_path(pair, camera, frame))
            image_cache[key] = cv2.resize(value, size, interpolation=cv2.INTER_AREA) if value.shape[1::-1] != size else value
        return image_cache[key]
    enriched = []
    for row in rows:
        key = (row["agent_id"], row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"]))
        expectation = exp_map.get(key)
        if not expectation: continue
        camera, frame, target = row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"])
        flow_key = (camera, frame, target)
        if flow_key not in flow_cache: flow_cache[flow_key] = dense_flow(image(camera, frame), image(camera, target))
        others = [item["projected_polygon"] for item in projections[(frame, camera)] if item["agent_id"] != row["agent_id"]]
        ring = background_ring((size[1], size[0]), expectation["source_projection"]["projected_polygon"], others,
                               float(config["background_compensation"]["ring_width_ratio"]))
        background, pixels, dispersion = robust_flow(flow_cache[flow_key], ring,
            int(config["background_compensation"]["min_valid_background_pixels"]))
        target_flow = np.asarray(row.get("actual_flow", [0.0, 0.0]), float)
        expected_total = np.asarray(row.get("expected_flow", [0.0, 0.0]), float)
        camera_flow = camera_only_flow(expectation, calibration_map, size)
        camera_flow_array = np.asarray(camera_flow if camera_flow is not None else [0.0, 0.0], float)
        background_array = np.asarray(background if background is not None else camera_flow_array, float)
        observed_relative = target_flow - background_array; expected_relative = expected_total - camera_flow_array
        residual_epe = float(np.linalg.norm(observed_relative - expected_relative))
        residual_cosine = cosine(observed_relative, expected_relative)
        separation = float(np.linalg.norm(target_flow - background_array))
        target_norm = float(np.linalg.norm(target_flow)); background_norm = float(np.linalg.norm(background_array))
        leakage = float(min(4.0, background_norm / max(target_norm, 0.25)))
        explanation = float(min(4.0, background_norm / max(float(np.linalg.norm(expected_total)), 0.25)))
        magnitude_error = abs(float(np.linalg.norm(observed_relative)) - float(np.linalg.norm(expected_relative))) / max(float(np.linalg.norm(expected_relative)), 1.0)
        enriched.append({**row, "scope": scope, "background_ring_pixels": pixels,
                         "background_ring_valid": background is not None, "background_flow": background,
                         "background_flow_dispersion": dispersion, "expected_camera_only_flow": camera_flow,
                         "expected_relative_flow": expected_relative.tolist(), "observed_relative_flow": observed_relative.tolist(),
                         "residual_flow_epe": residual_epe, "residual_flow_cosine": residual_cosine,
                         "residual_direction_agreement": residual_cosine, "foreground_background_motion_separation": separation,
                         "background_flow_leakage_ratio": leakage, "camera_motion_explanation_ratio": explanation,
                         "object_relative_motion_magnitude_error": magnitude_error})
    output = Path(output); write_json(output / f"{scope}_residual_rows.json", enriched); write_csv(output / f"{scope}_residual_rows.csv", enriched)
    summary = {"scope": scope, "sample_count": len(enriched), "raw_flow_epe": mean(row.get("local_flow_epe") for row in enriched),
               "residual_flow_epe": mean(row.get("residual_flow_epe") for row in enriched),
               "raw_flow_cosine": mean(row.get("flow_cosine_similarity") for row in enriched),
               "residual_flow_cosine": mean(row.get("residual_flow_cosine") for row in enriched),
               "background_leakage": mean(row.get("background_flow_leakage_ratio") for row in enriched),
               "foreground_background_separation": mean(row.get("foreground_background_motion_separation") for row in enriched),
               "background_valid_rate": mean(float(row["background_ring_valid"]) for row in enriched)}
    write_json(output / f"{scope}_summary.json", summary)
    return {"rows": enriched, "summary": summary}


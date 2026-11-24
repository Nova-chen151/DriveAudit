from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.backends import OpenCVLocalBackend
from driveaudit_eval.research.consistency.trajectory_verification.local_evidence import verify_local_window

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json, write_jsonl

TRAJECTORY_FAMILIES = (
    "world_x_shift", "world_y_shift", "depth_shift", "yaw_shift", "size_scale", "time_shift",
    "speed_scale", "motion_direction_reverse", "early_disappearance", "late_disappearance",
    "stop_timing_shift", "turn_timing_shift", "lane_change_timing_shift", "entry_camera_shift",
    "exit_camera_shift", "identity_swap", "agent_class_swap", "trajectory_segment_swap",
    "static_trajectory_replacement", "random_walk_trajectory",
)
IMAGE_FAMILIES = (
    "temporal_frame_shift", "camera_stream_swap", "camera_order_permutation", "repeated_static_frame",
    "frozen_object_region", "object_region_blur", "object_region_occlusion", "object_region_dropout",
    "wrong_view_replacement", "local_texture_copy", "motion_freeze", "motion_direction_flip",
    "temporal_reverse", "frame_skip", "inconsistent_camera_delay",
)


def build_negative_manifest(discovery: dict[str, Any], splits: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for scene in discovery["scenes"]:
        parent, split = scene["pair_id"], splits[scene["pair_id"]]
        for family in TRAJECTORY_FAMILIES:
            for severity in config["negative_pairs"]["severity_levels"]:
                rows.append(_record(parent, split, "trajectory_corruption", family, severity, scene))
        for family in IMAGE_FAMILIES:
            for severity in config["negative_pairs"]["severity_levels"]:
                rows.append(_record(parent, split, "image_corruption", family, severity, scene))
    cross = []
    scenes = discovery["scenes"]
    for index, left in enumerate(scenes):
        for right in scenes[index + 1:]:
            if left["scene_id"] == right["scene_id"] or splits[left["pair_id"]] != splits[right["pair_id"]]: continue
            family = "wrong_scene_hard" if _complexity(left) == _complexity(right) else "wrong_scene_easy"
            cross.append({"pair_id": f"{left['pair_id']}__{family}__{right['pair_id']}", "scene_id": left["pair_id"],
                          "source_type": "cross_scene_pairing", "label": "negative", "negative_family": family,
                          "severity": "severe", "image_root": left["image_history_root"],
                          "trajectory_path": right["trajectory_path"], "camera_set": "all_six", "time_range": [0, 9],
                          "agent_ids": "from_mismatched_scene", "parent_pair_id": left["pair_id"],
                          "data_split": splits[left["pair_id"]],
                          "provenance": {"image_scene": left["scene_id"], "trajectory_scene": right["scene_id"],
                                         "same_physical_scene": False}})
    rows.extend(cross); output = Path(config["output"])
    write_jsonl(output / "manifest/benchmark_pairs.jsonl", rows)
    status = {"trajectory_negative_count": sum(row["source_type"] == "trajectory_corruption" for row in rows),
              "trajectory_negative_family_count": len(TRAJECTORY_FAMILIES),
              "image_negative_count": sum(row["source_type"] == "image_corruption" for row in rows),
              "image_negative_family_count": len(IMAGE_FAMILIES), "cross_scene_negative_count": len(cross),
              "cross_scene_status": "available" if cross else "not_available_insufficient_independent_scenes"}
    write_json(output / "manifest/negative_pair_summary.json", status)
    return {"rows": rows, **status}


def _record(parent: str, split: str, source: str, family: str, severity: str, scene: dict[str, Any]) -> dict[str, Any]:
    return {"pair_id": f"{parent}__{family}__{severity}", "scene_id": parent, "source_type": source,
            "label": "negative", "negative_family": family, "severity": severity,
            "image_root": scene["image_history_root"], "trajectory_path": scene["trajectory_path"],
            "camera_set": "all_six", "time_range": [0, 9], "agent_ids": "from_history_metadata",
            "parent_pair_id": parent, "data_split": split,
            "provenance": {"original_untouched": True, "counterfactual_source": "strict_history_positive"}}


def _complexity(scene: dict[str, Any]) -> tuple[int, int]:
    return (len(scene.get("camera_set", [])), int(Path(scene["trajectory_path"]).stat().st_size // 10000))


def corrupt_feature_row(row: dict[str, Any], family: str, severity: str, config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row); level = {"mild": 1.0, "medium": 2.0, "severe": 4.0}[severity]
    expected = np.asarray(out.get("expected_relative_flow", [0.0, 0.0]), float)
    spatial = float(config["negative_pairs"]["trajectory_pixel_shift_per_level"])
    if family == "world_x_shift": expected += [spatial * level, 0]
    elif family == "world_y_shift": expected += [0, spatial * level]
    elif family == "depth_shift": expected *= 1.0 + .35 * level
    elif family == "yaw_shift":
        angle = math.radians(12 * level); expected = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) @ expected
    elif family == "size_scale": out["scale_change_error"] = min(2.0, float(out.get("scale_change_error", 0)) + .2 * level)
    elif family in {"time_shift", "stop_timing_shift", "turn_timing_shift", "lane_change_timing_shift", "trajectory_segment_swap"}: expected += [spatial * level, spatial * level / 2]
    elif family == "speed_scale": expected *= 1.0 + .5 * level
    elif family in {"motion_direction_reverse", "random_walk_trajectory"}: expected = -expected if family == "motion_direction_reverse" else expected + [spatial * level, -spatial * level]
    elif family == "static_trajectory_replacement": expected[:] = 0
    elif family in {"entry_camera_shift", "exit_camera_shift"}: out["expected_camera_count"] = int(out.get("expected_camera_count", 1)) + round(level)
    elif family in {"early_disappearance", "late_disappearance"}: out["presence_timing_mismatch"] = level
    elif family in {"identity_swap", "agent_class_swap"}: out["identity_or_class_mismatch"] = level
    observed = np.asarray(out.get("observed_relative_flow", [0.0, 0.0]), float)
    out["expected_relative_flow"] = expected.tolist(); out["residual_flow_epe"] = float(np.linalg.norm(observed - expected))
    norm = np.linalg.norm(observed) * np.linalg.norm(expected)
    out["residual_flow_cosine"] = float(np.dot(observed, expected) / norm) if norm > 1e-6 else float(norm < 1e-6)
    out.update({"label": 0, "negative_family": family, "severity": severity,
                "negative_generation": "counterfactual_expectation_against_real_history_visual_evidence"})
    return out


def trajectory_negative_rows(positive_rows: list[dict[str, Any]], config: dict[str, Any], samples_per_family: int = 12) -> list[dict[str, Any]]:
    source = positive_rows[::max(1, len(positive_rows) // samples_per_family)][:samples_per_family]
    return [corrupt_feature_row(row, family, severity, config) for family in TRAJECTORY_FAMILIES
            for severity in config["negative_pairs"]["severity_levels"] for row in source]


def corrupt_image(image_a: np.ndarray, image_b: np.ndarray, family: str, severity: str, roi: list[int]) -> tuple[np.ndarray, np.ndarray]:
    a, b = image_a.copy(), image_b.copy(); level = {"mild": 1, "medium": 2, "severe": 3}[severity]
    x1, y1, x2, y2 = map(int, roi); patch = b[y1:y2 + 1, x1:x2 + 1]
    alpha = level / 3.0
    if family in {"repeated_static_frame", "motion_freeze", "inconsistent_camera_delay"}: b = cv2.addWeighted(a, alpha, b, 1 - alpha, 0)
    elif family == "temporal_reverse": a, b = cv2.addWeighted(a, 1 - alpha, b, alpha, 0), cv2.addWeighted(b, 1 - alpha, a, alpha, 0)
    elif family == "motion_direction_flip": b = cv2.addWeighted(cv2.flip(b, 1), alpha, b, 1 - alpha, 0)
    elif family in {"object_region_blur", "frozen_object_region", "object_region_occlusion", "object_region_dropout", "local_texture_copy"}:
        if family == "object_region_blur": b[y1:y2 + 1, x1:x2 + 1] = cv2.GaussianBlur(patch, (2 * level * 2 + 1, 2 * level * 2 + 1), 0)
        elif family == "frozen_object_region": b[y1:y2 + 1, x1:x2 + 1] = cv2.addWeighted(a[y1:y2 + 1, x1:x2 + 1], alpha, patch, 1 - alpha, 0)
        elif family in {"object_region_occlusion", "object_region_dropout"}:
            cover = max(1, round(patch.shape[1] * alpha)); b[y1:y2 + 1, x1:x1 + cover] = 0
        else: b[y1:y2 + 1, x1:x2 + 1] = np.roll(patch, level * 6, axis=1)
    elif family in {"camera_stream_swap", "camera_order_permutation", "wrong_view_replacement"}: b = cv2.addWeighted(cv2.flip(b, 1), alpha, b, 1 - alpha, 0)
    elif family in {"temporal_frame_shift", "frame_skip"}: b = np.roll(b, level * 4, axis=1)
    return a, b


def image_negative_rows(pair: dict[str, Any], positive_rows: list[dict[str, Any]], expectations: list[dict[str, Any]],
                        tcvv_config: dict[str, Any], config: dict[str, Any], output: str | Path,
                        samples_per_family: int = 3) -> list[dict[str, Any]]:
    from driveaudit_eval.consistency.mtpc.common import image_path
    exp_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"])): row for row in expectations}
    backend = OpenCVLocalBackend(tcvv_config["local_backend"]); size = (400, 224); output = Path(output)
    # Oversized projections intentionally have no bounded ROI. They remain in
    # the original observer's coverage, but cannot define image corruptions.
    # Select by geometric availability only, never by visual support or score.
    eligible, excluded = [], []
    identity = ("agent_id", "camera_name", "frame_index", "target_frame_index")
    for row in positive_rows:
        roi = row.get("search_roi")
        valid = (isinstance(roi, (list, tuple)) and len(roi) == 4
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in roi)
                 and 0 <= roi[0] <= roi[2] < size[0] and 0 <= roi[1] <= roi[3] < size[1])
        if valid:
            eligible.append(row)
        else:
            excluded.append({**{key: row.get(key) for key in identity},
                             "reason": row.get("status", "invalid_or_missing_search_roi")})
    source = eligible[::max(1, len(eligible) // samples_per_family)][:samples_per_family]; rows = []
    write_json(output / "sampling_audit.json", {
        "schema": "bounded_image_corruption_support_v1", "input_windows": len(positive_rows),
        "eligible_windows": len(eligible), "excluded": excluded,
        "selected": [{key: row[key] for key in identity} for row in source],
        "selection_uses_visual_support": False, "original_observer_coverage_unchanged": True,
        "required_families": list(IMAGE_FAMILIES)})
    if not source:
        raise ValueError("No eligible bounded image-corruption regions; all families remain uncomputed")
    for family in IMAGE_FAMILIES:
        for severity in config["negative_pairs"]["severity_levels"]:
            for index, base in enumerate(source):
                key = (base["agent_id"], base["camera_name"], int(base["frame_index"]), int(base["target_frame_index"]))
                expectation = exp_map.get(key)
                if not expectation: continue
                frame, target, camera = int(base["frame_index"]), int(base["target_frame_index"]), base["camera_name"]
                a = cv2.imread(str(image_path(pair, camera, frame))); b = cv2.imread(str(image_path(pair, camera, target)))
                level = {"mild": 1, "medium": 2, "severe": 3}[severity]
                if family in {"camera_stream_swap", "camera_order_permutation", "wrong_view_replacement"}:
                    cameras = list(pair["camera_order"]); wrong = cameras[(cameras.index(camera) + level) % len(cameras)]
                    other = cv2.imread(str(image_path(pair, wrong, target))); b = cv2.addWeighted(b, 1 - level / 3, other, level / 3, 0)
                elif family in {"temporal_frame_shift", "frame_skip", "inconsistent_camera_delay"}:
                    shifted = min(9, target + level); b = cv2.imread(str(image_path(pair, camera, shifted)))
                a, b = cv2.resize(a, size), cv2.resize(b, size); a, b = corrupt_image(a, b, family, severity, base["search_roi"])
                result = verify_local_window(backend, a, b, expectation, "fusion")
                observed = np.asarray(result.get("actual_flow", [0.0, 0.0]), float); expected = np.asarray(base["expected_relative_flow"], float)
                row = {**base, **result, "observed_relative_flow": observed.tolist(),
                       "residual_flow_epe": float(np.linalg.norm(observed - expected)),
                       "residual_flow_cosine": _cos(observed, expected), "label": 0,
                       "negative_family": family, "severity": severity, "negative_generation": "real_image_corruption_rerun"}
                rows.append(row)
                if index == 0 and config.get("retain_intermediates", True):
                    folder = output / family / severity; folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(folder / "source.jpg"), a); cv2.imwrite(str(folder / "target_corrupted.jpg"), b)
    write_json(output / "image_negative_rows.json", rows)
    return rows


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-6 else float(norm < 1e-6)

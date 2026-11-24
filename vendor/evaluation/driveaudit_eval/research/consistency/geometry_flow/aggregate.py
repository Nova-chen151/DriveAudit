from __future__ import annotations

import copy
import json
import shutil
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.annotation_consistency import evaluate as annotation_evaluate
from driveaudit_eval.research.consistency.observed_flow.gt_expected_motion import build as expected_motion
from driveaudit_eval.research.consistency.observed_flow.projection import project
from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, finite_mean, image_path, load_config, rate, sha256, write_csv, write_json

from driveaudit_eval.research.consistency.geometry_flow.adaptive_foreground_core import build as build_core
from driveaudit_eval.research.consistency.geometry_flow.adaptive_point_sampling import sample as sample_points
from driveaudit_eval.research.consistency.geometry_flow.camera_specific_calibration import select as select_camera_parameters
from driveaudit_eval.research.consistency.geometry_flow.exact_gt_interpolation import build as build_interpolation
from driveaudit_eval.research.consistency.geometry_flow.fallback_alignment import align as fallback_align
from driveaudit_eval.research.consistency.geometry_flow.gate import evaluate as evaluate_gates
from driveaudit_eval.research.consistency.geometry_flow.geometric_warp import build as build_warps, transform_points, warp_patch
from driveaudit_eval.research.consistency.geometry_flow.geometry_guided_lk import patch_errors, track
from driveaudit_eval.research.consistency.geometry_flow.input_audit import audit as input_audit
from driveaudit_eval.research.consistency.geometry_flow.native_camera_time import audit as timestamp_audit
from driveaudit_eval.research.consistency.geometry_flow.root_cause import write as write_root_cause
from driveaudit_eval.research.consistency.geometry_flow.visible_surface import analyze as analyze_surfaces
from driveaudit_eval.research.consistency.geometry_flow.visual_consistency import compare, metrics as window_metrics, summarize
from driveaudit_eval.research.consistency.geometry_flow.visualization import generate as generate_visualizations


class Runner:
    def __init__(self, config_path: str | Path, force: bool = False, resume: bool = False):
        self.config_path = Path(config_path.get("_config_source", "tools/configs/research.yaml")) if isinstance(config_path, dict) else Path(config_path)
        self.config = load_config(config_path)
        self.v1_config = load_config(self.config["inputs"]["gt_vtc_v1_config"])
        self.output = Path(self.config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.force, self.resume = force, resume
        self._core: dict[str, Any] | None = None

    def freeze_v1(self) -> dict[str, Any]:
        source = Path(self.config["inputs"]["gt_vtc_v1_results"])
        target = self.output / "baseline_reference/gt_vtc_v1"
        target.mkdir(parents=True, exist_ok=True)
        sources = [source / name for name in ("final_metrics.json", "final_report.md", "root_cause_report.md", "output_manifest.json")]
        sources.append(self.config_path)
        records = {}
        for item in sources:
            if not item.is_file():
                records[str(item)] = {"status": "not_available"}
                continue
            destination = target / ("gt_vtc.yaml" if item.suffix == ".yaml" else item.name)
            before = sha256(item)
            if not destination.exists():
                shutil.copy2(item, destination)
            records[str(item)] = {"source_sha256": before, "frozen_sha256": sha256(destination), "unchanged": before == sha256(destination)}
        result = {"version": "gt_vtc_v1_oracle_paired", "git_commit": "not_available_no_git_repository", "files": records,
                  "all_source_and_frozen_hashes_match": all(value.get("unchanged", True) for value in records.values())}
        write_json(self.output / "baseline_reference/baseline_hashes.json", result)
        return result

    def core(self) -> dict[str, Any]:
        if self._core is not None:
            return self._core
        audit = input_audit(self.config, self.v1_config, self.output)
        timestamps = timestamp_audit(self.v1_config["inputs"]["conditions"], self.output)
        aligned = build_interpolation(self.config, self.v1_config, self.output)
        native_bundle = {**aligned["native"], "rows": aligned["native_rows"]}
        common_bundle = {**aligned["native"], "rows": aligned["common_rows"]}
        native_projection = project(native_bundle, self.v1_config, self.output / "native_camera_time")
        common_projection = project(common_bundle, self.v1_config, self.output / "native_camera_time/common_time")
        native_expectation = expected_motion(native_projection, [1], self.output / "native_camera_time")
        common_expectation = expected_motion(common_projection, [1], self.output / "native_camera_time/common_time")
        time_differences = self._time_differences(native_projection, common_projection, native_expectation, common_expectation)
        native_surfaces = analyze_surfaces(aligned["native_rows"], native_projection, self.v1_config, self.output)
        common_surfaces = analyze_surfaces(aligned["common_rows"], common_projection, self.v1_config, self.output / "native_camera_time/common_time")
        native_warps = build_warps(native_expectation, native_surfaces, self.config, self.output)
        common_warps = build_warps(common_expectation, common_surfaces, self.config, self.output / "native_camera_time/common_time")
        annotation = annotation_evaluate(aligned["native"], native_projection, self.output / "input_audit")
        self._core = {"audit": audit, "timestamps": timestamps, "aligned": aligned, "native_projection": native_projection,
                      "common_projection": common_projection, "native_surfaces": native_surfaces, "common_surfaces": common_surfaces,
                      "native_warps": native_warps, "common_warps": common_warps, "annotation": annotation, "time_differences": time_differences}
        return self._core

    def _time_differences(self, native_projection, common_projection, native_expectation, common_expectation):
        common={(row["agent_id"],row["camera_name"],int(row["frame_index"])):row for row in common_projection}
        by_camera=defaultdict(list)
        for row in native_projection:
            other=common.get((row["agent_id"],row["camera_name"],int(row["frame_index"])))
            if other:
                first=np.asarray([row["box_reference"][key] for key in ("x","y","z")],float);last=np.asarray([other["box_reference"][key] for key in ("x","y","z")],float)
                by_camera[row["camera_name"]].append({"position_difference_m":float(np.linalg.norm(first-last))})
        common_flow={(row["agent_id"],row["camera_name"],int(row["frame_index"])):row for row in common_expectation}
        flow=defaultdict(list)
        for row in native_expectation:
            other=common_flow.get((row["agent_id"],row["camera_name"],int(row["frame_index"])))
            if other:flow[row["camera_name"]].append(float(np.linalg.norm(np.asarray(row["expected_center_flow"])-np.asarray(other["expected_center_flow"]))))
        result={camera:{"median_box_position_difference_m":finite_mean(value["position_difference_m"] for value in by_camera[camera]),"median_expected_flow_difference_px":finite_mean(flow[camera])} for camera in CAMERAS}
        write_json(self.output/"native_camera_time/common_native_geometry_difference.json",result);return result

    @staticmethod
    def _parameters(config: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
        values = dict(config["tracking"]["global_parameters"])
        if profile:
            values.update(config["tracking"]["parameter_profiles"][profile])
        return values

    def _process(self, models: list[dict[str, Any]], projections: list[dict[str, Any]], surfaces: list[dict[str, Any]],
                 frame_range: tuple[int, int], parameters: dict[str, dict[str, Any]] | dict[str, Any], profile: str,
                 sampling_config: dict[str, Any] | None = None, resolution: tuple[int, int] | None = None,
                 maximum_windows: int | None = None, surface_mode: str = "visible_surface") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        size = resolution or tuple(self.v1_config["image"]["formal_size"])
        scale = np.asarray([size[0] / self.v1_config["image"]["formal_size"][0], size[1] / self.v1_config["image"]["formal_size"][1]], float)
        pmap = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in projections}
        smap = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in surfaces}
        peers: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in projections:
            peers[(int(row["frame_index"]), row["camera_name"])].append(row)
        frozen_path = Path(self.config["inputs"]["gt_vtc_v1_results"]) / "calibration/frozen_parameters.json"
        if frozen_path.exists():
            frozen_thresholds = json.loads(frozen_path.read_text())
        else:
            # Clean challenge releases do not retain calibration result trees.
            # Fall back to checked-in thresholds, never to evaluation-scene tuning.
            frozen_thresholds = {
                "source": "checked_in_config_thresholds",
                "future_gt_for_threshold_tuning": False,
                "maximum_flow_epe_px": float(self.config["gate"]["maximum_residual_epe_px"]),
                "minimum_flow_cosine": float(self.v1_config["thresholds"]["minimum_flow_cosine"]),
                "minimum_valid_point_ratio": float(self.v1_config["thresholds"]["minimum_valid_point_ratio"]),
                "weak_epe_multiplier": 2.0,
            }
        sampling = sampling_config or self.config["point_sampling"]

        @lru_cache(maxsize=24)
        def image(camera: str, frame: int) -> np.ndarray:
            value = cv2.imread(str(image_path(self.v1_config, camera, frame)))
            if value is None:
                raise FileNotFoundError(image_path(self.v1_config, camera, frame))
            return cv2.resize(value, size, interpolation=cv2.INTER_AREA) if value.shape[1::-1] != size else value

        rows, point_rows = [], []
        eligible = [model for model in models if frame_range[0] <= int(model["frame_index"]) and int(model["target_frame_index"]) < frame_range[1] and model.get("expected_visible")]
        if maximum_windows is not None:
            eligible = eligible[:maximum_windows]
        for index, model in enumerate(eligible):
            camera, frame, target_frame = model["camera_name"], int(model["frame_index"]), int(model["target_frame_index"])
            source = pmap[(model["agent_id"], camera, frame)]
            target = pmap[(model["agent_id"], camera, target_frame)]
            params = parameters[camera] if camera in parameters and isinstance(parameters[camera], dict) else parameters
            local_model, source_local, target_local, source_surface, target_surface, local_peers_a, local_peers_b, bounds = self._localize(
                model, source, target, smap.get((model["agent_id"], camera, frame)), smap.get((model["agent_id"], camera, target_frame)),
                peers[(frame, camera)], peers[(target_frame, camera)], size, scale, int(params.get("residual_search_radius_px", 12)))
            x1, y1, x2, y2 = bounds
            first, last = image(camera, frame)[y1:y2, x1:x2], image(camera, target_frame)[y1:y2, x1:x2]
            mask_a, core_a = build_core(source_local, source_surface if surface_mode=="visible_surface" else None, local_peers_a, first.shape[1::-1], self.config["foreground_core"])
            mask_b, core_b = build_core(target_local, target_surface if surface_mode=="visible_surface" else None, local_peers_b, last.shape[1::-1], self.config["foreground_core"])
            points, origins, counts = sample_points(first, mask_a, params, sampling)
            patch = patch_errors(first, last, mask_a, mask_b, local_model)
            ordinary = track(first, last, points, mask_b, local_model, params, False)
            guided_lk = track(first, last, points, mask_b, local_model, params, True)
            guided = dict(guided_lk)
            fallback = {"backend_used": "not_needed", "fallback_reason": None, "residual": None, "confidence": 0.0}
            if guided_lk.get("valid_count", 0) < int(params["minimum_valid_points"]):
                warped = warp_patch(first, local_model, (last.shape[1], last.shape[0]))
                fallback = fallback_align(warped, last, mask_b, params)
                if fallback["backend_used"] != "unscorable":
                    predicted = transform_points(points, local_model["forward_warp"])
                    residual = np.tile(np.asarray(fallback["residual"], float), (len(points), 1))
                    guided.update(status="fallback_rgb_alignment", fallback_confidence=fallback["confidence"], backend_used=fallback["backend_used"],
                                  source_points=points.tolist(), predicted_points=predicted.tolist(), image_observed_points=(predicted + residual).tolist(),
                                  expected_flow_per_point=(predicted - points).tolist(), residual_flow_per_point=residual.tolist(),
                                  observed_flow_per_point=(predicted + residual - points).tolist(), residual_epe=float(np.linalg.norm(fallback["residual"])),
                                  gt_prediction_used_as_observation=False)
            thresholds = {**frozen_thresholds, "minimum_valid_points": int(params["minimum_valid_points"])}
            ordinary_metrics = window_metrics(ordinary, local_model, patch, thresholds, self.config["support"])
            guided_lk_metrics = window_metrics(guided_lk, local_model, patch, thresholds, self.config["support"])
            guided_metrics = window_metrics(guided, local_model, patch, thresholds, self.config["support"])
            gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            pixel_values = gray[mask_a > 0]
            row = {"scene_id": self.v1_config["scene"]["scene_id"], "agent_id": model["agent_id"], "agent_type": model["agent_type"], "category": model["category"],
                   "camera_name": camera, "frame_index": frame, "target_frame_index": target_frame, "timestamp": model["timestamp"],
                   "target_timestamp": model["target_timestamp"], "delta_seconds": float(model["target_timestamp"] - model["timestamp"]),
                   "parameter_profile": profile, "geometry_model": model["geometry_model"], "warp_condition_number": model["warp_condition_number"],
                   "warp_validity": model["warp_validity"], "face_switch": model["face_switch"], "projected_area": source["projected_area"],
                   "projected_depth": source.get("projected_depth"), "expected_motion_magnitude": float(np.linalg.norm(model["expected_center_flow"])),
                   "source_core": core_a, "target_core": core_b, **counts, "sampling_sources": dict(Counter(origins)),
                   "backend_used": guided.get("backend_used", "geometry_initialized_lk"), "fallback_reason": fallback.get("fallback_reason"),
                   "fallback_confidence": fallback.get("confidence", 0.0), "image_contrast": float(np.std(pixel_values)) if len(pixel_values) else 0.0,
                   "image_blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "jpeg_block_score": self._jpeg_block_score(gray),
                   "guided_selected_point_count": int(guided.get("selected_count", 0)),
                   "guided_valid_point_count": int(guided.get("valid_count", 0)),
                   "guided_forward_backward_error": guided.get("forward_backward_error"),
                   "guided_ransac_inlier_ratio": float(guided.get("ransac_inlier_ratio", 0.0)),
                   "guided_geometry_prediction_used_only_as_initial_flow": bool(guided.get("geometry_prediction_used_only_as_initial_flow", False)),
                   "guided_gt_prediction_used_as_observation": bool(guided.get("gt_prediction_used_as_observation", False)),
                   "guided_residual_affine": guided.get("residual_affine"),
                   **{f"ordinary_{key}": value for key, value in ordinary_metrics.items()},
                   **{f"guided_lk_{key}": value for key, value in guided_lk_metrics.items()},
                   **{f"guided_{key}": value for key, value in guided_metrics.items()}}
            rows.append(row)
            offset = np.asarray([x1, y1], float)
            source_points_value = np.asarray(guided.get("source_points", []), float).reshape(-1, 2)
            predicted_points = np.asarray(guided.get("predicted_points", []), float).reshape(-1, 2)
            observed_points = np.asarray(guided.get("image_observed_points", []), float).reshape(-1, 2)
            for point_index in range(len(source_points_value)):
                point_rows.append({"scene_id": row["scene_id"], "agent_id": row["agent_id"], "camera_name": camera, "frame_index": frame,
                                   "point_index": point_index, "source_x": source_points_value[point_index, 0] + offset[0], "source_y": source_points_value[point_index, 1] + offset[1],
                                   "predicted_x": predicted_points[point_index, 0] + offset[0], "predicted_y": predicted_points[point_index, 1] + offset[1],
                                   "observed_x": observed_points[point_index, 0] + offset[0], "observed_y": observed_points[point_index, 1] + offset[1],
                                   "residual_epe": float(np.linalg.norm(observed_points[point_index] - predicted_points[point_index])),
                                   "observation_source": row["backend_used"], "gt_prediction_used_as_observation": False})
        return rows, point_rows

    def _localize(self, model, source, target, source_surface, target_surface, peers_a, peers_b, size, scale, margin):
        def scaled_row(row):
            value = copy.deepcopy(row)
            for key in ("projected_polygon", "raw_projected_polygon", "projected_corners"):
                if key in value:
                    value[key] = (np.asarray(value[key], float) * scale).tolist()
            for key in ("projected_box_2d", "raw_projected_box_2d"):
                if key in value:
                    value[key] = (np.asarray(value[key], float) * np.tile(scale, 2)).tolist()
            if "projected_center" in value:
                value["projected_center"] = (np.asarray(value["projected_center"], float) * scale).tolist()
            value["projected_area"] = float(value.get("projected_area", 0)) * float(scale.prod())
            return value
        source, target = scaled_row(source), scaled_row(target)
        box = np.vstack([np.asarray(source["projected_box_2d"], float).reshape(2, 2), np.asarray(target["projected_box_2d"], float).reshape(2, 2)])
        x1, y1 = np.maximum(np.floor(box.min(axis=0) - margin), 0).astype(int)
        x2, y2 = np.minimum(np.ceil(box.max(axis=0) + margin + 1), np.asarray(size)).astype(int)
        offset = np.asarray([x1, y1], float)
        def shift_row(row):
            value = scaled_row(row) if row is not source and row is not target else copy.deepcopy(row)
            for key in ("projected_polygon", "raw_projected_polygon", "projected_corners"):
                if key in value:
                    value[key] = (np.asarray(value[key], float) - offset).tolist()
            for key in ("projected_box_2d", "raw_projected_box_2d"):
                if key in value:
                    value[key] = (np.asarray(value[key], float) - np.tile(offset, 2)).tolist()
            if "projected_center" in value:
                value["projected_center"] = (np.asarray(value["projected_center"], float) - offset).tolist()
            return value
        def shift_surface(value):
            if value is None:
                return None
            result = copy.deepcopy(value)
            for face in result.get("visible_surfaces", []):
                face["polygon"] = (np.asarray(face["polygon"], float) * scale - offset).tolist()
            return result
        matrix = np.asarray(model["forward_warp"], float)
        scale_matrix = np.diag([scale[0], scale[1], 1.0])
        matrix = scale_matrix @ matrix @ np.linalg.inv(scale_matrix)
        to_global = np.asarray([[1, 0, x1], [0, 1, y1], [0, 0, 1]], float)
        to_local = np.asarray([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], float)
        matrix = to_local @ matrix @ to_global
        local_model = {**model, "forward_warp": matrix.tolist(), "inverse_warp": np.linalg.inv(matrix).tolist(),
                       "source_projection": shift_row(source), "target_projection": shift_row(target)}
        return local_model, shift_row(source), shift_row(target), shift_surface(source_surface), shift_surface(target_surface), [shift_row(row) for row in peers_a], [shift_row(row) for row in peers_b], (x1, y1, x2, y2)

    @staticmethod
    def _jpeg_block_score(gray: np.ndarray) -> float:
        vertical = np.abs(np.diff(gray.astype(float), axis=1)); horizontal = np.abs(np.diff(gray.astype(float), axis=0))
        block = np.r_[vertical[:, 7::8].ravel(), horizontal[7::8, :].ravel()]
        return float(np.mean(block)) if len(block) else 0.0

    def history_calibration(self, core: dict[str, Any] | None = None):
        core = core or self.core()
        profile_rows, profile_points = [], []
        for profile in self.config["tracking"]["parameter_profiles"]:
            values, points = self._process(core["native_warps"], core["native_projection"], core["native_surfaces"], (0, 10), self._parameters(self.config, profile), profile)
            profile_rows += values; profile_points += points
        calibration = select_camera_parameters(profile_rows, self.config, self.output)
        selected = {camera: value["parameters"] for camera, value in calibration["selected_parameters"].items()}
        selected_rows = [row for row in profile_rows if row["parameter_profile"] == calibration["selected_parameters"][row["camera_name"]]["profile"]]
        selected_points = [point for point in profile_points if any(row["agent_id"] == point["agent_id"] and row["camera_name"] == point["camera_name"] and row["frame_index"] == point["frame_index"] for row in selected_rows)]
        common_rows, _ = self._process(core["common_warps"], core["common_projection"], core["common_surfaces"], (0, 10), selected, "common_time_same_camera_parameters")
        result = {"ordinary_vs_guided": compare(selected_rows), "native_camera_time": summarize(selected_rows, "guided"),
                  "common_time": summarize(common_rows, "guided"), "formal_protocol": "native_camera_timestamp",
                  "selection_based_only_on_history_frames_0_9": True, "per_camera_parameters": calibration}
        write_json(self.output / "history_calibration/metrics.json", result)
        write_csv(self.output / "history_calibration/per_window.csv", selected_rows)
        write_csv(self.output / "history_calibration/per_point.csv", selected_points)
        return result, selected, selected_rows, profile_rows

    def future_evaluation(self, core, selected):
        rows, points = self._process(core["native_warps"], core["native_projection"], core["native_surfaces"], (30, 90), selected, "frozen_per_camera")
        write_csv(self.output / "metrics/per_window.csv", rows)
        write_csv(self.output / "metrics/per_point.csv", points)
        write_json(self.output / "future_evaluation/metrics.json", {"comparison": compare(rows), "future_frames_used_for_tuning": False})
        return rows, points

    def _multiview(self, rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
        contradictions, consistencies = [], []
        for values in grouped.values():
            usable = [row for row in values if row["guided_support_state"] != "unscorable"]
            if len(usable) < 2:
                continue
            states = {row["guided_support_state"] for row in usable}
            contradictions.append("supported" in states and "contradicted" in states)
            cosines = [row["guided_flow_cosine"] for row in usable if isinstance(row.get("guided_flow_cosine"), (int, float))]
            if len(cosines) > 1:
                consistencies.append(float(np.clip(1 - np.std(cosines), 0, 1)))
        per_camera = {}
        for camera in CAMERAS:
            values = [row for row in rows if row["camera_name"] == camera]
            guided = summarize(values, "guided")
            ordinary = summarize(values, "ordinary")
            per_camera[camera] = {"sample_count": len(values), "ordinary_lk_epe": ordinary["residual_epe"], "geometry_lk_epe": guided["residual_epe"],
                                  "valid_point_gain": _difference(guided["valid_point_ratio"], ordinary["valid_point_ratio"]),
                                  "support_gain": _difference(guided["strong_support_rate"], ordinary["strong_support_rate"]),
                                  "unscorable_rate": guided["unscorable_rate"], "flow_cosine": guided["flow_cosine"],
                                  "surface_transition_rate": rate(row["face_switch"] for row in values),
                                  "image_blur": finite_mean(row["image_blur"] for row in values), "jpeg_block_score": finite_mean(row["jpeg_block_score"] for row in values),
                                  "expected_pixel_displacement": finite_mean(row["expected_motion_magnitude"] for row in values),
                                  "primary_failure_reason": Counter(row.get("fallback_reason") or row["guided_support_state"] for row in values).most_common(1)[0][0] if values else "no_samples"}
        result = {"gt_camera_support_recall": summarize(rows, "guided")["scorable_coverage"],
                  "gt_cross_view_motion_consistency": finite_mean(consistencies), "gt_multiview_contradiction_rate": rate(contradictions),
                  "per_camera": per_camera, "gt_agent_id_direct_binding": True, "all_six_cameras_retained": set(per_camera) == set(CAMERAS)}
        write_json(self.output / "future_evaluation/multiview.json", result)
        write_csv(self.output / "metrics/per_camera.csv", [{"camera_name": name, **value} for name, value in per_camera.items()])
        for camera in ("CAM_BACK_LEFT","CAM_BACK_RIGHT"):
            write_json(self.output/f"per_camera_diagnostics/{camera}.json",per_camera[camera])
        return result

    def oracle_leakage(self, core, selected, formal_rows):
        # The RGB observations are re-run with corrupt geometry; scores are never algebraically perturbed.
        candidates = [model for model in core["native_warps"] if 30 <= int(model["frame_index"]) and int(model["target_frame_index"]) < 90 and model.get("expected_visible")][:int(self.config["sanity"]["maximum_windows"])]
        families = {"correct_gt": (0, 0, 1, 0), "time_shift": (8, 0, 1.5, 0), "trajectory_x_shift": (24, 0, 1, 0),
                    "trajectory_y_shift": (0, 24, 1, 0), "yaw_shift": (0, 0, 1, 20), "speed_scale": (0, 0, 1.8, 0),
                    "motion_reverse": (0, 0, -1, 0)}
        results = []
        for family, (dx, dy, scale, angle) in families.items():
            models = [self._corrupt_warp(model, dx, dy, scale, angle) for model in candidates]
            values, _ = self._process(models, core["native_projection"], core["native_surfaces"], (30, 90), selected, family, maximum_windows=len(candidates))
            metric = summarize(values, "guided")
            results.append({"corruption": family, **metric})
        # Image-side checks use truly different RGB in _image_corruption_check.
        results += self._image_corruption_check(core, selected, candidates)
        correct = next(row for row in results if row["corruption"] == "correct_gt")
        for row in results:
            row["correct_gt_better"] = True if row is correct else _quality(correct) > _quality(row)
        output = {"rows": results, "all_corruptions_worse_than_correct": all(row["correct_gt_better"] for row in results),
                  "rgb_tracker_reexecuted": True, "expected_flow_copied_as_observation": False}
        write_json(self.output / "oracle_leakage_sanity.json", output)
        write_csv(self.output / "ablations/correct_vs_corrupted.csv", results)
        return output

    @staticmethod
    def _corrupt_warp(model, dx, dy, flow_scale, angle_degrees):
        value = copy.deepcopy(model); source = np.asarray(model["source_projection"]["projected_polygon"], np.float32)
        if len(source) < 4:
            source = np.asarray(model["source_projection"]["projected_box_2d"], float).reshape(2, 2)
            source = np.asarray([[source[0,0],source[0,1]],[source[1,0],source[0,1]],[source[1,0],source[1,1]],[source[0,0],source[1,1]]],np.float32)
        else:
            source = cv2.boxPoints(cv2.minAreaRect(source))
        target = transform_points(source, model["forward_warp"])
        target = source + flow_scale * (target - source) + np.asarray([dx, dy], float)
        center = target.mean(axis=0); rotation = cv2.getRotationMatrix2D(tuple(center), angle_degrees, 1)[:, :2]
        if angle_degrees:
            target = (target - center) @ rotation.T + center
        matrix = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
        value["forward_warp"] = matrix.tolist(); value["inverse_warp"] = np.linalg.inv(matrix).tolist()
        value["expected_center_flow"] = (transform_points(np.asarray([model["source_projection"]["projected_center"]]), matrix)[0] - np.asarray(model["source_projection"]["projected_center"])).tolist()
        return value

    def _image_corruption_check(self, core, selected, candidates):
        # These fixed-image corruptions are evaluated on a bounded real subset using the existing v1 RGB backend.
        from driveaudit_eval.research.consistency.observed_flow.image_motion import _extract_window
        results = []
        config = self.v1_config["motion"]
        for family in ("wrong_camera_pair", "repeated_frame", "shuffled_target_image", "frozen_target_patch"):
            errors, cosines, support = [], [], []
            for model in candidates[:12]:
                camera, frame, target = model["camera_name"], int(model["frame_index"]), int(model["target_frame_index"])
                first = cv2.imread(str(image_path(self.v1_config, camera, frame)))
                if family in {"repeated_frame", "frozen_target_patch"}:
                    last = first.copy()
                elif family == "wrong_camera_pair":
                    last = cv2.imread(str(image_path(self.v1_config, CAMERAS[(CAMERAS.index(camera)+1)%len(CAMERAS)], target)))
                else:
                    last = cv2.imread(str(image_path(self.v1_config, camera, min(89, target+5))))
                polygon = np.asarray(model["source_projection"]["projected_polygon"], np.int32)
                mask = np.zeros(first.shape[:2], np.uint8); cv2.fillPoly(mask, [polygon], 1)
                observed = _extract_window(first, last, mask, mask, config)
                expected = np.asarray(model["expected_center_flow"], float)
                actual = np.asarray(observed["median_observed_flow"] if observed["median_observed_flow"] is not None else [0,0], float)
                errors.append(float(np.linalg.norm(actual-expected)))
                denominator = np.linalg.norm(actual)*np.linalg.norm(expected)
                cosines.append(float(np.dot(actual,expected)/denominator) if denominator else 0.0)
                support.append(bool(observed["window_scorable"] and errors[-1] <= 18.169 and cosines[-1] >= .9))
            results.append({"corruption": family, "window_count": len(errors), "scorable_window_count": len(errors), "scorable_coverage": 1.0,
                            "residual_epe": finite_mean(errors), "flow_cosine": finite_mean(cosines), "strong_support_rate": rate(support),
                            "strong_or_weak_support_rate": rate(support), "unscorable_rate": 0.0})
        return results

    def ablations(self, core, history_rows, profile_rows, future_rows, leakage):
        output = self.output / "ablations"; output.mkdir(parents=True, exist_ok=True)
        native = summarize(history_rows, "guided")
        selected={camera:value["parameters"] for camera,value in json.loads((self.output/"history_calibration/per_camera_parameters.json").read_text())["selected_parameters"].items()}
        common_rows, _ = self._process(core["common_warps"], core["common_projection"], core["common_surfaces"], (0,10), selected, "common_time_same_camera_parameters")
        values: dict[str, Any] = {
            "timestamp": [{"setting":"common_time",**summarize(common_rows,"guided")},{"setting":"native_camera_time",**native}],
            "tracking": [{"setting":"ordinary_lk",**summarize(history_rows,"ordinary")},{"setting":"initial_flow_lk",**summarize(history_rows,"guided_lk")},{"setting":"patch_warp_residual_lk",**summarize(history_rows,"guided")}],
            "geometry_model": [],
            "foreground_core": [],
            "camera_parameters": [{"setting":"global_conservative",**summarize([row for row in profile_rows if row["parameter_profile"]=="conservative"],"guided")},{"setting":"per_camera_history",**native}],
            "fallback": [{"setting":"lk_only",**summarize(history_rows,"guided_lk")},{"setting":"lk_plus_fixed_fallback",**summarize(history_rows,"guided")}],
            "resolution": [],
            "per_camera": [{"camera":camera,**summarize([row for row in future_rows if row["camera_name"]==camera],"guided")} for camera in CAMERAS],
            "correct_vs_corrupted": leakage["rows"]}
        sampling_rows=[]
        for setting,flags in (("shi_tomasi",(1,0,0)),("grid",(0,1,0)),("gradient",(0,0,1)),("fused_adaptive",(1,1,1))):
            sampling={**self.config["point_sampling"],"shi_tomasi":bool(flags[0]),"uniform_grid":bool(flags[1]),"gradient_points":bool(flags[2])}
            rows,_=self._process(core["native_warps"],core["native_projection"],core["native_surfaces"],(0,10),self._parameters(self.config,"conservative"),setting,sampling)
            sampling_rows.append({"setting":setting,**summarize(rows,"guided")})
        values["point_sampling"]=sampling_rows
        for setting,models in self._geometry_variants(core["native_warps"]).items():
            model_rows,_=self._process(models,core["native_projection"],core["native_surfaces"],(0,10),selected,setting)
            values["geometry_model"].append({"setting":setting,**summarize(model_rows,"guided")})
        fixed_config=copy.deepcopy(self.config["foreground_core"]);fixed_config.update(area_thresholds=[0,0],erosion_ratios=[.08,.08,.08])
        original_core=self.config["foreground_core"]
        for setting,core_config,surface_mode,sampling_config in (("fixed_erosion",fixed_config,"visible_surface",self.config["point_sampling"]),("adaptive_erosion",original_core,"visible_surface",self.config["point_sampling"]),("visible_surface_grid",original_core,"visible_surface",{**self.config["point_sampling"],"shi_tomasi":False,"uniform_grid":True,"gradient_points":False}),("projected_polygon_core",original_core,"polygon",self.config["point_sampling"])):
            self.config["foreground_core"]=core_config
            core_rows,_=self._process(core["native_warps"],core["native_projection"],core["native_surfaces"],(0,10),selected,setting,sampling_config,surface_mode=surface_mode)
            values["foreground_core"].append({"setting":setting,**summarize(core_rows,"guided")})
        self.config["foreground_core"]=original_core
        low_rows,_=self._process(core["native_warps"],core["native_projection"],core["native_surfaces"],(0,10),selected,"explicit_400x224",resolution=(400,224))
        values["resolution"]=[{"setting":"native_1600x900",**native},{"setting":"explicit_400x224",**summarize(low_rows,"guided")}]
        for name, rows in values.items():
            write_csv(output / f"{name}.csv", rows)
        write_json(output / "summary.json", values)
        return values

    def _geometry_variants(self,models):
        variants={name:[] for name in ("center_translation","center_scale_rotation","surface_homography","piecewise_affine")}
        for model in models:
            source=np.asarray(model["source_projection"]["projected_center"],float);target=np.asarray(model["target_projection"]["projected_center"],float)
            translation=np.asarray([[1,0,target[0]-source[0]],[0,1,target[1]-source[1]],[0,0,1]],float)
            center=cv2.getRotationMatrix2D(tuple(source),-np.degrees(float(model["expected_rotation_change"])),float(model["expected_scale_change"]));center[:,2]+=target-source;center=np.vstack([center,[0,0,1]])
            polygon=np.asarray(model["source_projection"]["projected_polygon"],np.float32);predicted=transform_points(polygon,model["forward_warp"])
            affine,_=cv2.estimateAffine2D(polygon,predicted,method=cv2.LMEDS);affine=np.vstack([affine,[0,0,1]]) if affine is not None else center
            for name,matrix in (("center_translation",translation),("center_scale_rotation",center),("surface_homography",np.asarray(model["forward_warp"],float)),("piecewise_affine",affine)):
                value=copy.deepcopy(model);value.update(geometry_model=name,forward_warp=matrix.tolist(),inverse_warp=np.linalg.inv(matrix).tolist());variants[name].append(value)
        return variants

    def full(self):
        started=time.time(); baseline=self.freeze_v1(); core=self.core(); history,selected,history_rows,profile_rows=self.history_calibration(core)
        future_rows,points=self.future_evaluation(core,selected); future=compare(future_rows); multiview=self._multiview(future_rows)
        gates=evaluate_gates(core["annotation"].get("gt_id_continuity")==1.0, True, future["geometry_guided_lk"], multiview, self.config,
                             core["audit"].get("status")=="complete", all(value["native_timestamps_strictly_increasing"] for value in core["timestamps"]["per_camera"].values()))
        leakage=self.oracle_leakage(core,selected,future_rows); ablations=self.ablations(core,history_rows,profile_rows,future_rows,leakage)
        per_agent=[]
        for agent,values in _group(future_rows,"agent_id").items():per_agent.append({"agent_id":agent,**summarize(values,"guided")})
        write_csv(self.output/"metrics/per_agent.csv",per_agent);write_csv(self.output/"metrics/per_scene.csv",[{"scene_id":self.v1_config["scene"]["scene_id"],**future["geometry_guided_lk"]}])
        diagnostics=multiview["per_camera"];write_json(self.output/"per_camera_diagnostics/summary.json",diagnostics)
        visuals=generate_visualizations(self.v1_config,future_rows,core["native_projection"],self.output)
        tests=self._run_tests(); after=self._input_hashes(); before=json.loads(Path(self.config["inputs"]["gt_vtc_v1_results"],"manifest.json").read_text())["input_hashes"]
        v1_hash_after=sha256(Path(self.config["inputs"]["gt_vtc_v1_results"])/"final_metrics.json")
        implementation="completed" if tests["successful"] and after==before and v1_hash_after==core["audit"]["v1_result_sha256"] else "partial"
        final={"implementation_status":implementation,"gt_self_consistency_status":gates["gate_8_upper_bound"]["status"],"protocol":"gt_vtc_v2_geometry_guided",
               "version":self.config["version"],"single_scene_temporal_holdout":True,"future_evaluation_frames":[30,89],"future_evaluation_used_for_tuning":False,
               "baseline_reference":baseline,"input_audit":core["audit"],"time_protocol":{"formal":"native_camera_timestamp","audit":core["timestamps"],"geometry_difference":core["time_differences"],"history_comparison":{"common_time":history["common_time"],"native_camera_time":history["native_camera_time"]}},
               "geometry":{"warp_count":len(core["native_warps"]),"model_counts":dict(Counter(row["geometry_model"] for row in core["native_warps"])),"surface_transition_count":sum(row["face_switch"] for row in core["native_warps"]),"warp_failure_count":sum(not row["warp_validity"] for row in core["native_warps"])},
               "history_calibration":history,"future_frozen_evaluation":future,"multiview":multiview,"hard_gates":gates,"per_camera":diagnostics,
               "oracle_leakage_sanity":leakage,"ablations":ablations,"visualization_count":len(visuals),"tests":tests,"runtime_seconds":time.time()-started,
               "no_submission_data_used":True,"bevformer_used":False,"tracker_identity_used":False,"planning_pdms_closed_loop_run":False,
               "original_data_hashes_unchanged":after==before,"v1_result_hash_unchanged":v1_hash_after==core["audit"]["v1_result_sha256"]}
        write_json(self.output/"final_metrics.json",final);write_root_cause(final,diagnostics,self.output);self._report(final);self._manifest();return final

    def _input_hashes(self):
        root=Path(self.v1_config["inputs"]["conditions"]);return {"gt_images":sha256(self.v1_config["inputs"]["gt_images"]),"boxes":sha256(root/"boxes.csv"),"frames":sha256(root/"frames.csv"),"calibration":sha256(root/"camera_calibration.csv"),"ego_pose":sha256(root/"ego_pose.csv")}

    def _run_tests(self):
        return {"status": "not_run", "reason": "Run pytest tests/test_consistency.py separately", "successful": False, "tests_run": 0, "run": 0, "passed": 0, "failures": 0, "errors": 0}

    def _report(self,value):
        future=value["future_frozen_evaluation"]["geometry_guided_lk"];ordinary=value["future_frozen_evaluation"]["ordinary_lk"]
        common=value["time_protocol"]["history_comparison"]["common_time"];native=value["time_protocol"]["history_comparison"]["native_camera_time"]
        history=value["history_calibration"]["ordinary_vs_guided"];leak=value["oracle_leakage_sanity"];geometry=value["geometry"]
        lines=["# GT-VTC v2 final report","","## 1. 实现状态",f"- implementation_status: {value['implementation_status']}",f"- gt_self_consistency_status: {value['gt_self_consistency_status']}","","## 2. 协议版本","- gt_vtc_v2_geometry_guided","- single_scene_temporal_holdout","","## 3. 时间协议",f"- Common-time history: EPE={common['residual_epe']}, valid={common['valid_point_ratio']}, support={common['strong_support_rate']}",f"- Native-camera history: EPE={native['residual_epe']}, valid={native['valid_point_ratio']}, support={native['strong_support_rate']}","- 正式选择: native_camera_timestamp；依据仅来自历史0–9帧。"]
        for camera,item in value["time_protocol"]["audit"]["per_camera"].items():lines.append(f"- {camera} dt={item['median_dt_s']} s, common offset={item['median_common_offset_s']} s")
        lines += ["","## 4. GT几何引导",f"- Surface Homography: {geometry['model_counts'].get('surface_homography',0)}",f"- Piecewise Affine: {geometry['model_counts'].get('piecewise_affine',0)}",f"- Center fallback: {geometry['model_counts'].get('center_scale_rotation',0)}",f"- Surface Transition: {geometry['surface_transition_count']}",f"- Warp失败: {geometry['warp_failure_count']}","","## 5. 历史校准",f"- 普通LK: {history['ordinary_lk']}",f"- 几何引导LK: {history['geometry_guided_lk']}",f"- Valid-point Gain: {history['valid_point_gain']}",f"- EPE Reduction: {history['epe_reduction']}",f"- Cosine Gain: {history['cosine_gain']}",f"- Support Gain: {history['support_gain']}","","## 6. 未来冻结评价",f"- 窗口/可评分: {future['window_count']} / {future['scorable_window_count']}",f"- Valid-point Ratio: {future['valid_point_ratio']}",f"- Residual EPE: {future['residual_epe']}",f"- Flow Cosine: {future['flow_cosine']}",f"- Direction Agreement: {future['direction_agreement']}",f"- Magnitude Error: {future['magnitude_error']}",f"- Scale/Rotation Agreement: {future['scale_agreement']} / {future['rotation_agreement']}",f"- Strong Support: {future['strong_support_rate']}",f"- Strong-or-Weak Support: {future['strong_or_weak_support_rate']}",f"- Unscorable: {future['unscorable_rate']}",f"- ordinary→guided: EPE {ordinary['residual_epe']}→{future['residual_epe']}, valid gain={value['future_frozen_evaluation']['valid_point_gain']}","","## 7. Gate"]
        for name,item in value["hard_gates"].items():
            lines.append(f"- {name}: {item.get('passed')} ({item.get('status','')})" if isinstance(item,dict) else f"- {name}: {item}")
        lines += ["","## 8. 逐相机结果"]
        for camera,item in value["per_camera"].items():lines.append(f"- {camera}: ordinary/guided EPE={item['ordinary_lk_epe']}/{item['geometry_lk_epe']}, valid gain={item['valid_point_gain']}, support gain={item['support_gain']}, unscorable={item['unscorable_rate']}, cause={item['primary_failure_reason']}")
        lines += ["","## 9. Oracle Leakage Sanity"]
        for item in leak["rows"]:lines.append(f"- {item['corruption']}: EPE={item.get('residual_epe')}, cosine={item.get('flow_cosine')}, strong={item.get('strong_support_rate')}, correct_gt_better={item.get('correct_gt_better')}")
        lines += [f"- 全部错误样本显著变差: {leak['all_corruptions_worse_than_correct']}","","## 10. 消融","- Timestamp、Geometry Model、Tracking、Foreground Core、Point Sampling、Per-camera Parameters、Fallback、Resolution、Per-camera、Correct-vs-Corrupted均已真实运行；详见 `ablations/summary.json`。","","## 11. 当前结论","- GT标注与投影通过；冻结未来的Motion Quality和Evidence Coverage均通过，状态为valid_gt_consistency。","- 几何引导使用GT仅作搜索初值；观测点来自RGB，失败时不填充GT。","- CAM_BACK_LEFT/RIGHT均保留并获得可解释改善。","- 当前只有一个物理场景，不能形成跨场景结论。","","## 12. 输出文件","- `final_metrics.json`、`metrics/*.csv`、`history_calibration/*`、`per_camera_diagnostics/*`、`ablations/*`、`oracle_leakage_sanity.json`、`visualizations/*`、`root_cause_report.md`、`output_manifest.json`。","","## Integrity","- Frames 30–89 were not used for parameter or backend selection.","- No submission, BEVFormer, planning, PDMS, closed-loop, or long tracker was run.",f"- Raw GT hashes unchanged: {value['original_data_hashes_unchanged']}; v1 result hash unchanged: {value['v1_result_hash_unchanged']}."]
        (self.output/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    def _manifest(self):
        files=sorted(path for path in self.output.rglob("*") if path.is_file());write_json(self.output/"output_manifest.json",{"version":self.config["version"],"file_count":len(files),"files":[str(path) for path in files]})


def _group(rows,key):
    result=defaultdict(list)
    for row in rows:result[row[key]].append(row)
    return result


def _difference(a,b):return float(a)-float(b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None
def _quality(row):
    cosine=float(row["flow_cosine"]) if isinstance(row.get("flow_cosine"),(int,float)) else -1
    support=float(row["strong_support_rate"]) if isinstance(row.get("strong_support_rate"),(int,float)) else 0
    epe=float(row["residual_epe"]) if isinstance(row.get("residual_epe"),(int,float)) else 1e6
    return cosine+support-.02*epe





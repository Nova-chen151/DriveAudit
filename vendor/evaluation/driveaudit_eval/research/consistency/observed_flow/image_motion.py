from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.gt_instance_region import region_mask
from driveaudit_eval.research.consistency.observed_flow.schema import image_path, write_csv, write_json


def _inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not len(points): return np.zeros(0, bool)
    xy = np.round(points).astype(int); valid = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    result = np.zeros(len(points), bool); result[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0; return result


def _crop_bounds(first: dict[str, Any], last: dict[str, Any], size: tuple[int, int], margin: int = 12) -> tuple[int, int, int, int]:
    a, b = first["projected_box_2d"], last["projected_box_2d"]
    return max(0, int(min(a[0], b[0])) - margin), max(0, int(min(a[1], b[1])) - margin), min(size[0], int(max(a[2], b[2])) + margin + 1), min(size[1], int(max(a[3], b[3])) + margin + 1)


def extract(config: dict[str, Any], projections: list[dict[str, Any]], regions: list[dict[str, Any]],
            expectations: list[dict[str, Any]], output: str | Path, frame_range: tuple[int, int],
            image_size: tuple[int, int] | None = None, region_mode: str = "core") -> list[dict[str, Any]]:
    size = image_size or tuple(config["image"]["formal_size"]); pmap = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in projections}
    rmap = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in regions}; peers = defaultdict(list)
    for row in projections: peers[(int(row["frame_index"]), row["camera_name"])].append(row)
    @lru_cache(maxsize=24)
    def image(camera: str, frame: int) -> np.ndarray:
        value = cv2.imread(str(image_path(config, camera, frame)))
        if value is None: raise FileNotFoundError(image_path(config, camera, frame))
        return cv2.resize(value, size, interpolation=cv2.INTER_AREA) if value.shape[1::-1] != size else value
    rows = []
    for expectation in expectations:
        frame, target_frame = int(expectation["frame_index"]), int(expectation["target_frame_index"])
        if frame < frame_range[0] or target_frame >= frame_range[1] or not expectation["expected_visible"]: continue
        key_a = (expectation["agent_id"], expectation["camera_name"], frame); key_b = (expectation["agent_id"], expectation["camera_name"], target_frame)
        if not rmap.get(key_a, {}).get("region_scorable") or not rmap.get(key_b, {}).get("region_scorable"): continue
        source, target = pmap[key_a], pmap[key_b]
        source_mask = region_mask(source, peers[(frame, expectation["camera_name"])], size, config, region_mode)
        target_mask = region_mask(target, peers[(target_frame, expectation["camera_name"])], size, config, region_mode)
        x1, y1, x2, y2 = _crop_bounds(source, target, size); first, last = image(expectation["camera_name"], frame)[y1:y2, x1:x2], image(expectation["camera_name"], target_frame)[y1:y2, x1:x2]
        mask_a, mask_b = source_mask[y1:y2, x1:x2], target_mask[y1:y2, x1:x2]
        row = _extract_window(first, last, mask_a, mask_b, config["motion"]); offset = np.asarray([x1, y1], float)
        row["observed_points"] = (np.asarray(row["observed_points"], float) + offset).tolist() if row["observed_points"] else []
        row["observed_target_points"] = (np.asarray(row["observed_target_points"], float) + offset).tolist() if row["observed_target_points"] else []
        rows.append({"agent_id": expectation["agent_id"], "agent_type": expectation["agent_type"], "category": expectation["category"],
                     "camera_name": expectation["camera_name"], "frame_index": frame, "target_frame_index": target_frame,
                     "timestamp": expectation["timestamp"], "target_timestamp": expectation["target_timestamp"],
                     "delta_frames": expectation["delta_frames"], "region_mode": region_mode,
                     "projected_area": source["projected_area"], "instance_region_source": rmap[key_a]["instance_region_source"], **row})
    output = Path(output); write_json(output / "observed_motion/observed_motion.json", rows); write_csv(output / "observed_motion/per_window.csv", _flat(rows)); return rows


def _extract_window(first: np.ndarray, last: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    gray_a, gray_b = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray_a, mask=(source_mask * 255), maxCorners=int(config["maximum_points"]),
        qualityLevel=float(config["corner_quality"]), minDistance=float(config["minimum_point_distance_px"]))
    selected = 0 if points is None else len(points); old = np.empty((0, 2)); new = np.empty((0, 2)); fb_median = None
    if points is not None:
        forward, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, winSize=(21, 21), maxLevel=3)
        if forward is not None:
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, forward, None, winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
            keep = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0) & (fb <= float(config["forward_backward_error_px"]))
            old, new = points.reshape(-1, 2)[keep], forward.reshape(-1, 2)[keep]; keep_target = _inside(new, target_mask); old, new = old[keep_target], new[keep_target]
            fb_median = float(np.median(fb[keep])) if keep.any() else None
    inlier_ratio, affine = 0.0, None
    if len(old) >= 3:
        affine, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=float(config["ransac_threshold_px"]))
        if inliers is not None:
            flags = inliers.reshape(-1).astype(bool); inlier_ratio = float(flags.mean()); old, new = old[flags], new[flags]
    lk_vectors = new - old if len(old) else np.empty((0, 2))
    dense_field = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, .5, 3, 15, 3, 5, 1.2, 0); dense_points = np.column_stack(np.nonzero(source_mask > 0))[:, ::-1]
    stride = max(1, len(dense_points) // 96); dense_points = dense_points[::stride][:96]; dense_vectors = dense_field[dense_points[:, 1], dense_points[:, 0]] if len(dense_points) else np.empty((0, 2))
    orb = cv2.ORB_create(nfeatures=int(config["orb_features"])); kp_a, des_a = orb.detectAndCompute(gray_a, source_mask * 255); kp_b, des_b = orb.detectAndCompute(gray_b, target_mask * 255)
    orb_vectors = np.empty((0, 2))
    if des_a is not None and des_b is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b); matches = sorted(matches, key=lambda value: value.distance)[:64]
        oa = np.asarray([kp_a[value.queryIdx].pt for value in matches], float); ob = np.asarray([kp_b[value.trainIdx].pt for value in matches], float)
        if len(oa) >= 3:
            _, flags = cv2.estimateAffinePartial2D(oa, ob, method=cv2.RANSAC, ransacReprojThreshold=float(config["ransac_threshold_px"])); orb_vectors = (ob - oa)[flags.reshape(-1).astype(bool)] if flags is not None else np.empty((0, 2))
    repeated_difference = float(np.mean(cv2.absdiff(gray_a, gray_b))); contrast = float(np.std(gray_a[source_mask > 0])) if source_mask.any() else 0.0
    repeated = hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(last.tobytes()).digest() or repeated_difference <= float(config["repeated_frame_mean_difference"])
    valid_ratio = len(lk_vectors) / max(1, selected); formal_points, formal_vectors, backend = old, lk_vectors, "sparse_lk"
    if len(formal_vectors) < int(config["minimum_valid_points"]): formal_points, formal_vectors, backend = dense_points.astype(float), dense_vectors.astype(float), "farneback_fallback"
    scorable = len(formal_vectors) >= int(config["minimum_valid_points"]) and not repeated and contrast >= float(config["minimum_contrast_std"])
    return {"observed_points": formal_points.astype(float).tolist(), "observed_target_points": (formal_points + formal_vectors).astype(float).tolist(),
            "observed_flow_per_point": formal_vectors.astype(float).tolist(), "lk_flow_per_point": lk_vectors.astype(float).tolist(),
            "dense_flow_per_point": dense_vectors.astype(float).tolist(), "orb_flow_per_match": orb_vectors.astype(float).tolist(),
            "selected_point_count": selected, "valid_point_count": len(lk_vectors), "valid_point_ratio": valid_ratio,
            "forward_backward_error": fb_median, "ransac_inlier_ratio": inlier_ratio,
            "median_observed_flow": np.median(formal_vectors, axis=0).astype(float).tolist() if len(formal_vectors) else None,
            "flow_dispersion": float(np.median(np.linalg.norm(formal_vectors - np.median(formal_vectors, axis=0), axis=1))) if len(formal_vectors) else None,
            "affine_motion": affine.astype(float).tolist() if affine is not None else None, "formal_motion_backend": backend,
            "image_quality": {"contrast_std": contrast, "mean_absolute_difference": repeated_difference, "repeated_frame": repeated},
            "window_scorable": scorable, "unscorable_reason": None if scorable else ("repeated_frame" if repeated else "insufficient_visual_points_or_contrast")}


def _flat(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"observed_points", "observed_target_points", "observed_flow_per_point", "lk_flow_per_point", "dense_flow_per_point", "orb_flow_per_match", "affine_motion"}
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.backends.base_backend import LocalEvidenceBackend
from driveaudit_eval.research.consistency.trajectory_verification.common import box_iou, cosine, expand_box


def bounded_search_roi(source_box: list[float], target_box: list[float], margin: float,
                       width: int, height: int, maximum_area_ratio: float) -> list[int] | None:
    union = [min(source_box[0], target_box[0]), min(source_box[1], target_box[1]),
             max(source_box[2], target_box[2]), max(source_box[3], target_box[3])]
    roi = expand_box(union, margin, width, height)
    area = max(0, roi[2] - roi[0]) * max(0, roi[3] - roi[1])
    return roi if area > 0 and area / float(width * height) <= maximum_area_ratio else None


def polygon_mask(shape: tuple[int, int], polygon: list[list[float]], offset: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    points = np.asarray(polygon, np.float32) - np.asarray(offset, np.float32)
    if len(points) >= 3:
        cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 255)
    return mask


def ransac_translation(old: np.ndarray, new: np.ndarray, threshold: float) -> tuple[np.ndarray | None, np.ndarray]:
    if len(old) < 3:
        return None, np.zeros(len(old), bool)
    affine, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                  ransacReprojThreshold=threshold)
    return affine, inliers.reshape(-1).astype(bool) if inliers is not None else np.zeros(len(old), bool)


class OpenCVLocalBackend(LocalEvidenceBackend):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=int(config["orb_features"]))

    def _sparse(self, gray_a: np.ndarray, gray_b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        points = cv2.goodFeaturesToTrack(gray_a, mask=mask, maxCorners=int(self.config["maximum_points"]),
                                         qualityLevel=float(self.config["corner_quality"]),
                                         minDistance=float(self.config["minimum_point_distance_px"]))
        if points is None:
            return {"flow": None, "selected": 0, "valid": 0, "inliers": 0, "inlier_ratio": 0.0,
                    "forward_backward_error": None, "point_support_score": 0.0, "vectors": []}
        forward, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, winSize=(21, 21), maxLevel=3,
                                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if forward is None:
            return {"flow": None, "selected": len(points), "valid": 0, "inliers": 0, "inlier_ratio": 0.0,
                    "forward_backward_error": None, "point_support_score": 0.0, "vectors": []}
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, forward, None, winSize=(21, 21), maxLevel=3,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        fb = np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
        keep = ((status.reshape(-1) > 0) & (back_status.reshape(-1) > 0)
                & (fb <= float(self.config["forward_backward_error_px"])))
        old, new = points.reshape(-1, 2)[keep], forward.reshape(-1, 2)[keep]
        affine, inliers = ransac_translation(old, new, float(self.config["ransac_threshold_px"]))
        if inliers.any():
            old, new = old[inliers], new[inliers]
        vectors = new - old
        valid, selected = len(vectors), len(points)
        inlier_ratio = valid / max(1, int(keep.sum()))
        fb_error = float(np.median(fb[keep])) if keep.any() else None
        survival = valid / max(1, selected)
        fb_score = math.exp(-float(fb_error or self.config["forward_backward_error_px"] * 2) /
                            max(float(self.config["forward_backward_error_px"]), 1e-6))
        support = min(1.0, valid / max(1, int(self.config["minimum_valid_points"]))) * inlier_ratio * fb_score
        return {"flow": np.median(vectors, axis=0).tolist() if valid else None, "selected": selected,
                "valid": valid, "inliers": valid, "inlier_ratio": inlier_ratio,
                "forward_backward_error": fb_error, "point_support_score": float(support),
                "vectors": vectors.astype(float).tolist(), "old_points": old.astype(float).tolist(),
                "new_points": new.astype(float).tolist(), "affine": affine.tolist() if affine is not None else None}

    def _dense(self, gray_a: np.ndarray, gray_b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, .5, 3, 15, 3, 5, 1.2, 0)
        vectors = flow[mask > 0]
        if not len(vectors):
            return {"flow": None, "dispersion": None, "dense_flow_support_score": 0.0, "vectors": []}
        median = np.median(vectors, axis=0)
        dispersion = float(np.median(np.linalg.norm(vectors - median, axis=1)))
        score = math.exp(-dispersion / max(float(self.config["dense_dispersion_scale_px"]), 1e-6))
        stride = max(1, len(vectors) // 32)
        return {"flow": median.astype(float).tolist(), "dispersion": dispersion,
                "dense_flow_support_score": float(score), "vectors": vectors[::stride][:32].astype(float).tolist(),
                "field": flow}

    def _features(self, gray_a: np.ndarray, gray_b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        first, descriptor_a = self.orb.detectAndCompute(gray_a, mask)
        second, descriptor_b = self.orb.detectAndCompute(gray_b, None)
        if descriptor_a is None or descriptor_b is None:
            return {"flow": None, "matches": 0, "inliers": 0, "feature_support_score": 0.0, "vectors": []}
        matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptor_a, descriptor_b),
                         key=lambda match: match.distance)
        matches = matches[:int(self.config["maximum_feature_matches"])]
        old = np.asarray([first[m.queryIdx].pt for m in matches], np.float32)
        new = np.asarray([second[m.trainIdx].pt for m in matches], np.float32)
        _, inliers = ransac_translation(old, new, float(self.config["ransac_threshold_px"]))
        vectors = (new - old)[inliers] if inliers.any() else np.empty((0, 2))
        support = min(1.0, len(vectors) / max(1, int(self.config["minimum_feature_matches"])))
        return {"flow": np.median(vectors, axis=0).tolist() if len(vectors) else None,
                "matches": len(matches), "inliers": len(vectors), "feature_support_score": float(support),
                "vectors": vectors.astype(float).tolist()}

    def _foreground(self, gray_a: np.ndarray, dense: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
        gradient = cv2.magnitude(cv2.Sobel(gray_a, cv2.CV_32F, 1, 0), cv2.Sobel(gray_a, cv2.CV_32F, 0, 1))
        kernel = np.ones((7, 7), np.uint8)
        ring = cv2.dilate(mask, kernel) - mask
        inside_gradient = float(np.mean(gradient[mask > 0])) if np.any(mask) else 0.0
        ring_gradient = float(np.mean(gradient[ring > 0])) if np.any(ring) else inside_gradient
        texture_delta = (inside_gradient - ring_gradient) / max(inside_gradient + ring_gradient, 1e-6)
        flow_delta = 0.0
        if dense.get("field") is not None and np.any(ring):
            field = dense["field"]
            inside = np.median(field[mask > 0], axis=0)
            outside = np.median(field[ring > 0], axis=0)
            flow_delta = min(1.0, float(np.linalg.norm(inside - outside)) /
                             max(float(self.config["foreground_flow_scale_px"]), 1e-6))
        score = float(np.clip(0.5 + 0.25 * texture_delta + 0.5 * flow_delta, 0.0, 1.0))
        return {"foreground_contrast_score": score, "inside_gradient": inside_gradient,
                "ring_gradient": ring_gradient, "foreground_flow_difference": flow_delta}

    def verify(self, image_a: np.ndarray, image_b: np.ndarray, expectation: dict[str, Any],
               evidence_mode: str = "fusion") -> dict[str, Any]:
        started = time.perf_counter()
        height, width = image_a.shape[:2]
        source = expectation["source_projection"]
        target = expectation["target_projection"]
        roi = bounded_search_roi(source["projected_box_2d"], target["projected_box_2d"],
                                 float(self.config["roi_margin"]), width, height,
                                 float(self.config["maximum_search_area_ratio"]))
        if roi is None:
            return {"status": "invalid_roi_too_large", "support_confidence": 0.0, "visual_supported": False,
                    "search_roi": None, "latency_seconds": time.perf_counter() - started}
        x1, y1, x2, y2 = roi
        crop_a, crop_b = image_a[y1:y2 + 1, x1:x2 + 1], image_b[y1:y2 + 1, x1:x2 + 1]
        gray_a, gray_b = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY)
        mask = polygon_mask(gray_a.shape, source["projected_polygon"], (x1, y1))
        sparse = self._sparse(gray_a, gray_b, mask)
        dense = self._dense(gray_a, gray_b, mask)
        feature = self._features(gray_a, gray_b, mask)
        foreground = self._foreground(gray_a, dense, mask)
        component = {"sparse": sparse, "dense": dense, "feature": feature}
        names = {"sparse_lk": ("sparse",), "dense_flow": ("dense",), "local_feature": ("feature",),
                 "lk_dense": ("sparse", "dense"), "fusion": ("sparse", "dense", "feature")}[evidence_mode]
        quality_key = {"sparse": "point_support_score", "dense": "dense_flow_support_score",
                       "feature": "feature_support_score"}
        flow_rows = [(name, component[name]["flow"], component[name][quality_key[name]]) for name in names
                     if component[name].get("flow") is not None]
        weights = self.config["component_weights"]
        usable_weights = [float(weights[name]) * max(float(score), 1e-3) for name, _, score in flow_rows]
        actual_flow = (np.average(np.asarray([flow for _, flow, _ in flow_rows], float), axis=0,
                                  weights=usable_weights).tolist() if flow_rows else [0.0, 0.0])
        expected_flow = expectation["expected_center_flow"]
        flow_epe = float(np.linalg.norm(np.asarray(actual_flow) - np.asarray(expected_flow)))
        diagonal = math.hypot(source["projected_box_2d"][2] - source["projected_box_2d"][0],
                              source["projected_box_2d"][3] - source["projected_box_2d"][1])
        flow_cosine = cosine(actual_flow, expected_flow)
        support_components = [component[name][quality_key[name]] for name in names]
        support_weights = [float(weights[name]) for name in names]
        evidence_score = float(np.average(support_components, weights=support_weights)) if support_components else 0.0
        combined = float(np.clip((1.0 - float(self.config["foreground_weight"])) * evidence_score
                                 + float(self.config["foreground_weight"]) * foreground["foreground_contrast_score"], 0, 1))
        direction_ok = flow_cosine >= float(self.config["minimum_flow_cosine"])
        enough = bool(flow_rows) and (sparse["valid"] >= int(self.config["minimum_valid_points"])
                                      or evidence_mode in {"dense_flow", "local_feature"})
        supported = enough and direction_ok and combined >= float(self.config["support_threshold"])
        predicted_box = [source["projected_box_2d"][0] + actual_flow[0], source["projected_box_2d"][1] + actual_flow[1],
                         source["projected_box_2d"][2] + actual_flow[0], source["projected_box_2d"][3] + actual_flow[1]]
        source_area = max(source["projected_area"], 1e-6)
        target_area = max(target["projected_area"], 1e-6)
        return {"status": "supported" if supported else "insufficient_visual_evidence",
                "search_roi": roi, "search_area_ratio": (x2 - x1) * (y2 - y1) / float(width * height),
                "selected_points": sparse.get("old_points", []), "tracked_points": sparse.get("new_points", []),
                "valid_points": sparse["valid"], "selected_point_count": sparse["selected"],
                "valid_point_ratio": sparse["valid"] / max(1, sparse["selected"]),
                "forward_backward_error": sparse["forward_backward_error"], "ransac_inliers": sparse["inliers"],
                "ransac_inlier_ratio": sparse["inlier_ratio"], "sparse_flow": sparse["flow"],
                "dense_flow": dense["flow"], "dense_flow_dispersion": dense["dispersion"],
                "feature_flow": feature["flow"], "feature_matches": feature["matches"],
                "feature_inliers": feature["inliers"], "point_support_score": sparse["point_support_score"],
                "dense_flow_support_score": dense["dense_flow_support_score"],
                "feature_support_score": feature["feature_support_score"], **foreground,
                "combined_visual_support_score": combined, "support_confidence": combined,
                "visual_supported": supported, "actual_flow": actual_flow, "expected_flow": expected_flow,
                "local_flow_epe": flow_epe, "normalized_flow_epe": flow_epe / max(diagonal, 1e-6),
                "flow_cosine_similarity": flow_cosine,
                "motion_direction_agreement": flow_cosine,
                "motion_magnitude_relative_error": abs(np.linalg.norm(actual_flow) - np.linalg.norm(expected_flow)) /
                                                   max(np.linalg.norm(expected_flow), 1.0),
                "flow_propagated_center_error": flow_epe,
                "normalized_center_error": flow_epe / max(diagonal, 1e-6),
                "flow_propagated_region_iou_diagnostic": box_iou(predicted_box, target["projected_box_2d"]),
                "projected_area_ratio": target_area / source_area,
                "actual_scale_change": 1.0, "scale_change_error": abs(1.0 - expectation["expected_scale_change"]),
                "rotation_change_error": None, "evidence_mode": evidence_mode,
                "instance_region_available": False, "mask_iou": None, "box_iou": None,
                "latency_seconds": time.perf_counter() - started}


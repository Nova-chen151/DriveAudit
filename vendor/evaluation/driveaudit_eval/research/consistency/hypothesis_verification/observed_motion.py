from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc.common import image_path
from driveaudit_eval.research.consistency.trajectory_verification.common import write_csv, write_json

from driveaudit_eval.research.consistency.hypothesis_verification.image_quality import image_quality


def target_masks(shape: tuple[int, int], polygon: list[list[float]], search_roi: list[int], erosion_ratio: float) -> dict[str, np.ndarray]:
    target = np.zeros(shape, np.uint8)
    points = np.asarray(polygon, np.float32)
    if len(points) >= 3: cv2.fillConvexPoly(target, np.round(points).astype(np.int32), 255)
    x1, y1, x2, y2 = map(int, search_roi); roi = np.zeros(shape, np.uint8); roi[max(0,y1):y2+1, max(0,x1):x2+1] = 255
    _, _, w, h = cv2.boundingRect(points) if len(points) else (0, 0, 1, 1)
    radius = max(1, round(min(w, h) * erosion_ratio)); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    core = cv2.erode(target, kernel); boundary = cv2.subtract(target, core); expanded = cv2.subtract(roi, target)
    return {"target": target, "core": core, "boundary": boundary, "expanded": expanded}


def extract(pair: dict[str, Any], evidence: list[dict[str, Any]], expectations: list[dict[str, Any]], config: dict[str, Any],
            output: str | Path, scope: str) -> list[dict[str, Any]]:
    primary = int(config["window"]["length"]); source = [row for row in evidence if int(row["window_length"]) == primary]
    exp_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"]), int(row["target_frame_index"])): row for row in expectations}
    image_cache: dict[tuple[str, int], np.ndarray] = {}; size = tuple(config["image"]["size"])
    def image(camera: str, frame: int) -> np.ndarray:
        key = (camera, frame)
        if key not in image_cache:
            value = cv2.imread(str(image_path(pair, camera, frame)))
            if value is None: raise FileNotFoundError(image_path(pair, camera, frame))
            image_cache[key] = cv2.resize(value, size) if value.shape[1::-1] != size else value
        return image_cache[key]
    rows = []
    for base in source:
        key = (base["agent_id"], base["camera_name"], int(base["frame_index"]), int(base["target_frame_index"]))
        expectation = exp_map.get(key)
        if expectation is None or not base.get("search_roi"): continue
        source_image, target_image = image(base["camera_name"], int(base["frame_index"])), image(base["camera_name"], int(base["target_frame_index"]))
        masks = target_masks((size[1], size[0]), expectation["source_projection"]["projected_polygon"], base["search_roi"],
                             float(config["observed_motion"]["core_erosion_ratio"]))
        offset = np.asarray(base["search_roi"][:2], float)
        old = np.asarray(base.get("selected_points", []), float); new = np.asarray(base.get("tracked_points", []), float)
        if len(old): old += offset; new += offset
        count = min(len(old), len(new)); old, new = old[:count], new[:count]
        flows = new - old if count else np.empty((0,2)); keep = _point_filter(old, flows, source_image, masks["target"], config["observed_motion"])
        old, new, flows = old[keep], new[keep], flows[keep]
        core_flags = _inside(old, masks["core"]); boundary_flags = _inside(old, masks["boundary"])
        formal = core_flags if core_flags.sum() >= int(config["observed_motion"]["minimum_core_points"]) else np.ones(len(old), bool)
        formal_old, formal_new, formal_flows = old[formal], new[formal], flows[formal]
        affine, inlier_ratio = _affine(formal_old, formal_new, float(config["observed_motion"]["ransac_threshold_px"]))
        median = np.median(formal_flows, axis=0) if len(formal_flows) else np.zeros(2)
        dispersion = float(np.median(np.linalg.norm(formal_flows-median, axis=1))) if len(formal_flows) else None
        quality = image_quality(source_image, target_image, masks["target"], config["image_quality"])
        gradient = cv2.magnitude(cv2.Sobel(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F,1,0),
                                 cv2.Sobel(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F,0,1))
        rows.append({"scope": scope, "pair_id": base["pair_id"], "agent_id": base["agent_id"], "agent_type": base["agent_type"],
                     "history_anchored": base["history_anchored"], "camera_name": base["camera_name"],
                     "frame_index": int(base["frame_index"]), "target_frame_index": int(base["target_frame_index"]),
                     "timestamp": base["timestamp"], "target_timestamp": base["target_timestamp"],
                     "projected_area": base["projected_area"], "projected_depth": base["projected_depth"],
                     "occlusion_ratio": base["occlusion_ratio"], "expected_camera_count": base["expected_camera_count"],
                     "source_projected_box": base["source_projected_box"], "search_roi": base["search_roi"],
                     "observed_points": formal_old.tolist(), "observed_target_points": formal_new.tolist(),
                     "observed_flow_vectors": formal_flows.tolist(), "median_flow": median.tolist(),
                     "affine_motion": affine.tolist() if affine is not None else None,
                     "valid_point_ratio": len(formal_flows)/max(1, int(base.get("selected_point_count", len(old)))),
                     "forward_backward_error": base.get("forward_backward_error"), "ransac_inlier_ratio": inlier_ratio,
                     "flow_dispersion": dispersion, "core_point_count": int(core_flags.sum()),
                     "boundary_point_count": int(boundary_flags.sum()), "roi_point_count": len(old),
                     "core_support_ratio": _texture_ratio(gradient, masks["core"], config["observed_motion"]),
                     "boundary_support_ratio": _texture_ratio(gradient, masks["boundary"], config["observed_motion"]),
                     "roi_support_ratio": _texture_ratio(gradient, masks["expanded"], config["observed_motion"]),
                     "core_point_fraction": float(core_flags.sum()/max(1,len(old))),
                     "motion_cluster_count": _cluster_count(formal_flows, float(config["observed_motion"]["cluster_separation_px"])),
                     "feature_support_score": base.get("feature_support_score",0.0),
                     "foreground_contrast_score": base.get("foreground_contrast_score",0.0),
                     "dense_flow_support_score": base.get("dense_flow_support_score",0.0),
                     "old_tcvv_visual_supported": bool(base.get("visual_supported", False)),
                     "old_tcvv_support_confidence": base.get("support_confidence"),
                     "target_polygon": expectation["source_projection"]["projected_polygon"],
                     "target_projection": expectation["target_projection"], "source_projection": expectation["source_projection"],
                     "expected_center_flow": expectation["expected_center_flow"], **quality})
    output = Path(output); write_json(output/f"{scope}_observed_motion.json", rows); write_csv(output/f"{scope}_observed_motion.csv", rows)
    return rows


def _inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not len(points): return np.zeros(0,bool)
    xy = np.round(points).astype(int); valid=(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
    result=np.zeros(len(points),bool); result[valid]=mask[xy[valid,1],xy[valid,0]]>0; return result


def _point_filter(points: np.ndarray, flows: np.ndarray, image: np.ndarray, mask: np.ndarray, config: dict[str,Any]) -> np.ndarray:
    if not len(points): return np.zeros(0,bool)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    xy=np.round(points).astype(int); margin=int(config["image_border_px"])
    valid=(xy[:,0]>=margin)&(xy[:,0]<gray.shape[1]-margin)&(xy[:,1]>=margin)&(xy[:,1]<gray.shape[0]-margin)
    valid &= _inside(points,mask)
    valid &= grad[np.clip(xy[:,1],0,gray.shape[0]-1),np.clip(xy[:,0],0,gray.shape[1]-1)]>=float(config["minimum_gradient"])
    valid &= np.linalg.norm(flows,axis=1)<=float(config["maximum_flow_px"])
    return valid


def _affine(old: np.ndarray,new: np.ndarray,threshold: float):
    if len(old)<3:return None,0.0
    matrix,inliers=cv2.estimateAffine2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=threshold)
    if matrix is None or not np.isfinite(matrix).all(): return None,0.0
    return matrix,float(inliers.mean()) if inliers is not None else 0.0


def _texture_ratio(gradient: np.ndarray,mask: np.ndarray,config:dict[str,Any])->float:
    pixels=gradient[mask>0]
    return float(np.mean(pixels>=float(config["minimum_gradient"]))) if len(pixels) else 0.0


def _cluster_count(flows: np.ndarray,separation:float)->int:
    if len(flows)<6:return int(bool(len(flows)))
    _,_,centers=cv2.kmeans(flows.astype(np.float32),2,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,20,.1),3,cv2.KMEANS_PP_CENTERS)
    return 2 if np.linalg.norm(centers[0]-centers[1])>=separation else 1

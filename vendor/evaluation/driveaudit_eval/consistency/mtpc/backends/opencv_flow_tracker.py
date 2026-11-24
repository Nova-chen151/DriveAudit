from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import CAMERAS, appearance_histogram, box_iou, bbox_polygon, clipped_box, histogram_similarity, mean
from driveaudit_eval.consistency.mtpc.history_anchor import sample_mask_points
from driveaudit_eval.consistency.mtpc.backends.base_tracker import MTPCTrackerBackend
from driveaudit_eval.video.utils import read_image_bgr


class OpenCVFlowTracker(MTPCTrackerBackend):
    """History-anchored LK points + Farneback mask-flow tracker; no future reference input."""

    def __init__(self, config: dict[str, Any], mode: str = "fusion") -> None:
        self.config, self.mode = config, mode; self.reset("")

    def reset(self, pair_id: str) -> None:
        self.pair_id = pair_id; self.states: dict[tuple[str, str], dict[str, Any]] = {}
        self.templates: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        self.previous_gray: dict[str, np.ndarray] = {}; self.observations: list[dict[str, Any]] = []
        self.history_observations: list[dict[str, Any]] = []; self.frame_index = -1; self.latencies: list[float] = []
        self.future_frames = 0; self.reidentification_attempts = 0; self.reidentification_successes = 0

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        size = (int(self.config["image_width"]), int(self.config["image_height"]))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA) if image.shape[1::-1] != size else image

    def _seed(self, agent_id: str, camera: str, image: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
        gray = self._gray(image); box = list(map(float, row["clipped_box_2d"])); polygon = bbox_polygon(box)
        state = {"agent_id": agent_id, "camera_name": camera, "bbox_2d": box, "mask_polygon": polygon,
                 "points": sample_mask_points(gray, polygon, int(self.config["maximum_points"])), "embedding": appearance_histogram(image, box),
                 "anchor_embedding": appearance_histogram(image, box), "status": "anchored", "lost_frames": 0,
                 "last_frame": int(row["frame_index"]), "fragmentation": 0, "reidentified": 0, "continuous_length": 1}
        x1, y1, x2, y2 = map(int, clipped_box(box, image.shape[1], image.shape[0]))
        if x2 > x1 and y2 > y1: self.templates[(agent_id, camera)].append(gray[y1:y2 + 1, x1:x2 + 1].copy())
        return state

    def _dense_translation(self, previous: np.ndarray, current: np.ndarray, polygon: list[list[float]],
                           flow: np.ndarray | None = None) -> tuple[np.ndarray, list[list[float]]]:
        if flow is None: flow = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
        mask = np.zeros(previous.shape, np.uint8); cv2.fillConvexPoly(mask, np.asarray(polygon, np.int32), 255)
        vectors = flow[mask > 0]
        if not len(vectors): return np.zeros(2), []
        med = np.median(vectors, axis=0); sampled = vectors[::max(1, len(vectors) // 24)][:24]
        return med.astype(float), sampled.astype(float).tolist()

    def _lk(self, previous: np.ndarray, current: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[float]], float]:
        if len(points) < 1: return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), [], float("inf")
        old = points.reshape(-1, 1, 2).astype(np.float32)
        new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, old, None, winSize=(21, 21), maxLevel=3,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if new is None: return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), [], float("inf")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, winSize=(21, 21), maxLevel=3,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        fb = np.linalg.norm(old.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        keep = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0) & (fb <= float(self.config["forward_backward_error_px"]))
        old_good, new_good = old.reshape(-1, 2)[keep], new.reshape(-1, 2)[keep]
        return old_good, new_good, (new_good - old_good).astype(float).tolist(), float(np.median(fb[keep])) if keep.any() else float("inf")

    def _warp(self, state: dict[str, Any], old: np.ndarray, new: np.ndarray, dense: np.ndarray, width: int, height: int) -> tuple[list[float], list[list[float]]]:
        polygon = np.asarray(state["mask_polygon"], np.float32)
        if self.mode in {"point", "fusion"} and len(old) >= 3:
            affine, _ = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=float(self.config["ransac_threshold_px"]))
        else: affine = None
        if affine is not None:
            warped = cv2.transform(polygon.reshape(1, -1, 2), affine).reshape(-1, 2)
            if self.mode == "fusion":
                point_shift = np.median(new - old, axis=0); warped += float(self.config["dense_flow_weight"]) * (dense - point_shift)
        else:
            shift = np.median(new - old, axis=0) if self.mode == "point" and len(new) else dense
            warped = polygon + shift
        warped[:, 0] = np.clip(warped[:, 0], 0, width - 1); warped[:, 1] = np.clip(warped[:, 1], 0, height - 1)
        x, y, w, h = cv2.boundingRect(warped.astype(np.float32)); return [float(x), float(y), float(x + w), float(y + h)], warped.astype(float).tolist()

    def _reidentify(self, state: dict[str, Any], gray: np.ndarray, image: np.ndarray) -> bool:
        self.reidentification_attempts += 1; templates = self.templates.get((state["agent_id"], state["camera_name"]), [])
        if not templates: return False
        box = state["bbox_2d"]; cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        bw, bh = max(4.0, box[2] - box[0]), max(4.0, box[3] - box[1]); radius = float(self.config["reid_search_expansion"])
        search = clipped_box([cx - bw * radius, cy - bh * radius, cx + bw * radius, cy + bh * radius], gray.shape[1], gray.shape[0])
        sx1, sy1, sx2, sy2 = map(int, search); crop = gray[sy1:sy2 + 1, sx1:sx2 + 1]
        best = None
        for template in templates[-3:]:
            if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]: continue
            result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED); _, score, _, location = cv2.minMaxLoc(result)
            if best is None or score > best[0]: best = (float(score), location, template.shape)
        if best is None or best[0] < float(self.config["reid_template_threshold"]): return False
        score, (x, y), shape = best; h, w = shape; state["bbox_2d"] = [sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h]
        state["mask_polygon"] = bbox_polygon(state["bbox_2d"]); state["points"] = sample_mask_points(gray, state["mask_polygon"], int(self.config["maximum_points"]))
        state["embedding"] = appearance_histogram(image, state["bbox_2d"]); state["status"] = "reidentified"; state["lost_frames"] = 0
        state["fragmentation"] += 1; state["reidentified"] += 1; self.reidentification_successes += 1
        return True

    def _advance(self, state: dict[str, Any], previous: np.ndarray, current: np.ndarray, image: np.ndarray, frame: int,
                 timestamp: float, dense_flow: np.ndarray | None = None) -> dict[str, Any]:
        dense, dense_vectors = self._dense_translation(previous, current, state["mask_polygon"], dense_flow)
        old, new, point_vectors, fb_error = self._lk(previous, current, state["points"])
        minimum = int(self.config["minimum_surviving_points"]); enough = len(new) >= minimum or (self.mode in {"box", "mask"} and np.isfinite(dense).all())
        if enough:
            box, polygon = self._warp(state, old, new, dense, image.shape[1], image.shape[0])
            embedding = appearance_histogram(image, box); similarity = histogram_similarity(state["anchor_embedding"], embedding)
            if similarity >= float(self.config["minimum_appearance_similarity"]):
                was_lost = state["lost_frames"] > 0; state.update(bbox_2d=box, mask_polygon=polygon, points=new,
                    embedding=embedding, status="reidentified" if was_lost else "tracked", lost_frames=0,
                    last_frame=frame, continuous_length=state["continuous_length"] + 1)
                if len(state["points"]) < int(self.config["reseed_below_points"]):
                    state["points"] = sample_mask_points(current, polygon, int(self.config["maximum_points"]))
                if was_lost: state["fragmentation"] += 1; state["reidentified"] += 1
            else: enough = False
        if not enough:
            state["lost_frames"] += 1; state["status"] = "temporarily_lost" if state["lost_frames"] <= int(self.config["maximum_lost_frames"]) else "terminated"
            if state["status"] == "temporarily_lost": self._reidentify(state, current, image)
        box = state["bbox_2d"]; center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        point_survival = len(new) / max(1, len(old)) if len(old) else 0.0
        confidence = max(0.0, min(1.0, point_survival * histogram_similarity(state["anchor_embedding"], state["embedding"]))) if state["status"] != "terminated" else 0.0
        return {"agent_id": state["agent_id"], "camera_name": state["camera_name"], "frame_index": frame, "timestamp": timestamp,
                "visible": state["status"] in {"tracked", "reidentified"}, "track_status": state["status"], "mask_polygon": state["mask_polygon"],
                "mask_source": "history_projection_propagated_by_opencv_flow", "bbox_2d": box, "center_2d": center,
                "tracked_points": state["points"].astype(float).tolist(), "appearance_embedding": state["embedding"], "confidence": confidence,
                "appearance_similarity": histogram_similarity(state["anchor_embedding"], state["embedding"]), "occlusion_score": NOT_AVAILABLE,
                "flow_vectors": point_vectors if point_vectors else dense_vectors, "median_flow": (np.median(point_vectors, axis=0).tolist() if point_vectors else dense.tolist()),
                "point_survival": point_survival, "forward_backward_error": fb_error if np.isfinite(fb_error) else NOT_AVAILABLE,
                "fragmentation": state["fragmentation"], "lost_frames": state["lost_frames"]}

    def initialize(self, history_images: dict[str, list[str]], projected_history_boxes: list[dict[str, Any]],
                   history_agent_ids: list[str], calibrations: list[dict[str, Any]], ego_poses: list[dict[str, Any]],
                   timestamps: list[float]) -> dict[str, Any]:
        del history_agent_ids, calibrations, ego_poses
        by_frame_camera: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in projected_history_boxes:
            if row.get("expected_visible"): by_frame_camera[(int(row["frame_index"]), row["camera_name"])].append(row)
        errors, flow_errors, ious, survival, expected, tracked = [], [], [], [], 0, 0
        for camera in CAMERAS:
            previous = None
            for frame, path in enumerate(history_images[camera]):
                image = self._prepare(read_image_bgr(path)); gray = self._gray(image)
                dense_flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0) if previous is not None else None
                for row in by_frame_camera.get((frame, camera), []):
                    expected += 1; key = (row["agent_id"], camera)
                    if key in self.states and previous is not None and self.states[key]["last_frame"] == frame - 1:
                        observation = self._advance(self.states[key], previous, gray, image, frame, timestamps[frame], dense_flow)
                        tracked += int(observation["visible"]); errors.append(float(np.linalg.norm(np.asarray(observation["center_2d"]) - row["center_2d"])))
                        ious.append(box_iou(observation["bbox_2d"], row["clipped_box_2d"])); survival.append(observation["point_survival"])
                        previous_rows = by_frame_camera.get((frame - 1, camera), [])
                        prior = next((value for value in previous_rows if value["agent_id"] == row["agent_id"]), None)
                        if prior and observation["flow_vectors"]:
                            actual = np.median(np.asarray(observation["flow_vectors"]), axis=0); expected_flow = np.asarray(row["center_2d"]) - prior["center_2d"]
                            flow_errors.append(float(np.linalg.norm(actual - expected_flow)))
                        self.history_observations.append({**observation, "expected_box_2d": row["clipped_box_2d"], "expected_center_2d": row["center_2d"]})
                    self.states[key] = self._seed(row["agent_id"], camera, image, row)
                previous = gray; self.previous_gray[camera] = gray
        last_history_frame = len(timestamps) - 1
        for state in self.states.values():
            if state["last_frame"] != last_history_frame:
                state["status"] = "temporarily_lost"; state["lost_frames"] = last_history_frame - state["last_frame"]
                state["points"] = np.empty((0, 2), np.float32)
        camera_support = defaultdict(lambda: [0, 0])
        for row in projected_history_boxes:
            if row.get("expected_visible"): camera_support[(row["agent_id"], row["frame_index"])][1] += 1
        for row in self.history_observations:
            if row["visible"]: camera_support[(row["agent_id"], row["frame_index"])][0] += 1
        return {"history_frames_used": len(timestamps), "history_projected_visible_agent_frames": expected,
                "history_mask_stability": mean(ious), "history_point_survival": mean(survival), "history_center_tracking_error_px": mean(errors),
                "history_flow_error_px": mean(flow_errors), "history_identity_persistence": tracked / expected if expected else NOT_AVAILABLE,
                "history_cross_view_consistency": mean(actual / expected_count for actual, expected_count in camera_support.values() if expected_count),
                "history_agent_count": len({row["agent_id"] for row in projected_history_boxes}), "mask_backend": "projected_box_mask_plus_dense_flow",
                "point_backend": "opencv_pyramidal_lucas_kanade", "flow_backend": "opencv_farneback"}

    def track_future_frame(self, frame_images: dict[str, np.ndarray], calibrations: list[dict[str, Any]],
                           ego_pose: dict[str, Any], timestamp: float) -> list[dict[str, Any]]:
        del calibrations, ego_pose; start = time.perf_counter(); self.frame_index += 1; rows = []
        for camera in CAMERAS:
            image = self._prepare(frame_images[camera]); current = self._gray(image); previous = self.previous_gray[camera]
            dense_flow = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
            for (agent_id, state_camera), state in list(self.states.items()):
                if state_camera != camera or state["status"] == "terminated": continue
                rows.append(self._advance(state, previous, current, image, self.frame_index, timestamp, dense_flow))
            self.previous_gray[camera] = current
        self.observations.extend(rows); self.future_frames += 1; self.latencies.append(time.perf_counter() - start); return rows

    def finalize(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, "backend": "opencv_lk_farneback_ransac", "mode": self.mode,
                "observations": self.observations, "history_observations": self.history_observations,
                "state_count": len(self.states), "future_frames": self.future_frames, "mean_frame_seconds": mean(self.latencies),
                "total_seconds": float(sum(self.latencies)), "gpu_used": False, "peak_gpu_memory_mib": 0.0,
                "reidentification_attempts": self.reidentification_attempts, "reidentification_successes": self.reidentification_successes,
                "future_submission_trajectory_accessed": False, "future_gt_accessed": False,
                "checkpoint": NOT_AVAILABLE, "random_initialization_used": False}

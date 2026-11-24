from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.agents.core import angle_delta

from driveaudit_eval.consistency.shared.categories import semantic_category
from driveaudit_eval.consistency.shared.common import rotated_iou


@dataclass
class Track:
    track_id: str
    category: str
    state: np.ndarray
    covariance: np.ndarray
    box: dict[str, float]
    last_timestamp: float
    confirmed: bool = False
    hits: int = 1
    consecutive_hits: int = 1
    lost_frames: int = 0
    fragmentation: int = 0
    lifecycle_state: str = "tentative"
    observations: list[dict[str, Any]] = field(default_factory=list)
    history_agent_id: str | None = None


class BEVTracker:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.reset("")

    def reset(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self.tracks: dict[str, Track] = {}
        self.finished: dict[str, Track] = {}
        self.counter = 0
        self.diagnostics = {"births": 0, "recoveries": 0, "terminated": 0, "duplicates_suppressed": 0}

    def initialize_history(self, samples: list[dict[str, Any]], final_frame_index: int) -> int:
        """Seed only agents present on the last history frame; earlier frames provide state history."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(str(sample["agent_id"]), []).append(sample)
        for agent_id, history in grouped.items():
            history.sort(key=lambda row: row["timestamp"])
            if int(history[-1]["frame_index"]) != final_frame_index:
                continue
            last = history[-1]
            box = dict(last["box_bev"])
            state = np.array([box["x"], box["y"], last["velocity"][0], last["velocity"][1]], dtype=float)
            track_id = f"hist:{agent_id}"
            track = Track(track_id, last["category"], state, np.eye(4), box, float(last["timestamp"]),
                          confirmed=True, hits=len(history), consecutive_hits=len(history),
                          lifecycle_state="confirmed", history_agent_id=agent_id)
            track.observations = [dict(row, source="history_known") for row in history]
            self.tracks[track_id] = track
        return len(self.tracks)

    def _predict(self, track: Track, timestamp: float) -> None:
        dt = max(0.0, float(timestamp) - track.last_timestamp)
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = float(self.config["process_noise"]) * max(dt, 1e-3)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + np.eye(4) * q
        track.box["x"], track.box["y"] = map(float, track.state[:2])
        track.last_timestamp = float(timestamp)

    def _same_class(self, a: str, b: str) -> bool:
        return semantic_category(a) == semantic_category(b) if self.config.get("semantic_class_gate", False) else a == b

    def _distance_gate(self, category: str) -> float:
        by_type = self.config.get("gates", {}).get("center_distance_m_by_type", {})
        return float(by_type.get(semantic_category(category), self.config["gates"]["center_distance_m"]))

    def _deduplicate(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        threshold = float(self.config.get("duplicate_nms_iou", 1.1))
        if threshold > 1 or len(detections) < 2:
            return detections
        kept = []
        for detection in sorted(detections, key=lambda row: float(row["score"]), reverse=True):
            if any(self._same_class(detection["category"], other["category"])
                   and rotated_iou(detection["box_bev"], other["box_bev"]) >= threshold for other in kept):
                self.diagnostics["duplicates_suppressed"] += 1
            else:
                kept.append(detection)
        return kept

    def _cost(self, track: Track, detection: dict[str, Any]) -> float:
        box = detection["box_bev"]
        gates, weights = self.config["gates"], self.config["weights"]
        distance = float(np.linalg.norm(track.state[:2] - [box["x"], box["y"]]))
        heading = abs(angle_delta(float(track.box["yaw"]), float(box["yaw"])))
        size_ratio = max(track.box["length"] / box["length"], box["length"] / track.box["length"],
                         track.box["width"] / box["width"], box["width"] / track.box["width"])
        iou = rotated_iou(track.box, box)
        distance_gate = self._distance_gate(detection["category"])
        if (not self._same_class(track.category, detection["category"]) or distance > distance_gate or
                heading > gates["heading_rad"] or size_ratio > gates["size_ratio"] or iou < gates["minimum_iou"]):
            return 1e6
        velocity = float(np.linalg.norm(track.state[2:] - np.asarray(detection["velocity"], dtype=float)))
        size = abs(math.log(size_ratio))
        return (weights["position"] * distance / distance_gate + weights["iou"] * (1 - iou) +
                weights["heading"] * heading / gates["heading_rad"] + weights["velocity"] * min(velocity / 10.0, 1.0) +
                weights["size"] * size)

    def update(self, detections: list[dict[str, Any]], timestamp: float, frame_index: int) -> list[dict[str, Any]]:
        detections = self._deduplicate(detections)
        tracks = list(self.tracks.values())
        for track in tracks:
            self._predict(track, timestamp)
        assignments: list[tuple[int, int]] = []
        if tracks and detections:
            matrix = np.array([[self._cost(track, detection) for detection in detections] for track in tracks])
            assignments = [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(matrix)) if matrix[i, j] < 1e5]
        matched_tracks, matched_detections = {i for i, _ in assignments}, {j for _, j in assignments}
        output = []
        for track_index, detection_index in assignments:
            track, detection = tracks[track_index], detections[detection_index]
            if track.lost_frames:
                track.fragmentation += 1
                self.diagnostics["recoveries"] += 1
            measurement = np.array([detection["box_bev"]["x"], detection["box_bev"]["y"],
                                    detection["velocity"][0], detection["velocity"][1]], dtype=float)
            noise = np.diag([float(self.config["measurement_noise"])] * 2 +
                            [float(self.config.get("velocity_measurement_noise", self.config["measurement_noise"]))] * 2)
            gain = track.covariance @ np.linalg.inv(track.covariance + noise)
            track.state += gain @ (measurement - track.state)
            track.covariance = (np.eye(4) - gain) @ track.covariance
            track.box = dict(detection["box_bev"])
            track.last_timestamp = float(timestamp)
            track.hits += 1
            track.consecutive_hits += 1
            track.lost_frames = 0
            track.confirmed = track.confirmed or track.consecutive_hits >= int(self.config["min_confirmed_frames"])
            track.lifecycle_state = "confirmed" if track.confirmed else "tentative"
            observation = {"frame_index": frame_index, "timestamp": float(timestamp), "box_bev": dict(track.box),
                           "velocity": list(detection["velocity"]), "score": detection["score"],
                           "native_velocity": list(detection["velocity"]),
                           "kalman_velocity": track.state[2:].astype(float).tolist(),
                           "camera_support": detection.get("camera_support", []), "source": "image_detection"}
            track.observations.append(observation)
            output.append({**observation, "image_track_id": track.track_id, "category": track.category})
        for index, track in enumerate(tracks):
            if index not in matched_tracks:
                track.lost_frames += 1
                track.consecutive_hits = 0
                track.lifecycle_state = "lost"
                maximum_lost = int(self.config["max_lost_frames"] if track.confirmed else
                                   self.config.get("tentative_max_lost_frames", self.config["max_lost_frames"]))
                if track.lost_frames > maximum_lost:
                    track.lifecycle_state = "terminated"
                    self.diagnostics["terminated"] += 1
                    self.finished[track.track_id] = self.tracks.pop(track.track_id)
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            self.counter += 1
            box = dict(detection["box_bev"])
            track_id = f"track:{self.counter:04d}"
            state = np.array([box["x"], box["y"], *detection["velocity"]], dtype=float)
            observation = {"frame_index": frame_index, "timestamp": float(timestamp), "box_bev": box,
                           "velocity": list(detection["velocity"]), "score": detection["score"],
                           "native_velocity": list(detection["velocity"]),
                           "kalman_velocity": state[2:].astype(float).tolist(),
                           "camera_support": detection.get("camera_support", []), "source": "image_detection"}
            confirmed = int(self.config["min_confirmed_frames"]) <= 1
            self.tracks[track_id] = Track(track_id, detection["category"], state, np.eye(4), box,
                                          float(timestamp), confirmed=confirmed,
                                          lifecycle_state="confirmed" if confirmed else "tentative",
                                          observations=[observation])
            self.diagnostics["births"] += 1
            output.append({**observation, "image_track_id": track_id, "category": detection["category"]})
        return output

    def export(self, confirmed_only: bool = False) -> list[dict[str, Any]]:
        result = []
        for track in [*self.finished.values(), *self.tracks.values()]:
            if confirmed_only and not track.confirmed:
                continue
            future = [row for row in track.observations if row.get("source") == "image_detection"]
            result.append({"image_track_id": track.track_id, "history_agent_id": track.history_agent_id,
                           "category": track.category, "agent_type": semantic_category(track.category),
                           "confirmed": track.confirmed, "lifecycle_state": track.lifecycle_state,
                           "track_length": len(future),
                           "fragmentation": track.fragmentation, "lost_duration_frames": track.lost_frames,
                           "mean_confidence": float(np.mean([row["score"] for row in future])) if future else "not_available",
                           "camera_support_count": int(sum(len(row.get("camera_support", [])) for row in future)),
                           "observations": track.observations})
        return sorted(result, key=lambda row: row["image_track_id"])

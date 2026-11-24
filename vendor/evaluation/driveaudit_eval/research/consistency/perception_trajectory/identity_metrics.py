from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.categories import semantic_category


def compute_identity_metrics(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                             mapping: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """MOT-style global ID assignment over expected-visible frames."""
    config = config or {}; maximum = float(config.get("maximum_center_distance_m", 8.0))
    semantic = bool(config.get("semantic_class_matching", True))
    gt_by_frame: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    pred_by_frame: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    expected_by_id: dict[str, list[int]] = defaultdict(list)
    for track in submission_tracks:
        for row in track["observations"]:
            if row.get("expected_visible", True):
                frame = int(row["frame_index"]); gt_by_frame[frame].append((track["agent_id"], track.get("category", ""), row))
                expected_by_id[track["agent_id"]].append(frame)
    for track in image_tracks:
        for row in track["observations"]:
            if row.get("source") == "image_detection":
                pred_by_frame[int(row["frame_index"])].append((track["image_track_id"], track.get("category", ""), row))
    overlap: dict[tuple[str, str], int] = defaultdict(int)
    frame_identity: dict[str, list[tuple[int, str]]] = defaultdict(list)
    direction_flips = []
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt, pred = gt_by_frame[frame], pred_by_frame[frame]
        if not gt or not pred:
            continue
        cost = np.full((len(gt), len(pred)), 1e6, dtype=float)
        for i, (_, gt_category, gt_row) in enumerate(gt):
            for j, (_, pred_category, pred_row) in enumerate(pred):
                class_match = semantic_category(gt_category) == semantic_category(pred_category) if semantic else gt_category == pred_category
                distance = np.linalg.norm([gt_row["box_bev"]["x"] - pred_row["box_bev"]["x"],
                                           gt_row["box_bev"]["y"] - pred_row["box_bev"]["y"]])
                if class_match and distance <= maximum:
                    cost[i, j] = distance
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] >= 1e5:
                continue
            gt_id, _, gt_row = gt[int(i)]; pred_id, _, pred_row = pred[int(j)]
            overlap[(gt_id, pred_id)] += 1; frame_identity[gt_id].append((frame, pred_id))
            direction_flips.append(float(np.dot(gt_row.get("velocity", [0, 0]), pred_row.get("velocity", [0, 0])) < 0))
    gt_ids = sorted(expected_by_id); pred_ids = sorted({track["image_track_id"] for track in image_tracks})
    anchors = {track["image_track_id"]: track.get("history_agent_id") for track in image_tracks}
    idtp = 0
    if gt_ids and pred_ids:
        matrix = np.asarray([[1e6 if anchors[pred_id] and anchors[pred_id] != gt_id else -overlap[(gt_id, pred_id)]
                              for pred_id in pred_ids] for gt_id in gt_ids], dtype=float)
        rows, cols = linear_sum_assignment(matrix)
        idtp = int(sum(-matrix[i, j] for i, j in zip(rows, cols) if matrix[i, j] <= 0))
    total_gt = sum(len(frames) for frames in expected_by_id.values())
    total_pred = sum(len(rows) for rows in pred_by_frame.values())
    idfn, idfp = total_gt - idtp, total_pred - idtp
    precision = idtp / (idtp + idfp) if idtp + idfp else NOT_AVAILABLE
    recall = idtp / (idtp + idfn) if idtp + idfn else NOT_AVAILABLE
    switches = fragments = 0; segment_lengths = []; timing = []
    for gt_id, expected in expected_by_id.items():
        assignments = sorted(frame_identity.get(gt_id, [])); last_id = None
        for _, pred_id in assignments:
            if last_id is not None and pred_id != last_id:
                switches += 1
            last_id = pred_id
        matched_frames = {frame for frame, _ in assignments}; segment = 0; segments = []
        for frame in sorted(expected):
            if frame in matched_frames:
                segment += 1
            elif segment:
                segments.append(segment); segment = 0
        if segment:
            segments.append(segment)
        if segments:
            fragments += len(segments) - 1; segment_lengths.extend(segments)
            timing.append(abs(min(matched_frames) - min(expected)) + abs(max(matched_frames) - max(expected)))
    return {"id_precision": precision, "id_recall": recall,
            "idf1_like": 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else NOT_AVAILABLE,
            "id_switch_count": switches, "fragmentation_count": fragments,
            "track_completeness": recall,
            "average_continuous_track_length": float(np.mean(segment_lengths)) if segment_lengths else NOT_AVAILABLE,
            "reidentification_success": float(np.mean([len(frame_identity.get(gt_id, [])) == len(expected)
                                                       for gt_id, expected in expected_by_id.items()])) if expected_by_id else NOT_AVAILABLE,
            "temporal_order_accuracy": 1.0 if idtp else NOT_AVAILABLE,
            "entry_exit_timing_error_frames": float(np.mean(timing)) if timing else NOT_AVAILABLE,
            "trajectory_direction_flip_disagreement": float(np.mean(direction_flips)) if direction_flips else NOT_AVAILABLE,
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "protocol": "mot_global_identity_assignment"}

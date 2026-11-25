from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.agents.core import angle_delta

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.categories import semantic_category
from driveaudit_eval.consistency.shared.common import percentile_summary, rotated_iou


def compute_detection_metrics(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                              config: dict[str, Any]) -> dict[str, Any]:
    ground_truth: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    predictions: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for track in submission_tracks:
        for row in track["observations"]:
            if row.get("expected_visible", False): ground_truth[row["frame_index"]].append((track["agent_id"], track["category"], row))
    for track in image_tracks:
        for row in track["observations"]:
            if row.get("source") == "image_detection": predictions[row["frame_index"]].append((track["image_track_id"], track["category"], row))
    tp = fp = fn = 0; centers, ious, headings, lengths, widths, classes = [], [], [], [], [], []
    rows = []
    for frame in sorted(set(ground_truth) | set(predictions)):
        gt, pred = ground_truth[frame], predictions[frame]
        matched = []
        if gt and pred:
            matrix = np.full((len(gt), len(pred)), 1e6, dtype=float)
            semantic = bool(config.get("semantic_class_matching", False))
            for i, a in enumerate(gt):
                for j, b in enumerate(pred):
                    class_match = semantic_category(a[1]) == semantic_category(b[1]) if semantic else a[1] == b[1]
                    distance = np.linalg.norm([a[2]["box_bev"]["x"] - b[2]["box_bev"]["x"],
                                               a[2]["box_bev"]["y"] - b[2]["box_bev"]["y"]])
                    if class_match and distance <= float(config["maximum_center_distance_m"]) and \
                            rotated_iou(a[2]["box_bev"], b[2]["box_bev"]) >= float(config["minimum_rotated_iou"]):
                        matrix[i, j] = distance
            matched = [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(matrix))
                       if matrix[i, j] < 1e5]
        used_gt, used_pred = {i for i, _ in matched}, {j for _, j in matched}
        for i, j in matched:
            agent_id, gt_class, a = gt[i]; image_id, pred_class, b = pred[j]
            ab, bb = a["box_bev"], b["box_bev"]
            center = float(np.linalg.norm([ab["x"] - bb["x"], ab["y"] - bb["y"]]))
            iou = rotated_iou(ab, bb)
            tp += 1; centers.append(center); ious.append(iou); headings.append(abs(angle_delta(ab["yaw"], bb["yaw"])))
            lengths.append(abs(ab["length"] - bb["length"])); widths.append(abs(ab["width"] - bb["width"])); classes.append(gt_class == pred_class)
            rows.append({"frame_index": frame, "timestamp": a["timestamp"], "submission_agent_id": agent_id,
                         "image_track_id": image_id, "status": "matched", "center_error": center, "rotated_bev_iou": iou,
                         "heading_error": headings[-1], "length_error": lengths[-1], "width_error": widths[-1],
                         "class_correct": classes[-1]})
        for i, (agent_id, _, a) in enumerate(gt):
            if i not in used_gt:
                fn += 1; rows.append({"frame_index": frame, "timestamp": a["timestamp"], "submission_agent_id": agent_id,
                                      "image_track_id": None, "status": "image_trajectory_missing"})
        for j, (image_id, _, b) in enumerate(pred):
            if j not in used_pred:
                fp += 1; rows.append({"frame_index": frame, "timestamp": b["timestamp"], "submission_agent_id": None,
                                      "image_track_id": image_id, "status": "unmatched_image_detection"})
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "visible_detection_recall": tp / (tp + fn) if tp + fn else NOT_AVAILABLE,
            "visible_detection_precision": tp / (tp + fp) if tp + fp else NOT_AVAILABLE,
            "visible_detection_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else NOT_AVAILABLE,
            "center_error": percentile_summary(centers), "rotated_bev_iou": float(np.mean(ious)) if ious else NOT_AVAILABLE,
            "heading_mae": float(np.mean(headings)) if headings else NOT_AVAILABLE,
            "box_length_mae": float(np.mean(lengths)) if lengths else NOT_AVAILABLE,
            "box_width_mae": float(np.mean(widths)) if widths else NOT_AVAILABLE,
            "class_accuracy": float(np.mean(classes)) if classes else NOT_AVAILABLE, "per_frame_rows": rows}

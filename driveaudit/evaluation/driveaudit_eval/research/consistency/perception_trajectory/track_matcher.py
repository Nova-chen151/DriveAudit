from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.agents.core import angle_delta

from driveaudit_eval.consistency.shared.categories import semantic_category
from driveaudit_eval.consistency.shared.common import rotated_iou


def _future_observations(track: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in track["observations"] if row.get("source") == "image_detection"]


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return math.inf
    cost = np.full((len(a) + 1, len(b) + 1), np.inf); cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1] / max(len(a), len(b)))


def _track_cost(submission: dict[str, Any], image: dict[str, Any], config: dict[str, Any]) -> tuple[float, int]:
    if ((semantic_category(submission["category"]) != semantic_category(image["category"]))
            if config.get("semantic_class_matching", False) else submission["category"] != image["category"]):
        return 1e6, 0
    submitted = {row["frame_index"]: row for row in submission["observations"] if row.get("expected_visible", True)}
    extracted = {row["frame_index"]: row for row in _future_observations(image)}
    common = sorted(set(submitted) & set(extracted))
    if len(common) < int(config["minimum_common_visible_frames"]):
        return 1e6, len(common)
    centers, ious, headings, velocities, sizes = [], [], [], [], []
    for frame in common:
        a, b = submitted[frame], extracted[frame]
        centers.append(np.linalg.norm([a["box_bev"]["x"] - b["box_bev"]["x"], a["box_bev"]["y"] - b["box_bev"]["y"]]))
        ious.append(rotated_iou(a["box_bev"], b["box_bev"]))
        headings.append(abs(angle_delta(a["box_bev"]["yaw"], b["box_bev"]["yaw"])))
        velocities.append(np.linalg.norm(np.asarray(a.get("velocity", [0, 0])) - np.asarray(b.get("velocity", [0, 0]))))
        sizes.append(abs(math.log(a["box_bev"]["length"] / b["box_bev"]["length"])) +
                     abs(math.log(a["box_bev"]["width"] / b["box_bev"]["width"])))
    if np.mean(centers) > float(config["maximum_mean_center_distance_m"]):
        return 1e6, len(common)
    sxy = np.asarray([[submitted[i]["box_bev"]["x"], submitted[i]["box_bev"]["y"]] for i in sorted(submitted)])
    ixy = np.asarray([[extracted[i]["box_bev"]["x"], extracted[i]["box_bev"]["y"]] for i in sorted(extracted)])
    entry = abs(min(submitted) - min(extracted)); exit_ = abs(max(submitted) - max(extracted))
    w = config["weights"]
    cost = (w["center"] * np.mean(centers) / 10.0 + w["iou"] * (1 - np.mean(ious)) +
            w["heading"] * np.mean(headings) / math.pi + w["velocity"] * min(np.mean(velocities) / 10.0, 1) +
            w["size"] * np.mean(sizes) + w["entry_time"] * min(entry / 10.0, 1) +
            w["exit_time"] * min(exit_ / 10.0, 1) + w["dtw"] * min(_dtw(sxy, ixy) / 10.0, 1))
    return float(cost), len(common)


def match_tracks(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                 config: dict[str, Any]) -> list[dict[str, Any]]:
    """One scene-level assignment. It never remaps identities frame by frame."""
    mapping, used_submission, used_image = [], set(), set()
    by_submission = {track["agent_id"]: i for i, track in enumerate(submission_tracks)}
    for j, image in enumerate(image_tracks):
        anchor = image.get("history_agent_id")
        if anchor in by_submission:
            i = by_submission[anchor]
            cost, common = _track_cost(submission_tracks[i], image, config)
            status = "matched_history_anchor" if cost <= float(config.get("history_anchor_maximum_cost", config["maximum_cost"])) else "ambiguous_match"
            mapping.append({"submission_agent_id": anchor, "image_track_id": image["image_track_id"],
                            "match_cost": cost, "match_confidence": max(0.0, 1.0 - cost),
                            "common_visible_frames": common, "match_status": status})
            used_submission.add(i); used_image.add(j)
    remaining_s = [i for i in range(len(submission_tracks)) if i not in used_submission]
    remaining_i = [j for j in range(len(image_tracks)) if j not in used_image]
    if remaining_s and remaining_i:
        details = [[_track_cost(submission_tracks[i], image_tracks[j], config) for j in remaining_i] for i in remaining_s]
        matrix = np.asarray([[item[0] for item in row] for row in details])
        for row, col in zip(*linear_sum_assignment(matrix)):
            i, j = remaining_s[int(row)], remaining_i[int(col)]
            if matrix[row, col] > float(config["maximum_cost"]):
                continue
            cost, common = details[int(row)][int(col)]
            mapping.append({"submission_agent_id": submission_tracks[i]["agent_id"],
                            "image_track_id": image_tracks[j]["image_track_id"], "match_cost": cost,
                            "match_confidence": max(0.0, 1.0 - cost / float(config["maximum_cost"])),
                            "common_visible_frames": common,
                            "match_status": "ambiguous_match" if submission_tracks[i].get("history_present") else "matched_future_entry"})
            used_submission.add(i); used_image.add(j)
    for i, track in enumerate(submission_tracks):
        if i not in used_submission:
            status = "not_observable" if not any(row.get("expected_visible", True) for row in track["observations"]) else "unmatched_submission_agent"
            mapping.append({"submission_agent_id": track["agent_id"], "image_track_id": None, "match_cost": "not_available",
                            "match_confidence": "not_available", "common_visible_frames": 0, "match_status": status})
    for j, track in enumerate(image_tracks):
        if j not in used_image:
            mapping.append({"submission_agent_id": None, "image_track_id": track["image_track_id"], "match_cost": "not_available",
                            "match_confidence": "not_available", "common_visible_frames": 0, "match_status": "unmatched_image_track"})
    return mapping

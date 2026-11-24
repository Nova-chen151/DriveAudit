from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory.bev_geometry_metrics import compute_bev_geometry
from driveaudit_eval.research.consistency.perception_trajectory.identity_metrics import compute_identity_metrics
from driveaudit_eval.research.consistency.perception_trajectory.motion_metrics import compute_motion_metrics
from driveaudit_eval.research.consistency.perception_trajectory.presence_metrics import compute_presence_metrics
from driveaudit_eval.research.consistency.perception_trajectory.time_alignment import diagnose_time_shift
from driveaudit_eval.research.consistency.perception_trajectory.track_matcher import match_tracks
from driveaudit_eval.consistency.shared.common import write_json


LEVELS = {"mild": 0.5, "medium": 1.5, "severe": 4.0}


def ideal_tracks(frames: int = 20, dt: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    submitted, images = [], []
    for index, (agent_id, lateral) in enumerate((("A", 1.0), ("B", -3.0), ("C", 5.0), ("D", -7.0))):
        observations = []
        for frame in range(frames):
            x = frame * (1.0 + index * .2); y = lateral + .05 * frame * frame / frames
            yaw = .1 + .015 * frame
            observations.append({"frame_index": frame, "timestamp": frame * dt, "expected_visible": True,
                                 "box_bev": {"x": x, "y": y, "z": 0.0, "height": 1.6,
                                             "length": 4.5, "width": 1.8, "yaw": yaw},
                                 "velocity": [(1.0 + index * .2) / dt, .1 * frame / frames / dt]})
        submitted.append({"agent_id": agent_id, "category": "car", "history_present": True,
                          "observations": copy.deepcopy(observations)})
        images.append({"image_track_id": f"hist:{agent_id}", "history_agent_id": agent_id, "category": "car",
                       "fragmentation": 0, "observations": [{**copy.deepcopy(row), "source": "image_detection", "score": 1.0}
                                                             for row in observations]})
    return submitted, images


def corrupt(tracks: list[dict[str, Any]], kind: str, strength: float) -> list[dict[str, Any]]:
    result = copy.deepcopy(tracks)
    if kind == "missing_agent": return result[:-min(len(result) - 1, max(1, int(round(strength))))]
    if kind == "duplicated_agent":
        for index in range(max(1, int(round(strength)))):
            duplicate = copy.deepcopy(result[index % len(result)]); duplicate["agent_id"] += f"_duplicate_{index}"; result.append(duplicate)
        return result
    if kind == "identity_swap":
        result[0]["observations"], result[1]["observations"] = result[1]["observations"], result[0]["observations"]; return result
    for track in result:
        rows = track["observations"]
        if kind == "early_disappearance": del rows[max(1, len(rows) - int(2 * strength)):]
        elif kind == "late_appearance": del rows[:min(len(rows) - 1, int(2 * strength))]
        for row in rows:
            box = row["box_bev"]
            if kind == "global_x_shift": box["x"] += strength
            elif kind == "global_y_shift": box["y"] += strength
            elif kind == "lateral_trajectory_shift": box["y"] += strength
            elif kind == "yaw_sign_error": box["yaw"] *= max(-1.0, 1.0 - strength)
            elif kind == "xy_swap": box["x"], box["y"] = box["y"], box["x"]
            elif kind == "time_shift":
                shift = max(1, int(round(strength))); row["frame_index"] += shift; row["timestamp"] += shift * .1
            elif kind == "speed_scaling": box["x"] *= 1 + .1 * strength
            elif kind == "wrong_box_size": box["length"] *= 1 + .2 * strength; box["width"] *= 1 + .2 * strength
            elif kind == "lane_change_timing_shift": box["y"] += .3 * strength * math.sin(max(0, row["frame_index"] - 5) / 5)
    return result


def evaluate_pair(submission: list[dict[str, Any]], images: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    mapping = match_tracks(submission, images, config)
    return {"presence": compute_presence_metrics(submission, images, mapping),
            "geometry": compute_bev_geometry(submission, images, mapping, 0.0),
            "motion": compute_motion_metrics(submission, images, mapping),
            "identity": compute_identity_metrics(submission, images, mapping), "mapping": mapping}


def run_corruption_benchmark(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    submission, images = ideal_tracks()
    self_consistency = evaluate_pair(submission, images, config)
    self_test = {"passed": self_consistency["geometry"]["bev_center_ade"] < 1e-9 and
                          self_consistency["geometry"]["visible_fde"] < 1e-9 and
                          self_consistency["presence"]["visible_agent_recall"] == 1.0 and
                          self_consistency["identity"]["idf1_like"] == 1.0,
                 "bev_ade": self_consistency["geometry"]["bev_center_ade"],
                 "visible_fde": self_consistency["geometry"]["visible_fde"],
                 "heading_mae": self_consistency["geometry"]["heading_mae"],
                 "speed_mae": self_consistency["motion"]["speed_mae"],
                 "visible_agent_recall": self_consistency["presence"]["visible_agent_recall"],
                 "idf1": self_consistency["identity"]["idf1_like"]}
    kinds = ["global_x_shift", "global_y_shift", "yaw_sign_error", "xy_swap", "time_shift", "speed_scaling",
             "identity_swap", "missing_agent", "duplicated_agent", "early_disappearance", "late_appearance",
             "lateral_trajectory_shift", "wrong_box_size", "lane_change_timing_shift"]
    rows = []
    for kind in kinds:
        for level, strength in LEVELS.items():
            metrics = evaluate_pair(corrupt(submission, kind, strength), images, config)
            rows.append({"corruption": kind, "level": level, "strength": strength,
                         "bev_ade": metrics["geometry"]["bev_center_ade"],
                         "box_length_mae": metrics["geometry"]["box_length_mae"],
                         "heading_mae": metrics["geometry"]["heading_mae"],
                         "speed_mae": metrics["motion"]["speed_mae"],
                         "visible_recall": metrics["presence"]["visible_agent_recall"],
                         "unmatched_submission_ratio": metrics["presence"]["unmatched_submission_ratio"],
                         "unmatched_image_track_ratio": metrics["presence"]["unmatched_image_track_ratio"],
                         "id_switch_count": metrics["identity"]["id_switch_count"]})
    targets = {"global_x_shift": "bev_ade", "global_y_shift": "bev_ade", "lateral_trajectory_shift": "bev_ade",
               "wrong_box_size": "box_length_mae", "yaw_sign_error": "heading_mae", "time_shift": "bev_ade",
               "speed_scaling": "speed_mae", "missing_agent": "unmatched_image_track_ratio",
               "duplicated_agent": "unmatched_submission_ratio"}
    monotonic = {}
    for kind, metric in targets.items():
        values = [row[metric] for row in rows if row["corruption"] == kind]
        monotonic[kind] = bool(all(isinstance(value, (float, int)) for value in values) and values == sorted(values))
    stability = {"position_shift_preserves_heading": all(row["heading_mae"] < 1e-9 for row in rows if row["corruption"] in {"global_x_shift", "global_y_shift"}),
                 "yaw_error_preserves_centers": all(row["bev_ade"] < 1e-9 for row in rows if row["corruption"] == "yaw_sign_error")}
    result = {"protocol": "ideal_BEV_observation_fixed_images", "self_consistency": self_test,
              "corruption_types": len(kinds), "levels_per_type": len(LEVELS), "results": rows,
              "monotonic_checks": monotonic, "passed_monotonic_checks": sum(monotonic.values()),
              "unrelated_metric_stability_checks": stability}
    write_json(output / "results.json", result); write_json(output / "self_consistency.json", self_test)
    fig, axis = plt.subplots(figsize=(7, 4))
    for kind in ("global_x_shift", "global_y_shift", "lateral_trajectory_shift"):
        selected = [row for row in rows if row["corruption"] == kind]
        axis.plot([row["strength"] for row in selected], [row["bev_ade"] for row in selected], marker="o", label=kind)
    axis.set(xlabel="corruption strength", ylabel="BEV ADE (m)", title="Ideal-observation corruption response"); axis.legend(); fig.tight_layout()
    fig.savefig(output / "corruption_response.png", dpi=140); plt.close(fig)
    return result

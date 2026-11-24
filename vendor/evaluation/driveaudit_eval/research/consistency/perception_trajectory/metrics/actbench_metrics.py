from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.common import wrap_angle


def _motion_class(rows: list[dict[str, Any]]) -> str:
    rows = sorted(rows, key=lambda row: row["timestamp"])
    if len(rows) < 2:
        return "not_available"
    xy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in rows], dtype=float)
    velocity = np.asarray([row.get("velocity", [0.0, 0.0]) for row in rows], dtype=float)
    speeds = np.linalg.norm(velocity, axis=1)
    start_yaw, end_yaw = float(rows[0]["box_bev"]["yaw"]), float(rows[-1]["box_bev"]["yaw"])
    delta = xy[-1] - xy[0]; lateral = float(np.dot(delta, [-np.sin(start_yaw), np.cos(start_yaw)]))
    yaw_change = wrap_angle(end_yaw - start_yaw)
    if np.linalg.norm(delta) < 1.0 and float(np.mean(speeds)) < 0.7: return "stopped"
    if speeds[0] < 0.7 and speeds[-1] > 1.5: return "starting"
    if speeds[0] > 1.5 and speeds[-1] < 0.7: return "stopping"
    if speeds[-1] - speeds[0] > 1.0: return "accelerating"
    if speeds[0] - speeds[-1] > 1.0: return "decelerating"
    if abs(lateral) > 1.5 and abs(yaw_change) < 0.35: return "shifting_left" if lateral > 0 else "shifting_right"
    if abs(yaw_change) >= 0.2: return "curving_left" if yaw_change > 0 else "curving_right"
    return "straight_constant_high_speed" if float(np.mean(speeds)) >= 5.0 else "straight_constant_low_speed"


def compute_actbench_metrics(reference_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                             mapping: list[dict[str, Any]]) -> dict[str, Any]:
    refs = {row["agent_id"]: row for row in reference_tracks}; preds = {row["image_track_id"]: row for row in image_tracks}
    records = []
    for match in mapping:
        sid, iid = match.get("submission_agent_id"), match.get("image_track_id")
        if not sid or not iid or sid not in refs or iid not in preds:
            continue
        a = {int(row["frame_index"]): row for row in refs[sid]["observations"] if row.get("expected_visible", False)}
        b = {int(row["frame_index"]): row for row in preds[iid]["observations"] if row.get("source") == "image_detection"}
        common = sorted(set(a) & set(b))
        if not common: continue
        distances = [float(np.linalg.norm([a[f]["box_bev"]["x"] - b[f]["box_bev"]["x"],
                                           a[f]["box_bev"]["y"] - b[f]["box_bev"]["y"]])) for f in common]
        instruction, executed = _motion_class([a[f] for f in common]), _motion_class([b[f] for f in common])
        records.append({"agent_id": sid, "instruction_class": instruction, "executed_class": executed,
                        "correct": instruction == executed, "ade": float(np.mean(distances)), "fde": distances[-1],
                        "samples": len(common)})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records: confusion[row["instruction_class"]][row["executed_class"]] += 1
    classes = sorted({row["instruction_class"] for row in records})
    macro = np.mean([np.mean([row["correct"] for row in records if row["instruction_class"] == category]) for category in classes]) if classes else None
    total_samples = sum(row["samples"] for row in records)
    completion = {category: float(np.mean([row["correct"] for row in records if row["instruction_class"] == category]))
                  for category in classes}
    groups = {"straight": {"straight_constant_high_speed", "straight_constant_low_speed"}, "left_turn": {"curving_left"},
              "right_turn": {"curving_right"}, "lane_change": {"shifting_left", "shifting_right"},
              "stop": {"stopping", "stopped"}, "accelerate": {"accelerating", "starting"}, "decelerate": {"decelerating"}}
    completion.update({name: float(np.mean([row["correct"] for row in records if row["instruction_class"] in labels]))
                       if any(row["instruction_class"] in labels for row in records) else NOT_AVAILABLE for name, labels in groups.items()})
    return {
        "reference_semantics": "submitted_trajectory_as_instruction_reference",
        "act_ego_motion_accuracy": NOT_AVAILABLE, "act_ego_ade": NOT_AVAILABLE, "act_ego_fde": NOT_AVAILABLE,
        "ego_unavailable_reason": "no_ego_instruction_track_in_submission_reference",
        "act_agent_motion_accuracy_adapted": float(np.mean([row["correct"] for row in records])) if records else NOT_AVAILABLE,
        "act_macro_motion_accuracy": float(macro) if macro is not None else NOT_AVAILABLE,
        "act_micro_motion_accuracy": float(sum(row["correct"] * row["samples"] for row in records) / total_samples) if total_samples else NOT_AVAILABLE,
        "act_agent_ade_adapted": float(np.mean([row["ade"] for row in records])) if records else NOT_AVAILABLE,
        "act_agent_fde_adapted": float(np.mean([row["fde"] for row in records])) if records else NOT_AVAILABLE,
        "act_macro_ade": float(np.mean([np.mean([row["ade"] for row in records if row["instruction_class"] == category]) for category in classes])) if classes else NOT_AVAILABLE,
        "act_micro_ade": float(sum(row["ade"] * row["samples"] for row in records) / total_samples) if total_samples else NOT_AVAILABLE,
        "motion_confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "action_completion_rate": completion, "matched_coverage_required": True,
        "sample_count": len(records), "implementation_level": "CONCEPTUAL_ADAPTATION",
        "adaptation_note": "ACT-Bench is Ego/front-camera; this extension classifies matched background-agent tracks.",
        "per_agent": records,
    }

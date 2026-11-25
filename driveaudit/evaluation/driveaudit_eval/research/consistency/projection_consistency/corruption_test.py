from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.geometry_flow.visual_consistency import summarize

from driveaudit_eval.research.consistency.projection_consistency.schema import mean, write_csv, write_json

ALIASES = {
    "trajectory_x_shift": "x_shift",
    "trajectory_y_shift": "y_shift",
    "wrong_camera_pair": "wrong_camera",
    "shuffled_target_image": "shuffled_image",
    "frozen_target_patch": "frozen_patch",
}


def evaluate_scene(runner: Any, core: dict[str, Any], selected: dict[str, Any], formal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_maximum = runner.config["sanity"]["maximum_windows"]
    runner.config["sanity"]["maximum_windows"] = int(runner.config.get("image_trajectory_consistency", {}).get("corruption_windows", original_maximum))
    raw = runner.oracle_leakage(core, selected, formal_rows)
    runner.config["sanity"]["maximum_windows"] = original_maximum
    actual_time_shift = _actual_time_shift(runner, core, selected)
    rows = []
    for row in raw["rows"]:
        if row["corruption"] == "time_shift":
            row = actual_time_shift
        row = {**row, "corruption": ALIASES.get(row["corruption"], row["corruption"])}
        rows.append(row)
    correct = next(row for row in rows if row["corruption"] == "correct_gt")
    for row in rows:
        row["quality_index"] = quality(row)
        row["effect_size_vs_correct"] = quality(correct) - quality(row)
        row["correct_gt_better"] = row["corruption"] == "correct_gt" or quality(correct) > quality(row)
    result = {
        "rows": rows,
        "rgb_tracker_reexecuted": True,
        "expected_flow_copied_as_observation": False,
        "all_corruptions_worse_than_correct": all(row["correct_gt_better"] for row in rows),
    }
    write_json(runner.output / "corruption_test/scene_summary.json", result)
    write_csv(runner.output / "corruption_test/correct_vs_corrupted.csv", rows)
    return result


def aggregate(scene_results: dict[str, dict[str, Any]], output: str | Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat = []
    for scene_id, result in scene_results.items():
        for row in result["rows"]:
            item = {"scene_id": scene_id, **row}
            flat.append(item); grouped[row["corruption"]].append(item)
    summary = []
    for family, rows in sorted(grouped.items()):
        comparisons = [row for row in rows if family != "correct_gt"]
        summary.append({
            "corruption": family,
            "scene_count": len(rows),
            "residual_epe": mean(row.get("residual_epe") for row in rows),
            "flow_cosine": mean(row.get("flow_cosine") for row in rows),
            "direction_agreement": mean(row.get("direction_agreement") for row in rows),
            "strong_support_rate": mean(row.get("strong_support_rate") for row in rows),
            "win_rate": mean(row.get("correct_gt_better") for row in comparisons) if comparisons else 1.0,
            "tie_rate": mean(abs(float(row.get("effect_size_vs_correct", 0))) <= 1e-9 for row in comparisons) if comparisons else 0.0,
            "loss_rate": mean(not row.get("correct_gt_better") for row in comparisons) if comparisons else 0.0,
            "mean_effect_size": mean(row.get("effect_size_vs_correct") for row in comparisons) if comparisons else 0.0,
        })
    result = {
        "scene_count": len(scene_results),
        "all_families": summary,
        "failed_corruptions": [row["corruption"] for row in summary if row["corruption"] != "correct_gt" and row["win_rate"] < 1.0],
    }
    output = Path(output)
    write_csv(output / "corruption_test/correct_vs_corrupted.csv", flat)
    write_json(output / "corruption_test/corruption_summary.json", result)
    return result


def quality(row: dict[str, Any]) -> float:
    support = _number(row.get("strong_support_rate"), 0.0)
    cosine = _number(row.get("flow_cosine"), -1.0)
    residual = _number(row.get("residual_epe"), 1000.0)
    return support + 0.25 * cosine - 0.01 * residual


def _actual_time_shift(runner: Any, core: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    maximum = int(runner.config["sanity"]["maximum_windows"])
    projection_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in core["native_projection"]}
    candidates = [model for model in core["native_warps"] if 10 <= int(model["frame_index"]) and int(model["target_frame_index"]) < 89 and model.get("expected_visible")
                  and projection_map.get((model["agent_id"], model["camera_name"], int(model["target_frame_index"]) + 1), {}).get("projected_valid")][:maximum]
    shifted = []
    for model in candidates:
        value = copy.deepcopy(model)
        value["target_frame_index"] = int(value["target_frame_index"]) + 1
        value["corruption_note"] = "candidate target RGB shifted by one native frame; expected warp remains t-to-t+1"
        shifted.append(value)
    rows, _ = runner._process(shifted, core["native_projection"], core["native_surfaces"], (10, 90), selected, "time_shift", maximum_windows=maximum)
    metric = summarize(rows, "guided")
    return {"corruption": "time_shift", **metric, "corruption_protocol": "actual_target_rgb_plus_one_frame"}


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json, write_jsonl


def scene_splits(scene_ids: list[str], holdout: str, ratios: dict[str, float], seed: int) -> dict[str, str]:
    unique = sorted(set(scene_ids)); result = {holdout: "test"} if holdout in unique else {}
    rest = [scene for scene in unique if scene != holdout]; random.Random(seed).shuffle(rest)
    n = len(rest); calibration_end = round(n * float(ratios["calibration_ratio"]))
    validation_end = calibration_end + round(n * float(ratios["validation_ratio"]))
    for index, scene in enumerate(rest):
        result[scene] = "calibration" if index < calibration_end else "validation" if index < validation_end else "test"
    return result


def build_positive_pairs(discovery: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    holdout = config["benchmark"]["current_holdout_scene"]
    scene_ids = [row["pair_id"] for row in discovery["scenes"]]
    splits = scene_splits(scene_ids, holdout, config["splits"], int(config["benchmark"]["random_seed"]))
    rows = []
    for scene in discovery["scenes"]:
        rows.append({"pair_id": scene["pair_id"] + "__history_real", "scene_id": scene["pair_id"],
                     "source_type": "history_real", "label": "positive", "negative_family": None,
                     "severity": None, "image_root": scene["image_history_root"],
                     "trajectory_path": scene["trajectory_path"], "camera_set": "all_six", "time_range": [0, 9],
                     "agent_ids": "from_history_metadata", "parent_pair_id": scene["pair_id"],
                     "data_split": splits[scene["pair_id"]],
                     "provenance": {"strict_real_history_pair": True, "used_as_calibration_positive": splits[scene["pair_id"]] == "calibration"}})
        rows.append({"pair_id": scene["pair_id"] + "__submission", "scene_id": scene["pair_id"],
                     "source_type": "submission", "label": "unlabeled", "negative_family": None,
                     "severity": None, "image_root": scene["generated_future_root"],
                     "trajectory_path": scene["trajectory_path"], "camera_set": "all_six", "time_range": [10, 89],
                     "agent_ids": "from_submission", "parent_pair_id": scene["pair_id"],
                     "data_split": splits[scene["pair_id"]],
                     "provenance": {"submission_as_positive": False, "used_for_threshold_selection": False}})
    output = Path(config["output"])
    write_jsonl(output / "manifest/positive_and_unlabeled_pairs.jsonl", rows)
    write_json(output / "manifest/splits.json", {"scene_splits": splits, "current_holdout_is_test": splits.get(holdout) == "test",
                                                   "same_scene_variants_share_split": True})
    return {"rows": rows, "splits": splits}


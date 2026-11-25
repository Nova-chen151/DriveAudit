from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json


def determine_status(scene_count: int, validation_count: int, discrimination: dict[str, Any],
                     monotonic_pass_rate: float | None, human: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    requirements = config["leaderboard"]["requirements"]
    synthetic = (discrimination.get("auroc", 0) >= float(requirements["minimum_auroc"])
                 and discrimination.get("false_positive_rate", 1) <= float(requirements["maximum_hard_negative_fpr"])
                 and (monotonic_pass_rate or 0) >= float(requirements["minimum_monotonic_pass_rate"]))
    if scene_count < int(requirements["minimum_calibration_scenes"]): status = "prototype_single_scene" if scene_count == 1 else "insufficient_scenes_for_leaderboard_calibration"
    elif not synthetic: status = "benchmark_validated_multi_scene"
    elif human.get("status") != "completed": status = "benchmark_validated_synthetic_only"
    else: status = "benchmark_human_correlated"
    total_allowed = (status in {"benchmark_human_correlated", "leaderboard_ready_provisional", "leaderboard_ready_frozen"}
                     and validation_count >= int(requirements["minimum_validation_scenes"])
                     and bool(config["leaderboard"]["enable_total_score"]))
    return {"status": status, "score_version": config["leaderboard"]["score_version"],
            "enable_total_score": total_allowed, "requested_total_score": bool(config["leaderboard"]["enable_total_score"]),
            "requirements": requirements, "synthetic_validation_passed": synthetic,
            "frozen": False, "reason": "insufficient independent scenes" if scene_count < int(requirements["minimum_calibration_scenes"]) else None}


def write_protocol(result: dict[str, Any], output: str | Path) -> None:
    write_json(Path(output) / "leaderboard_status.json", result)


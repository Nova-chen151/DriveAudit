from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def validate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    observable = [row for row in rows if row["motion_state"] != "geometry_unobservable"]
    scored = [row for row in observable if row.get("visually_scorable")]
    wins = Counter(row.get("winning_hypothesis") for row in scored)
    cfg = config["history_model_gate"]; total = max(1, len(scored))
    target_win = wins["target"]/total; background_win = wins["background"]/total
    false_contradiction = sum(row["motion_state"] == "observable_contradicted" for row in scored)/total
    residuals = [row.get("target_stats", {}).get("robust_residual") for row in scored]
    ratios = [float(r)/max(float(row.get("sigma_history", 1)), 1e-6) for r, row in zip(residuals, scored) if isinstance(r, (int, float))]
    checks = {"minimum_observable_windows": len(scored) >= int(cfg["min_observable_windows"]),
              "target_win_rate": target_win >= float(cfg["min_target_win_rate_observable"]),
              "false_contradiction_rate": false_contradiction <= float(cfg["max_false_contradiction_rate"]),
              "target_residual_sigma_ratio": (float(np.mean(ratios)) if ratios else float("inf")) <= float(cfg["max_target_residual_sigma_ratio"])}
    passed = all(checks.values())
    return {"status": "passed" if passed else "unscorable_hypothesis_model_failure", "passed": passed,
            "history_total_windows": len(rows), "history_geometry_observable_windows": len(observable),
            "history_geometry_observable_coverage": len(observable)/max(1, len(rows)),
            "history_target_win_rate_all": sum(row.get("winning_hypothesis") == "target" for row in rows)/max(1, len(rows)),
            "history_target_win_rate_observable": target_win, "history_background_win_rate_observable": background_win,
            "history_zero_win_rate_observable": wins["zero"]/total, "history_other_win_rate_observable": wins["other"]/total,
            "history_target_margin_observable": _mean([row.get("target_hypothesis_margin") for row in scored]),
            "history_motion_support_rate_observable": sum(row["motion_state"] == "observable_supported" for row in scored)/total,
            "history_false_contradiction_rate_observable": false_contradiction,
            "history_target_residual_sigma_ratio": float(np.mean(ratios)) if ratios else None, "checks": checks,
            "thresholds_provisional": True, "future_results_used_to_set_gate": False}


from driveaudit_eval.utils.aggregation import mean as _mean

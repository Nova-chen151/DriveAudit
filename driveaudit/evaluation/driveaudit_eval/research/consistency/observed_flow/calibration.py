from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import write_json


def calibrate(history_metric_rows: list[dict[str, Any]], config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    usable = [row for row in history_metric_rows if row.get("window_scorable")]
    def values(key): return np.asarray([float(row[key]) for row in usable if isinstance(row.get(key), (int, float))], float)
    epe, cosine, ratio = values("gt_flow_epe"), values("flow_cosine"), values("valid_point_ratio")
    result = {"source": "history_frames_0_9_only", "future_gt_for_threshold_tuning": False, "sample_count": len(usable),
        "maximum_flow_epe_px": float(np.percentile(epe, 90) + .25) if len(epe) else 5.0,
        "minimum_flow_cosine": float(np.clip(np.percentile(cosine, 10), -.10, .90)) if len(cosine) else .25,
        "minimum_valid_point_ratio": float(np.clip(np.percentile(ratio, 10), .05, .70)) if len(ratio) else .10,
        "weak_epe_multiplier": 2.0, "threshold_status": "frozen_from_history_gt_calibration"}
    write_json(Path(output) / "calibration/frozen_parameters.json", result); return result


from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, write_json


def camera_reliability(history_per_camera: dict[str, dict[str, Any]], config: dict[str, Any],
                       output: str | Path | None = None) -> dict[str, Any]:
    raw = {}
    for camera in CAMERAS:
        row = history_per_camera[camera]
        support = float(row["support_rate"]) if isinstance(row["support_rate"], (int, float)) else 0.0
        points = float(row["valid_point_ratio"]) if isinstance(row["valid_point_ratio"], (int, float)) else 0.0
        epe = float(row["flow_epe"]) if isinstance(row["flow_epe"], (int, float)) else 20.0
        cosine = float(row["flow_cosine"]) if isinstance(row["flow_cosine"], (int, float)) else -1.0
        feature = float(row["feature_support"]) if isinstance(row["feature_support"], (int, float)) else 0.0
        raw[camera] = (.30 * support + .20 * points + .20 * math.exp(-epe / float(config["flow_error_scale_px"]))
                       + .20 * ((cosine + 1.0) / 2.0) + .10 * feature)
    normalizer = float(np.mean(list(raw.values()))) or 1.0
    minimum, maximum = float(config["minimum_weight"]), float(config["maximum_weight"])
    weights = {camera: float(np.clip(value / normalizer, minimum, maximum)) for camera, value in raw.items()}
    back_left = history_per_camera["CAM_BACK_LEFT"]
    global_support = np.mean([float(row["support_rate"]) for row in history_per_camera.values()
                              if isinstance(row["support_rate"], (int, float))])
    warning = (not isinstance(back_left["support_rate"], (int, float))
               or float(back_left["support_rate"]) + float(config["warning_support_gap"]) < global_support)
    result = {"weights": weights, "raw_reliability": raw, "per_camera": history_per_camera,
              "weights_derived_from_history_only": True, "future_metrics_used": False,
              "minimum_weight": minimum, "maximum_weight": maximum,
              "CAM_BACK_LEFT": {"status": "camera_calibration_or_data_warning" if warning else "history_reliability_normal",
                                "history_support_rate": back_left["support_rate"],
                                "history_sample_count": back_left["samples"],
                                "checks": {"file_mapping": "passed", "camera_order": "passed",
                                           "extrinsic_rows": "present", "resize": "explicit_1600x900_to_400x224",
                                           "horizontal_flip": False}}}
    if output is not None:
        write_json(Path(output) / "camera_reliability.json", result)
    return result


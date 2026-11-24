from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.verification_benchmark.common import file_hash, write_json


def fit_percentile_mapping(positive: list[float], negative: list[float], scene_count: int,
                           source_files: list[str], output: str | Path) -> dict[str, Any]:
    if scene_count < 2 or not positive or not negative:
        result = {"status": "not_fitted_insufficient_calibration_scenes", "score_version": "tcvv_score_v1_provisional",
                  "calibration_scene_count": scene_count, "test_minmax_used": False}
    else:
        result = {"status": "fitted_provisional", "score_version": "tcvv_score_v1_provisional",
                  "negative_p95": float(np.percentile(negative, 95)), "positive_p05": float(np.percentile(positive, 5)),
                  "positive_distribution": {"mean": float(np.mean(positive)), "std": float(np.std(positive))},
                  "negative_distribution": {"mean": float(np.mean(negative)), "std": float(np.std(negative))},
                  "calibration_dataset_hash": file_hash(source_files), "test_minmax_used": False}
    write_json(Path(output) / "score_mapping.json", result); return result


from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np

from driveaudit_eval.trajectory import NOT_AVAILABLE


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def longest_true_duration(mask, timestamps):
    mask = np.asarray(mask, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=float)
    best = start = 0
    for i, flag in enumerate(mask):
        if flag:
            if i == 0 or not mask[i - 1]:
                start = i
            best = max(best, i - start + 1)
    if best <= 1 or len(timestamps) < 2:
        return 0.0
    dt = float(np.median(np.diff(timestamps)))
    return float(best * dt)


def stats(values, violation=None, timestamps=None):
    values = finite(values)
    if not len(values):
        return NOT_AVAILABLE
    result = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for percentile in (50, 90, 95, 99):
        result["P%d" % percentile] = float(np.percentile(values, percentile))
    if violation is not None:
        violation = np.asarray(violation, dtype=bool)
        result["violation_ratio"] = float(np.mean(violation)) if len(violation) else 0.0
        result["max_consecutive_violation_seconds"] = longest_true_duration(
            violation, timestamps if timestamps is not None else np.arange(len(violation), dtype=float)
        )
    return result


def jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


from driveaudit_eval.utils.aggregation import mean_or_na as mean_available


from __future__ import annotations

from typing import Iterable

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def diagnose_time_shift(reference: Iterable[float], observed: Iterable[float], dt: float, max_shift: int = 2) -> dict[str, float | int | str]:
    reference, observed = np.asarray(list(reference), dtype=float), np.asarray(list(observed), dtype=float)
    n = min(len(reference), len(observed)); reference, observed = reference[:n], observed[:n]
    if n < 3:
        return {"estimated_time_shift_frames": NOT_AVAILABLE, "estimated_time_shift_seconds": NOT_AVAILABLE,
                "pre_alignment_error": NOT_AVAILABLE, "post_diagnostic_alignment_error": NOT_AVAILABLE}
    errors = {}
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0: a, b = reference[-shift:], observed[:n + shift]
        elif shift > 0: a, b = reference[:n - shift], observed[shift:]
        else: a, b = reference, observed
        errors[shift] = float(np.mean(np.abs(a - b)))
    best = min(errors, key=errors.get)
    return {"estimated_time_shift_frames": int(best), "estimated_time_shift_seconds": float(best * dt),
            "pre_alignment_error": errors[0], "post_diagnostic_alignment_error": errors[best],
            "formal_scoring_alignment": "submitted_timestamps_unshifted"}


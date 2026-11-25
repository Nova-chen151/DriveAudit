from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.backends import OpenCVLocalBackend


def verify_local_window(backend: OpenCVLocalBackend, image_a: np.ndarray, image_b: np.ndarray,
                        expectation: dict[str, Any], evidence_mode: str = "fusion") -> dict[str, Any]:
    return backend.verify(image_a, image_b, expectation, evidence_mode)


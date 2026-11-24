from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LocalEvidenceBackend(ABC):
    @abstractmethod
    def verify(self, image_a: np.ndarray, image_b: np.ndarray, expectation: dict[str, Any],
               evidence_mode: str = "fusion") -> dict[str, Any]:
        """Verify one bounded local window without retaining state across windows."""


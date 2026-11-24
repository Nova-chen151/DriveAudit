from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBackend(ABC):
    """Perception-only contract. Future structured trajectories are intentionally absent."""

    @abstractmethod
    def initialize(self, history_frames: list[dict[str, Any]], history_boxes: list[dict[str, Any]],
                   calibration: list[dict[str, Any]], ego_poses: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame_images: dict[str, str], calibration: list[dict[str, Any]],
              ego_pose: dict[str, Any], timestamp: float) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, scene_id: str) -> None:
        raise NotImplementedError


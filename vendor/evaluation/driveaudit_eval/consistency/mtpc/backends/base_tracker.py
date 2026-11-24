from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MTPCTrackerBackend(ABC):
    @abstractmethod
    def initialize(self, history_images: dict[str, list[str]], projected_history_boxes: list[dict[str, Any]],
                   history_agent_ids: list[str], calibrations: list[dict[str, Any]], ego_poses: list[dict[str, Any]],
                   timestamps: list[float]) -> dict[str, Any]: ...

    @abstractmethod
    def track_future_frame(self, frame_images: dict[str, Any], calibrations: list[dict[str, Any]],
                           ego_pose: dict[str, Any], timestamp: float) -> list[dict[str, Any]]: ...

    @abstractmethod
    def finalize(self) -> dict[str, Any]: ...

    @abstractmethod
    def reset(self, pair_id: str) -> None: ...


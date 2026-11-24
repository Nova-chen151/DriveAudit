from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SceneRef:
    index: int
    relative_path: str
    dataset: str
    scene_id: str
    root: Path
    asset_root: Optional[Path] = None
    trajectory_root: Optional[Path] = None


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    intrinsics: np.ndarray
    camera_to_ego: np.ndarray


@dataclass(frozen=True)
class AgentBox:
    frame_index: int
    agent_id: str
    category: str
    center_xyz: np.ndarray
    size_wlh: np.ndarray
    yaw: float


@dataclass(frozen=True)
class MapCondition:
    mask: np.ndarray
    ground_height_m: np.ndarray
    extent_m: Tuple[float, float, float, float]
    resolution_m: float
    available_channels: Tuple[str, ...]
    unavailable_channels: Tuple[str, ...]
    vectors: Mapping[str, object]


@dataclass(frozen=True)
class SceneInputs:
    scene: SceneRef
    camera_order: Tuple[str, ...]
    history_rgb: Mapping[str, Tuple[Path, ...]]
    calibrations: Mapping[str, CameraCalibration]
    ego_to_anchor: np.ndarray
    ego_trajectory: np.ndarray
    agents_by_frame: Mapping[int, Tuple[AgentBox, ...]]
    map_condition: MapCondition
    prompt: str
    timeline_us: np.ndarray


@dataclass(frozen=True)
class AuditReport:
    scene_count: int
    history_image_count: int
    reference_future_file_count: int
    dataset_counts: Mapping[str, int]
    errors: Tuple[str, ...]

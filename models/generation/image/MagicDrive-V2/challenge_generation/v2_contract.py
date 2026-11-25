"""Public data contract and deterministic map cropping for Challenger v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MAP_CHANNELS = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
    "road_block",
)
MAP_SHAPE = (8, 1024, 1024)
CROP_SHAPE = (8, 200, 200)
MAP_SCHEMA_VERSION = "challenger-map-v2.0"
MAP_RESOLUTION_M = 0.5
MAP_EXTENT_M = (-256.0, 256.0, -256.0, 256.0)
MAP_COORDINATE_FRAME = "history_last_ego_x_forward_y_left_meters"
MAP_ARRAY_AXES = "channel,x_forward,y_left"


@dataclass(frozen=True)
class MapMetadata:
    schema_version: str
    channels: tuple[str, ...]
    resolution_m: float
    extent_m: tuple[float, float, float, float]
    coordinate_frame: str
    array_axes: str

    @classmethod
    def from_dict(cls, value: dict) -> "MapMetadata":
        metadata = cls(
            schema_version=str(value["schema_version"]),
            channels=tuple(value["channels"]),
            resolution_m=float(value["resolution_m"]),
            extent_m=tuple(float(item) for item in value["extent_m"]),
            coordinate_frame=str(value["coordinate_frame"]),
            array_axes=str(value["array_axes"]),
        )
        metadata.validate()
        return metadata

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "channels": list(self.channels),
            "resolution_m": self.resolution_m,
            "extent_m": list(self.extent_m),
            "coordinate_frame": self.coordinate_frame,
            "array_axes": self.array_axes,
        }

    def validate(self) -> None:
        expected = MapMetadata(
            schema_version=MAP_SCHEMA_VERSION,
            channels=MAP_CHANNELS,
            resolution_m=MAP_RESOLUTION_M,
            extent_m=MAP_EXTENT_M,
            coordinate_frame=MAP_COORDINATE_FRAME,
            array_axes=MAP_ARRAY_AXES,
        )
        if self != expected:
            raise ValueError(f"Unexpected Challenger v2 map metadata: {self}")


def default_metadata() -> MapMetadata:
    return MapMetadata(
        schema_version=MAP_SCHEMA_VERSION,
        channels=MAP_CHANNELS,
        resolution_m=MAP_RESOLUTION_M,
        extent_m=MAP_EXTENT_M,
        coordinate_frame=MAP_COORDINATE_FRAME,
        array_axes=MAP_ARRAY_AXES,
    )


def load_map(scene_dir: Path) -> tuple[np.ndarray, MapMetadata]:
    map_dir = Path(scene_dir) / "map"
    with (map_dir / "map_meta.json").open("r", encoding="utf-8") as handle:
        metadata = MapMetadata.from_dict(json.load(handle))
    with np.load(map_dir / "semantic_map.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"mask"}:
            raise ValueError(f"semantic_map.npz must contain only 'mask': {archive.files}")
        mask = np.asarray(archive["mask"])
    if mask.shape != MAP_SHAPE or mask.dtype != np.uint8:
        raise ValueError(
            f"Challenger v2 map must be uint8 {MAP_SHAPE}, got {mask.dtype} {mask.shape}"
        )
    if np.any(mask > 1):
        raise ValueError("Challenger v2 map channels must be binary")
    return mask, metadata


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def crop_map(
    mask: np.ndarray,
    metadata: MapMetadata,
    ego_position: np.ndarray,
    ego_yaw: float,
    renderer_origin: np.ndarray,
    local_to_ego_rotation: Iterable[Iterable[float]] | None = None,
) -> np.ndarray:
    """Crop one 100 m model BEV using only a predicted anchor-relative pose."""
    metadata.validate()
    if mask.shape != MAP_SHAPE or mask.dtype != np.uint8:
        raise ValueError(f"Expected uint8 map {MAP_SHAPE}, got {mask.dtype} {mask.shape}")
    position = np.asarray(ego_position, dtype=np.float64)
    origin = np.asarray(renderer_origin, dtype=np.float64)
    if position.shape != (2,) or origin.shape not in {(2,), (3,)}:
        raise ValueError("ego_position and renderer_origin have invalid shapes")
    if not np.isfinite(position).all() or not np.isfinite(origin).all() or not math.isfinite(ego_yaw):
        raise ValueError("Map crop transform must be finite")
    local_rotation = (
        np.eye(2, dtype=np.float64)
        if local_to_ego_rotation is None
        else np.asarray(local_to_ego_rotation, dtype=np.float64)
    )
    if local_rotation.shape != (2, 2) or not np.isfinite(local_rotation).all():
        raise ValueError("local_to_ego_rotation must be a finite 2x2 matrix")

    axis = (np.arange(CROP_SHAPE[1], dtype=np.float64) - 100.0) * metadata.resolution_m
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    local_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    ego_points = local_points @ local_rotation.T + origin[None, :2]
    anchor_points = ego_points @ _rotation(float(ego_yaw)).T + position[None]
    indices = np.rint(anchor_points / metadata.resolution_m + 512.0).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= MAP_SHAPE[1]):
        lower = anchor_points.min(axis=0)
        upper = anchor_points.max(axis=0)
        raise ValueError(
            "Predicted map crop is outside static map: "
            f"x=[{lower[0]:.2f},{upper[0]:.2f}], y=[{lower[1]:.2f},{upper[1]:.2f}]"
        )
    sampled = mask[:, indices[:, 0], indices[:, 1]]
    return sampled.reshape(CROP_SHAPE).astype(np.float32, copy=False)


def validate_public_scene(scene_dir: Path) -> None:
    load_map(scene_dir)

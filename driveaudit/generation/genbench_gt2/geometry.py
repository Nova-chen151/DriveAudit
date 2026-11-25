"""Coordinate transforms and camera projection for GT2 model conditions."""

from dataclasses import replace
from typing import Tuple

import numpy as np

from .schema import AgentBox, CameraCalibration


def _matrix4(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("{} must be a finite 4x4 matrix".format(name))
    return matrix


def apply_linear(matrix, points):
    """Apply a small coordinate matrix without BLAS GEMM packing kernels.

    NumPy 1.23.3/OpenBLAS on some cluster CPU nodes corrupts 4xN products
    for N>=200. Explicit contractions also preserve the homogeneous column.
    """
    return np.einsum('ij,...j->...i', np.asarray(matrix), np.asarray(points), optimize=False)


def anchor_to_target(ego_to_anchor: np.ndarray) -> np.ndarray:
    """Return the target-ego transform from its camera/agent anchor pose."""
    return np.linalg.inv(_matrix4(ego_to_anchor, "ego_to_anchor"))


def transform_points(points_anchor: np.ndarray, ego_to_anchor: np.ndarray) -> np.ndarray:
    points = np.asarray(points_anchor, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_anchor must have shape [N,3]")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return apply_linear(anchor_to_target(ego_to_anchor), homogeneous)[:, :3]


def transform_agent_to_target(agent: AgentBox, ego_to_anchor: np.ndarray) -> AgentBox:
    inverse = anchor_to_target(ego_to_anchor)
    center = transform_points(np.asarray(agent.center_xyz, dtype=np.float64)[None], ego_to_anchor)[0]
    heading = inverse[:3, :3] @ np.array([np.cos(agent.yaw), np.sin(agent.yaw), 0.0])
    yaw = float(np.arctan2(heading[1], heading[0]))
    return replace(agent, center_xyz=center, yaw=((yaw + np.pi) % (2 * np.pi)) - np.pi)


def box_corners(box: AgentBox) -> np.ndarray:
    """Return the eight oriented cuboid corners in target-ego coordinates."""
    width, length, height = np.asarray(box.size_wlh, dtype=np.float64)
    local = np.array([
        [sx * length / 2.0, sy * width / 2.0, sz * height / 2.0]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ])
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    local[:, :2] = local[:, :2] @ np.array([[c, s], [-s, c]])
    return local + np.asarray(box.center_xyz, dtype=np.float64)


def scaled_calibration(calibration: CameraCalibration, source_size: Tuple[int, int], target_size: Tuple[int, int]) -> CameraCalibration:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image sizes must be positive")
    intrinsics = np.asarray(calibration.intrinsics, dtype=np.float64).copy()
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    intrinsics[0] *= target_width / source_width
    intrinsics[1] *= target_height / source_height
    return replace(calibration, intrinsics=intrinsics, camera_to_ego=np.asarray(calibration.camera_to_ego, dtype=np.float64).copy())


def project_points(points_target: np.ndarray, calibration: CameraCalibration):
    points = np.asarray(points_target, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_target must have shape [N,3]")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera_xyz = apply_linear(np.linalg.inv(_matrix4(calibration.camera_to_ego, "camera_to_ego")), homogeneous)[:, :3]
    pixels_h = apply_linear(np.asarray(calibration.intrinsics, dtype=np.float64), camera_xyz)
    visible = camera_xyz[:, 2] > 1e-5
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[visible] = pixels_h[visible, :2] / pixels_h[visible, 2:3]
    return pixels, visible

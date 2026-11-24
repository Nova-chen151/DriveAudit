from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def overlapping_camera_pairs(cameras):
    """Measured pinhole frusta, sampled at 5--80 m; matches test actual visibility.

    Calibration establishes potential overlap, not an occlusion-free surface.
    The symmetric grid is independent of dataset camera names.
    """
    from itertools import combinations
    pairs, audit = [], []
    for a, b in combinations(cameras, 2):
        fractions = []
        for source, target in ((a, b), (b, a)):
            w, h = source['challenge_image_size']
            u, v = np.meshgrid(np.linspace(.5, w-.5, 41), np.linspace(.5, h-.5, 23))
            rays = np.stack((u.ravel(), v.ravel(), np.ones(u.size)))
            rays = np.linalg.solve(np.asarray(source['challenge_intrinsics']), rays)
            transform = np.linalg.inv(np.asarray(target['camera_to_ego_4x4'])) @ np.asarray(source['camera_to_ego_4x4'])
            tw, th = target['challenge_image_size']
            visible = np.zeros(u.size, dtype=bool)
            for depth in (5., 10., 20., 40., 80.):
                points = transform[:3, :3] @ (rays*depth) + transform[:3, 3:4]
                pixels = np.asarray(target['challenge_intrinsics']) @ points
                z = pixels[2]
                xy = pixels[:2] / np.where(abs(z)>1e-12, z, np.nan)
                visible |= (points[2]>0) & (xy[0]>=0) & (xy[0]<tw) & (xy[1]>=0) & (xy[1]<th)
            fractions.append(float(visible.mean()))
        valid = max(fractions) >= .01
        pair = [a['camera'], b['camera']]
        audit.append(dict(pair=pair, directional_overlap_fraction=fractions, valid=valid))
        if valid:
            pairs.append(pair)
    return pairs, audit


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(t, dtype=np.float64).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def scale_intrinsics(
    k: np.ndarray,
    original_size: Optional[Sequence[int]],
    target_size: Optional[Sequence[int]],
) -> tuple[np.ndarray, Dict[str, float]]:
    k_scaled = np.asarray(k, dtype=np.float64).reshape(3, 3).copy()
    debug = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "original_width": None,
        "original_height": None,
        "target_width": None,
        "target_height": None,
    }
    if original_size is None or target_size is None:
        return k_scaled, debug
    original_width, original_height = float(original_size[0]), float(original_size[1])
    target_width, target_height = float(target_size[0]), float(target_size[1])
    if original_width <= 0 or original_height <= 0:
        return k_scaled, debug
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    k_scaled[0, 0] *= scale_x
    k_scaled[0, 2] *= scale_x
    k_scaled[1, 1] *= scale_y
    k_scaled[1, 2] *= scale_y
    debug.update(
        {
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "target_width": int(target_width),
            "target_height": int(target_height),
        }
    )
    return k_scaled, debug


def prepare_camera_for_eval(
    camera: Dict[str, np.ndarray],
    original_size: Optional[Sequence[int]],
    target_size: Optional[Sequence[int]],
    scale_k: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    k_original = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    if scale_k:
        k_scaled, debug = scale_intrinsics(k_original, original_size, target_size)
    else:
        k_scaled, debug = scale_intrinsics(k_original, None, None)
    prepared = {
        "K": k_scaled,
        "R": np.asarray(camera["R"], dtype=np.float64).reshape(3, 3),
        "t": np.asarray(camera["t"], dtype=np.float64).reshape(3),
    }
    debug.update(
        {
            "K_original": k_original.tolist(),
            "K_scaled": k_scaled.tolist(),
        }
    )
    return prepared, debug


def fundamental_from_extrinsics(
    cam_i: Dict[str, np.ndarray],
    cam_j: Dict[str, np.ndarray],
    convention: str = "sensor_to_ego",
) -> Optional[np.ndarray]:
    try:
        k_i = np.asarray(cam_i["K"], dtype=np.float64).reshape(3, 3)
        k_j = np.asarray(cam_j["K"], dtype=np.float64).reshape(3, 3)
        r_i = np.asarray(cam_i["R"], dtype=np.float64).reshape(3, 3)
        r_j = np.asarray(cam_j["R"], dtype=np.float64).reshape(3, 3)
        t_i = np.asarray(cam_i["t"], dtype=np.float64).reshape(3)
        t_j = np.asarray(cam_j["t"], dtype=np.float64).reshape(3)
    except Exception:
        return None

    if convention in {"sensor_to_ego", "camera_to_world", "camera_to_ego"}:
        # X_common = R_i X_i + t_i
        # X_j = R_j^T (X_common - t_j)
        r_ij = r_j.T @ r_i
        t_ij = r_j.T @ (t_i - t_j)
    elif convention in {"world_to_camera", "ego_to_camera"}:
        # X_i = R_i X_common + t_i
        r_ij = r_j @ r_i.T
        t_ij = t_j - r_ij @ t_i
    else:
        return None
    e = skew(t_ij) @ r_ij
    try:
        f = np.linalg.inv(k_j).T @ e @ np.linalg.inv(k_i)
    except np.linalg.LinAlgError:
        return None
    norm = np.linalg.norm(f)
    return None if norm <= 1e-12 else f / norm


def epipolar_errors(points_i: np.ndarray, points_j: np.ndarray, f: np.ndarray) -> np.ndarray:
    points_i = np.asarray(points_i, dtype=np.float64)
    points_j = np.asarray(points_j, dtype=np.float64)
    if points_i.size == 0 or points_j.size == 0:
        return np.empty((0,), dtype=np.float64)
    ones = np.ones((points_i.shape[0], 1), dtype=np.float64)
    x_i = np.concatenate([points_i[:, :2], ones], axis=1)
    x_j = np.concatenate([points_j[:, :2], ones], axis=1)
    lines_j = (f @ x_i.T).T
    numerator = np.abs(np.sum(x_j * lines_j, axis=1))
    denom = np.sqrt(lines_j[:, 0] ** 2 + lines_j[:, 1] ** 2) + 1e-12
    return numerator / denom


def ransac_fundamental(
    points_i: np.ndarray, points_j: np.ndarray
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    points_i = np.asarray(points_i, dtype=np.float32)
    points_j = np.asarray(points_j, dtype=np.float32)
    if len(points_i) < 8 or len(points_j) < 8:
        return None, np.zeros((len(points_i),), dtype=bool)
    try:
        import cv2
    except Exception:
        return None, np.zeros((len(points_i),), dtype=bool)
    f, mask = cv2.findFundamentalMat(points_i, points_j, cv2.FM_RANSAC, 3.0, 0.99)
    if f is None or mask is None:
        return None, np.zeros((len(points_i),), dtype=bool)
    return f, mask.reshape(-1).astype(bool)

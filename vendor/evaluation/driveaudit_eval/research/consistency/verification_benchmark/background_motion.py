from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def polygon_mask(shape: tuple[int, int], polygon: Iterable[Iterable[float]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8); points = np.asarray(list(polygon), np.float32)
    if len(points) >= 3: cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 255)
    return mask


def background_ring(shape: tuple[int, int], polygon: list[list[float]], other_polygons: list[list[list[float]]],
                    width_ratio: float) -> np.ndarray:
    target = polygon_mask(shape, polygon)
    if not np.any(target): return target
    x, y, w, h = cv2.boundingRect(np.asarray(polygon, np.float32)); radius = max(1, round(max(w, h) * width_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.subtract(cv2.dilate(target, kernel), target)
    for other in other_polygons:
        ring[polygon_mask(shape, other) > 0] = 0
    return ring


def dense_flow(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)


def robust_flow(flow: np.ndarray, mask: np.ndarray, minimum_pixels: int) -> tuple[list[float] | None, int, float | None]:
    vectors = flow[mask > 0]
    if len(vectors) < minimum_pixels: return None, len(vectors), None
    median = np.median(vectors, axis=0); dispersion = np.median(np.linalg.norm(vectors - median, axis=1))
    return median.astype(float).tolist(), len(vectors), float(dispersion)


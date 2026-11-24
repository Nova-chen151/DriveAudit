from __future__ import annotations

import math

import numpy as np


def points_xy(value):
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
        raise ValueError("points must be finite Nx2 coordinates")
    return points[:, :2]


def wrap_angle(value):
    return (np.asarray(value, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def trajectory_headings(points, initial_heading=0.0):
    pts = points_xy(points)
    if len(pts) < 2:
        return np.full(len(pts), float(initial_heading))
    delta = np.diff(np.vstack(([pts[0] - [math.cos(initial_heading), math.sin(initial_heading)]], pts)), axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    still = np.linalg.norm(delta, axis=1) < 1e-5
    for index in range(len(headings)):
        if still[index]:
            headings[index] = headings[index - 1] if index else initial_heading
    return headings


def obb_polygon(center, heading, length, width):
    from shapely.geometry import Polygon
    center = np.asarray(center, dtype=float)[:2]
    local = np.asarray([[length / 2, width / 2], [length / 2, -width / 2],
                        [-length / 2, -width / 2], [-length / 2, width / 2]])
    c, s = math.cos(float(heading)), math.sin(float(heading))
    return Polygon(local @ np.asarray([[c, s], [-s, c]]) + center)


def footprint_coverage(drivable, centers, headings, length, width):
    values = []
    for point, heading in zip(points_xy(centers), np.asarray(headings, dtype=float)):
        polygon = obb_polygon(point, heading, length, width)
        values.append(float(polygon.intersection(drivable).area / polygon.area) if polygon.area else 0.0)
    return np.asarray(values, dtype=float)

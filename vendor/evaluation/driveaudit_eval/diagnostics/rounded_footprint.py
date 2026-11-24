"""Independent XY footprint proxy; never imported by formal collision metrics.

Boxes use [..., (x, y, yaw, length, width)] in metres/radians. For overlapping
rectangular cores, the nearest Minkowski boundary is the least separating-axis
translation (including containment). Outside, the closest rectangle vertices
to the opposite rectangle give the Euclidean distance. Subtracting both disk
radii gives the signed distance of the rounded footprints.
"""
import numpy as np


def signed_distance(a, b, corner_rounding_factor=0.0):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    c = float(corner_rounding_factor)
    if a.shape[-1] != 5 or not 0 <= c < 1:
        raise ValueError('expected (...,5) boxes and 0 <= rounding factor < 1')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a[..., 3:] <= 0) or np.any(b[..., 3:] <= 0):
        raise ValueError('boxes must be finite with positive length/width')
    delta = b[..., :2] - a[..., :2]  # Recenter before constructing any corners.
    ra = c * a[..., 3:].min(axis=-1) / 2
    rb = c * b[..., 3:].min(axis=-1) / 2
    ha, hb = a[..., 3:] / 2 - ra[..., None], b[..., 3:] / 2 - rb[..., None]
    ua = np.stack([np.cos(a[..., 2]), np.sin(a[..., 2])], axis=-1)
    ub = np.stack([np.cos(b[..., 2]), np.sin(b[..., 2])], axis=-1)
    va = np.stack([-ua[..., 1], ua[..., 0]], axis=-1)
    vb = np.stack([-ub[..., 1], ub[..., 0]], axis=-1)
    axes = np.stack([ua, va, ub, vb], axis=-2)
    dot = lambda x, y: np.sum(x * y, axis=-1)
    support_a = ha[..., 0, None] * abs(dot(axes, ua[..., None, :])) + ha[..., 1, None] * abs(dot(axes, va[..., None, :]))
    support_b = hb[..., 0, None] * abs(dot(axes, ub[..., None, :])) + hb[..., 1, None] * abs(dot(axes, vb[..., None, :]))
    separation = (abs(dot(axes, delta[..., None, :])) - support_a - support_b).max(axis=-1)
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ca = signs[:, 0, None] * ha[..., 0, None, None] * ua[..., None, :] + signs[:, 1, None] * ha[..., 1, None, None] * va[..., None, :]
    cb = signs[:, 0, None] * hb[..., 0, None, None] * ub[..., None, :] + signs[:, 1, None] * hb[..., 1, None, None] * vb[..., None, :]
    def vertex_distance(points, u, v, half):
        local = np.stack([dot(points, u[..., None, :]), dot(points, v[..., None, :])], axis=-1)
        return np.linalg.norm(np.maximum(abs(local) - half[..., None, :], 0), axis=-1).min(axis=-1)
    outside = np.minimum(vertex_distance(ca - delta[..., None, :], ub, vb, hb),
                         vertex_distance(cb + delta[..., None, :], ua, va, ha))
    return np.where(separation <= 0, separation, outside) - ra - rb


def nearest_distances(boxes, valid, evaluated_indices, corner_rounding_factor=.7):
    """NaN + false for unavailable neighbours; evaluated indices retain order."""
    boxes, valid = np.asarray(boxes, float), np.asarray(valid, bool)
    indices = np.asarray(evaluated_indices, int)
    if boxes.ndim != 2 or boxes.shape[1] != 5 or valid.shape != (len(boxes),):
        raise ValueError('expected N x 5 boxes and N validity flags')
    if np.any(indices < 0) or np.any(indices >= len(boxes)) or len(set(indices)) != len(indices):
        raise ValueError('evaluated indices must be unique and in bounds')
    result = np.full(len(indices), np.nan)
    for k, i in enumerate(indices):
        neighbours = np.flatnonzero(valid & (np.arange(len(boxes)) != i))
        if valid[i] and len(neighbours):
            result[k] = signed_distance(boxes[i], boxes[neighbours], corner_rounding_factor).min()
    return result, np.isfinite(result)

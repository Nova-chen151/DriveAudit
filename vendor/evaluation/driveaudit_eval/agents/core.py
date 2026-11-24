from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# These are explicit on purpose.  Never rely on dict insertion order for cameras.
WORLD_DREAMER_CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
]
UNIAD_CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
VAD_CAMERA_ORDER = UNIAD_CAMERA_ORDER[:]
CAMERA_SET = set(WORLD_DREAMER_CAMERA_ORDER)
EGO_WIDTH = 1.85
EGO_LENGTH = 4.084


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def json_cell(value: str) -> Any:
    return json.loads(value) if value else []


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def quat_to_yaw(q: Sequence[float]) -> float:
    # nuScenes quaternion convention: w, x, y, z.
    w, x, y, z = map(float, q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_rotation(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = map(float, q)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm <= 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


def ego_to_world(points: Sequence[Sequence[float]], position: Sequence[float], yaw: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return pts[:, :2] @ rot.T + np.asarray(position[:2], dtype=float)


def world_to_ego(points: Sequence[Sequence[float]], position: Sequence[float], yaw: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)[:, :2] - np.asarray(position[:2], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]], dtype=float).T


def derive_kinematics(poses: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    cur = poses[index]
    t = float(cur["timestamp"])
    p = np.asarray(cur["position"], dtype=float)
    yaw = float(cur["yaw"])
    if index == 0:
        prev, dt = cur, 1.0 / 12.0
    else:
        prev = poses[index - 1]
        dt = max((t - float(prev["timestamp"])) / 1e6, 1e-4)
    velocity_world = (p - np.asarray(prev["position"], dtype=float)) / dt
    velocity_ego = world_to_ego([velocity_world[:2] + p[:2]], p, yaw)[0] / 1.0
    if index < 2:
        accel_ego = np.zeros(2)
        yaw_rate = 0.0
    else:
        old = derive_kinematics(poses, index - 1)["velocity"]
        accel_ego = (velocity_ego - np.asarray(old[:2])) / dt
        yaw_rate = angle_delta(yaw, float(prev["yaw"])) / dt
    return {"velocity": [float(velocity_ego[0]), float(velocity_ego[1]), 0.0],
            "acceleration": [float(accel_ego[0]), float(accel_ego[1]), 0.0], "yaw_rate": float(yaw_rate)}


def angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def derive_command(poses: List[Dict[str, Any]], index: int) -> Tuple[str, int]:
    """Reference-route-only command; this is allowed agent input, never object/map truth."""
    last = min(len(poses) - 1, index + 12)
    heading = math.atan2(poses[last]["position"][1] - poses[index]["position"][1],
                         poses[last]["position"][0] - poses[index]["position"][0])
    delta = angle_delta(heading, poses[index]["yaw"])
    if delta > 0.35:
        return "LEFT", 1
    if delta < -0.35:
        return "RIGHT", 0
    return "STRAIGHT", 2


def image_info(path: Path) -> Dict[str, Any]:
    from PIL import Image
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "valid": False}
    if not path.exists():
        result["reason"] = "missing"
        return result
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        result.update({"valid": bool(rgb.ndim == 3 and rgb.shape[2] == 3),
                       "size": [int(rgb.shape[1]), int(rgb.shape[0])], "mode": "RGB",
                       "mean": float(rgb.mean()), "std": float(rgb.std()),
                       "all_black": bool(np.max(rgb) == 0), "all_white": bool(np.min(rgb) == 255)})
        result["fingerprint"] = hashlib.sha256(rgb.tobytes()).hexdigest()[:16]
        if result["all_black"] or result["all_white"]:
            result["valid"] = False
            result["reason"] = "uniform_black_or_white"
    except Exception as exc:
        result["reason"] = f"unreadable:{type(exc).__name__}"
    return result


def compose_worlddreamer_png(camera_images: Dict[str, str], height: int = 224, width: int = 400) -> str:
    """Create the exact vertical order consumed by the bundled API services."""
    from PIL import Image
    canvas = Image.new("RGB", (width, height * 6))
    for i, camera in enumerate(WORLD_DREAMER_CAMERA_ORDER):
        with Image.open(camera_images[camera]) as source:
            resampling = getattr(Image, "Resampling", Image)
            image = source.convert("RGB").resize((width, height), resampling.LANCZOS)
        canvas.paste(image, (0, i * height))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    local = np.array([[length/2, width/2], [length/2, -width/2], [-length/2, -width/2], [-length/2, width/2]])
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _axes(poly: np.ndarray) -> Iterable[np.ndarray]:
    for i in range(4):
        edge = poly[(i + 1) % 4] - poly[i]
        norm = np.linalg.norm(edge)
        if norm:
            yield np.array([-edge[1], edge[0]]) / norm


def obb_intersects(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-9) -> bool:
    for axis in list(_axes(a)) + list(_axes(b)):
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() - tolerance or pb.max() < pa.min() - tolerance:
            return False
    return True


def polygon_mask_ratio(mask: np.ndarray, center_xy: Sequence[float], yaw: float,
                       length: float, width: float, resolution: float = 0.5, bound: float = 50.0) -> float:
    # mask uses WorldDreamer's transposed local-lidar representation: [x_pixel, y_pixel].
    samples = []
    for dx in np.linspace(-length/2, length/2, 9):
        for dy in np.linspace(-width/2, width/2, 5):
            c, s = math.cos(yaw), math.sin(yaw)
            samples.append((center_xy[0] + c*dx - s*dy, center_xy[1] + s*dx + c*dy))
    good = 0
    for x, y in samples:
        ix, iy = int((x + bound) / resolution), int((y + bound) / resolution)
        if 0 <= ix < mask.shape[0] and 0 <= iy < mask.shape[1] and bool(mask[ix, iy]):
            good += 1
    return good / len(samples)


def dac_from_ratios(ratios: Sequence[float]) -> float:
    value = 1.0
    for ratio in ratios:
        if ratio < 0.2:
            return 0.0
        if ratio < 0.5:
            value = min(value, 0.5)
    return value


def trajectory_kinematics(points: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    if len(points) < 2:
        zeros = np.zeros(1)
        return {k: zeros for k in ("speed", "lon_acc", "lat_acc", "jerk", "yaw_rate", "yaw_acc", "curvature", "curvature_rate")}
    delta = np.diff(points, axis=0)
    speed = np.linalg.norm(delta, axis=1) / dt
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    yaw_rate = np.diff(headings) / dt if len(headings) > 1 else np.zeros(1)
    acc = np.diff(speed) / dt if len(speed) > 1 else np.zeros(1)
    jerk = np.diff(acc) / dt if len(acc) > 1 else np.zeros(1)
    yaw_acc = np.diff(yaw_rate) / dt if len(yaw_rate) > 1 else np.zeros(1)
    curvature = yaw_rate / np.maximum(speed[:-1], 1e-3) if len(speed) > 1 else np.zeros(1)
    curvature_rate = np.diff(curvature) / dt if len(curvature) > 1 else np.zeros(1)
    # The existing scorer represents lateral acceleration as zero.  Here preserve that threshold but report geometry.
    lat_acc = speed[:-1] * yaw_rate if len(speed) > 1 else np.zeros(1)
    return {"speed": speed, "lon_acc": acc, "lat_acc": lat_acc, "jerk": jerk,
            "yaw_rate": yaw_rate, "yaw_acc": yaw_acc, "curvature": curvature,
            "curvature_rate": curvature_rate}


COMFORT_LIMITS = {"lat_acc": 4.89, "lon_acc_pos": 2.40, "lon_acc_neg": -4.05,
                  "yaw_acc": 1.93, "jerk": 8.37, "yaw_rate": 0.95}


def comfort(points: np.ndarray, dt: float) -> Dict[str, Any]:
    k = trajectory_kinematics(points, dt)
    violations = []
    if np.max(np.abs(k["lat_acc"])) > COMFORT_LIMITS["lat_acc"]: violations.append("lateral_acceleration")
    if np.max(k["lon_acc"]) > COMFORT_LIMITS["lon_acc_pos"]: violations.append("longitudinal_acceleration")
    if np.min(k["lon_acc"]) < COMFORT_LIMITS["lon_acc_neg"]: violations.append("longitudinal_deceleration")
    if np.max(np.abs(k["jerk"])) > COMFORT_LIMITS["jerk"]: violations.append("jerk")
    if np.max(np.abs(k["yaw_rate"])) > COMFORT_LIMITS["yaw_rate"]: violations.append("yaw_rate")
    if np.max(np.abs(k["yaw_acc"])) > COMFORT_LIMITS["yaw_acc"]: violations.append("yaw_acceleration")
    return {"comfort_score": 0.0 if violations else 1.0, "comfort_violation_types": violations,
            "longitudinal_acceleration_max": float(np.max(np.abs(k["lon_acc"]))),
            "lateral_acceleration_max": float(np.max(np.abs(k["lat_acc"]))),
            "jerk_max": float(np.max(np.abs(k["jerk"]))), "yaw_rate_max": float(np.max(np.abs(k["yaw_rate"]))),
            "yaw_acceleration_max": float(np.max(np.abs(k["yaw_acc"]))),
            "curvature_max": float(np.max(np.abs(k["curvature"]))), "kinematics": k}


def line_progress(route: np.ndarray, start: np.ndarray, point: np.ndarray) -> float:
    best_s, best_d = 0.0, float("inf")
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        v = b - a; length = float(np.linalg.norm(v))
        if length < 1e-8: continue
        u = float(np.clip(np.dot(point - a, v) / (length*length), 0, 1))
        candidate = a + u * v; distance = float(np.linalg.norm(point - candidate))
        if distance < best_d: best_s, best_d = total + u * length, distance
        total += length
    return best_s


def route_progress(route: np.ndarray, start: np.ndarray, points: np.ndarray, valid_steps: int) -> Tuple[float, float]:
    if len(route) < 2 or valid_steps <= 0: return 0.0, 0.0
    s0 = line_progress(route, start, start)
    last = points[min(valid_steps - 1, len(points) - 1)]
    return max(0.0, line_progress(route, start, last) - s0), float(np.linalg.norm(last - start))


def wasserstein(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b: return None
    x, y = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.sort(np.unique(np.r_[x, y]))
    return float(np.trapz(np.abs(np.searchsorted(x, grid, side="right")/len(x) - np.searchsorted(y, grid, side="right")/len(y)), grid))


def js_divergence(a: Sequence[float], b: Sequence[float], bins: int = 20) -> Optional[float]:
    if not a or not b: return None
    lo, hi = min(min(a), min(b)), max(max(a), max(b))
    if hi <= lo: return 0.0
    pa, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False); pb, _ = np.histogram(b, bins=edges, density=False)
    pa, pb = pa / max(pa.sum(), 1), pb / max(pb.sum(), 1); m = (pa + pb) / 2
    def kl(p, q):
        nz = p > 0
        return float(np.sum(p[nz] * np.log2(p[nz] / np.maximum(q[nz], 1e-12))))
    return (kl(pa, m) + kl(pb, m)) / 2


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b: return None
    x, y = np.sort(a), np.sort(b); grid = np.sort(np.unique(np.r_[x, y]))
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right")/len(x) - np.searchsorted(y, grid, side="right")/len(y))))


from driveaudit_eval.utils.aggregation import percentile


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


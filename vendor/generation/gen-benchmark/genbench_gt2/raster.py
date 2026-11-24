"""Deterministic camera-space condition rasters for GT2 model adapters."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

from .constants import IMAGE_SIZE, MAP_CHANNELS
from .geometry import apply_linear, anchor_to_target, box_corners, project_points, scaled_calibration, transform_agent_to_target, transform_points
from .schema import CameraCalibration, SceneInputs


AGENT_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
_MAP_COLORS = {
    "divider": (255, 255, 255), "ped_crossing": (255, 255, 0),
    "boundary": (255, 0, 0), "drivable_area": (0, 255, 0),
    "road_divider": (255, 255, 255), "lane_divider": (255, 255, 255),
    "walkway": (255, 255, 0), "stop_line": (255, 255, 0),
    "carpark_area": (0, 255, 0), "road_block": (255, 0, 0),
}
_VECTOR_STYLE = {
    "divider": ((255, 255, 255), 2), "ped_crossing": ((255, 255, 0), 3),
    "boundary": ((255, 0, 0), 3), "drivable_area": ((0, 255, 0), 2),
}
_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


@dataclass(frozen=True)
class CameraCondition:
    box_rgb: np.ndarray
    depth_by_class: np.ndarray
    hdmap_rgb: np.ndarray
    ray_rgb: np.ndarray


def class_index(category: str):
    try:
        return AGENT_CLASSES.index(category)
    except ValueError:
        return None


def warp_semantic_map(mask: np.ndarray, extent_m: Tuple[float, float, float, float], resolution_m: float, target_to_anchor: np.ndarray, crop_extent_m: Tuple[float, float, float, float], output_resolution_m: float, available_channels: Sequence[str], output_shape: Tuple[int, int]):
    """Nearest-sample a semantic map into target ego, retaining array [C,x,y]."""
    source = np.asarray(mask)
    if source.ndim != 3 or source.shape[0] != len(MAP_CHANNELS):
        raise ValueError("mask must have shape [8,x_forward,y_left]")
    if resolution_m <= 0 or output_resolution_m <= 0:
        raise ValueError("map resolutions must be positive")
    target_to_anchor = np.asarray(target_to_anchor, dtype=np.float64)
    if target_to_anchor.shape != (4, 4) or not np.isfinite(target_to_anchor).all():
        raise ValueError("target_to_anchor must be a finite 4x4 matrix")
    output_x, output_y = output_shape
    x0, _, y0, _ = crop_extent_m
    x = x0 + (np.arange(output_x) + 0.5) * output_resolution_m
    y = y0 + (np.arange(output_y) + 0.5) * output_resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    target = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size), np.ones(xx.size)], axis=1)
    anchor = apply_linear(target_to_anchor, target)[:, :3]
    source_x0, _, source_y0, _ = extent_m
    ix = np.rint((anchor[:, 0] - source_x0) / resolution_m - 0.5).astype(np.int64)
    iy = np.rint((anchor[:, 1] - source_y0) / resolution_m - 0.5).astype(np.int64)
    in_bounds = (ix >= 0) & (ix < source.shape[1]) & (iy >= 0) & (iy < source.shape[2])
    output = np.zeros((len(MAP_CHANNELS), output_x, output_y), dtype=source.dtype)
    availability = np.asarray([channel in available_channels for channel in MAP_CHANNELS], dtype=bool)
    for channel in np.flatnonzero(availability):
        flat = output[channel].ravel()
        flat[in_bounds] = source[channel, ix[in_bounds], iy[in_bounds]]
    return output, availability


def _clip_image_segment(first: np.ndarray, second: np.ndarray, width: int, height: int):
    """Clip a 2D segment to inclusive image coordinates, or return None."""
    delta = second - first
    low, high = 0.0, 1.0
    for p, q in ((-delta[0], first[0]), (delta[0], width - 1 - first[0]), (-delta[1], first[1]), (delta[1], height - 1 - first[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        value = q / p
        if p < 0:
            low = max(low, value)
        else:
            high = min(high, value)
        if low > high:
            return None
    return first + low * delta, first + high * delta


def _draw_polyline(image: np.ndarray, points: np.ndarray, calibration: CameraCalibration, color: Tuple[int, int, int], width: int) -> None:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = apply_linear(np.linalg.inv(calibration.camera_to_ego), homogeneous)[:, :3]
    intrinsics = np.asarray(calibration.intrinsics, dtype=np.float64)
    for first, second in zip(camera_points[:-1], camera_points[1:]):
        if first[2] <= 1e-5 and second[2] <= 1e-5:
            continue
        if first[2] <= 1e-5:
            first = first + ((1e-5 - first[2]) / (second[2] - first[2])) * (second - first)
        elif second[2] <= 1e-5:
            second = first + ((1e-5 - first[2]) / (second[2] - first[2])) * (second - first)
        projected = apply_linear(intrinsics, np.stack((first, second)))
        pixels = projected[:, :2] / projected[:, 2:3]
        clipped = _clip_image_segment(pixels[0], pixels[1], image.shape[1], image.shape[0])
        if clipped is not None:
            start, end = clipped
            # OpenCV bindings are stricter than NumPy about point scalar
            # types.  Keep the conversion at the final drawing boundary.
            start_point = tuple(int(value) for value in np.rint(start))
            end_point = tuple(int(value) for value in np.rint(end))
            cv2.line(image, start_point, end_point, color, width, lineType=cv2.LINE_8)


def _vector_label(line: object, classes: Sequence[str]):
    if not isinstance(line, dict):
        return None
    label = line.get("label")
    if isinstance(label, int) and not isinstance(label, bool):
        return classes[label] if 0 <= label < len(classes) else None
    return label if label in classes else None


def _render_vectors(image: np.ndarray, inputs: SceneInputs, ego_to_anchor: np.ndarray, calibration: CameraCalibration) -> None:
    vectors = inputs.map_condition.vectors
    classes = vectors.get("classes", ()) if isinstance(vectors, dict) else ()
    lines = vectors.get("lines", ()) if isinstance(vectors, dict) else ()
    if not isinstance(classes, (list, tuple)) or not isinstance(lines, (list, tuple)):
        return
    for line in lines:
        label = _vector_label(line, classes)
        style = _VECTOR_STYLE.get(label)
        points = line.get("points") if isinstance(line, dict) else None
        if style is None or not isinstance(points, (list, tuple)) or len(points) < 2:
            continue
        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 3 and np.isfinite(values).all():
            _draw_polyline(image, transform_points(values, ego_to_anchor), calibration, *style)


def _render_semantic_fallback(image: np.ndarray, inputs: SceneInputs, ego_to_anchor: np.ndarray, calibration: CameraCalibration) -> None:
    map_condition = inputs.map_condition
    mask = map_condition.mask
    x0, _, y0, _ = map_condition.extent_m
    inverse = anchor_to_target(ego_to_anchor)
    for channel, name in enumerate(MAP_CHANNELS):
        if name not in map_condition.available_channels:
            continue
        cells = np.argwhere(mask[channel] != 0)
        if not len(cells):
            continue
        anchor_xy = np.column_stack((
            x0 + (cells[:, 0] + 0.5) * map_condition.resolution_m,
            y0 + (cells[:, 1] + 0.5) * map_condition.resolution_m,
        ))
        heights = map_condition.ground_height_m[cells[:, 0], cells[:, 1]].astype(np.float64)
        anchor_points = np.column_stack((anchor_xy, np.where(np.isfinite(heights), heights, 0.0)))
        target_points = apply_linear(inverse, np.column_stack((anchor_points, np.ones(len(anchor_points)))))[:, :3]
        in_crop = (np.abs(target_points[:, 0]) <= 50.0) & (np.abs(target_points[:, 1]) <= 50.0)
        target_points = target_points[in_crop]
        if not len(target_points):
            continue
        cell_ids = np.floor(target_points[:, :2]).astype(np.int64)
        _, first = np.unique(cell_ids, axis=0, return_index=True)
        pixels, visible = project_points(target_points[first], calibration)
        height, width = image.shape[:2]
        for pixel in pixels[visible]:
            u, v = np.rint(pixel).astype(int)
            if 0 <= u < width and 0 <= v < height:
                image[v, u] = _MAP_COLORS[name]


def _render_boxes(box_image: np.ndarray, depth_image: np.ndarray, inputs: SceneInputs, frame_index: int, ego_to_anchor: np.ndarray, calibration: CameraCalibration) -> None:
    rendered = []
    for agent in inputs.agents_by_frame.get(frame_index, ()):
        index = class_index(agent.category)
        if index is None:
            continue
        corners = box_corners(transform_agent_to_target(agent, ego_to_anchor))
        pixels, visible = project_points(corners, calibration)
        camera_xyz = apply_linear(np.linalg.inv(calibration.camera_to_ego), np.column_stack((corners, np.ones(8))))[:, :3]
        if not visible.any():
            continue
        for first, second in _EDGES:
            if visible[first] and visible[second]:
                # A box can be in front of the camera while projecting far
                # outside the image (e.g. an object almost on the camera
                # plane).  Passing such coordinates directly to OpenCV can
                # overflow its C int parser.  Clip in float space first.
                clipped = _clip_image_segment(
                    pixels[first], pixels[second], box_image.shape[1], box_image.shape[0]
                )
                if clipped is None:
                    continue
                point_a, point_b = clipped
                point_a = tuple(int(value) for value in np.rint(point_a))
                point_b = tuple(int(value) for value in np.rint(point_b))
                cv2.line(box_image, point_a, point_b, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        rendered.append((float(np.mean(camera_xyz[visible, 2])), index, pixels, visible, camera_xyz[:, 2]))
    for mean_depth, index, pixels, visible, depths in sorted(rendered, reverse=True, key=lambda item: item[0]):
        near = np.argsort(depths)[:4]
        if not np.all(visible[near]):
            continue
        # Keep polygon coordinates representable even when the projected
        # face extends many orders of magnitude beyond the image.
        polygon_pixels = np.clip(
            pixels[near],
            [0.0, 0.0],
            [box_image.shape[1] - 1.0, box_image.shape[0] - 1.0],
        )
        polygon = cv2.convexHull(np.rint(polygon_pixels).astype(np.int32)).reshape(-1, 2)
        plane = depth_image[:, :, index].copy()
        cv2.fillConvexPoly(plane, polygon, int(np.clip(np.rint(mean_depth * 3.0), 0, 255)), lineType=cv2.LINE_8)
        depth_image[:, :, index] = plane


def camera_ray_encoding(calibration: CameraCalibration, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = apply_linear(np.linalg.inv(calibration.intrinsics), np.stack((u, v, np.ones_like(u)), axis=-1))
    directions = apply_linear(np.asarray(calibration.camera_to_ego, dtype=np.float64)[:3, :3], rays)
    low, high = directions.min(), directions.max()
    if high <= low:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return np.rint((directions - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)


def render_camera_conditions(inputs: SceneInputs, frame_index: int, camera: str, size: Tuple[int, int]) -> CameraCondition:
    if camera not in inputs.calibrations:
        raise ValueError("camera is not calibrated: {}".format(camera))
    if frame_index < 0 or frame_index >= len(inputs.ego_to_anchor):
        raise ValueError("frame index is outside ego_to_anchor")
    width, height = size
    calibration = scaled_calibration(inputs.calibrations[camera], IMAGE_SIZE, size)
    box_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth_by_class = np.zeros((height, width, len(AGENT_CLASSES)), dtype=np.uint8)
    hdmap_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    ego_to_anchor = inputs.ego_to_anchor[frame_index]
    _render_boxes(box_rgb, depth_by_class, inputs, frame_index, ego_to_anchor, calibration)
    _render_vectors(hdmap_rgb, inputs, ego_to_anchor, calibration)
    _render_semantic_fallback(hdmap_rgb, inputs, ego_to_anchor, calibration)
    return CameraCondition(box_rgb, depth_by_class, hdmap_rgb, camera_ray_encoding(calibration, size))

"""BaseDreamer raster conventions, kept independent of model imports for tests."""
import cv2
import numpy as np

BOX_EDGES = ((1, ((0,3),(1,2),(3,2),(3,7),(4,7),(2,6),(5,6),(6,7))),
             (2, ((0,1),(0,4),(1,5),(4,5))))


def _clip_parameters(first, second, bounds):
    delta = second - first
    enter, leave = 0., 1.
    for axis, (low, high) in enumerate(bounds):
        if abs(delta[axis]) < 1e-15:
            if first[axis] < low or first[axis] > high:
                return None
            continue
        a, b = (low-first[axis])/delta[axis], (high-first[axis])/delta[axis]
        enter, leave = max(enter, min(a,b)), min(leave, max(a,b))
        if enter > leave:
            return None
    return enter, leave


def clip_native_polyline(points, half_extent=50.):
    """Match the released vector map's 100m patch before camera projection."""
    runs, current = [], []
    for first, second in zip(points[:-1], points[1:]):
        interval = _clip_parameters(first, second, ((-half_extent,half_extent),(-half_extent,half_extent)))
        if interval is None:
            if len(current) > 1: runs.append(np.array(current))
            current = []
            continue
        start, end = (first + t*(second-first) for t in interval)
        if current and not np.allclose(current[-1], start):
            if len(current) > 1: runs.append(np.array(current))
            current = []
        if not current: current.append(start)
        current.append(end)
    if len(current) > 1: runs.append(np.array(current))
    return runs


def _line(canvas, first, second, value, width, kind=cv2.LINE_8):
    # Standard coordinates retain upstream integer truncation exactly. At the
    # camera plane positive depths can create coordinates outside cv2's int32.
    # Clip only that exceptional case in floating point, including stroke width.
    if max(np.abs(first).max(), np.abs(second).max()) > 2**30:
        bounds = ((-float(width), float(canvas.shape[1]+width)),
                  (-float(width), float(canvas.shape[0]+width)))
        interval = _clip_parameters(np.asarray(first), np.asarray(second), bounds)
        if interval is None: return
        first, second = [first+t*(second-first) for t in interval]
    cv2.line(canvas, tuple(np.asarray(first).astype(int)),
             tuple(np.asarray(second).astype(int)), value, width, kind)


def bev_raster(lines_native, drivable):
    # Upstream visualize_bev_hdmap draws [class,y,x], then returns [class,x,y].
    canvas = np.zeros((3, 200, 200), dtype=np.uint8)
    for channel, points in lines_native:
        pixels = ((np.asarray(points)[:, :2] + 50) * 2).astype(int)
        for point in pixels:
            cv2.circle(canvas[channel], tuple(point), 1, 1, -1)
        for first, second in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas[channel], tuple(first), tuple(second), 1, 1)
    return np.concatenate((canvas.transpose(0, 2, 1), drivable[None]), axis=0)


def _camera_points(points, calibration):
    return np.einsum('ij,nj->ni', np.linalg.inv(calibration.camera_to_ego),
        np.column_stack((points, np.ones(len(points)))), optimize=False)[:, :3]


def _pixels(points_camera, calibration):
    projected = np.einsum('ij,nj->ni', calibration.intrinsics, points_camera, optimize=False)
    return projected[:, :2] / projected[:, 2:3]


def camera_raster(lines_ego, ground_ego, boxes_ego, calibration):
    """Original full-resolution drawing, class isolation, and stride/crop."""
    canvas = np.zeros((14, 900, 1600), dtype=np.uint8)
    for channel, points in lines_ego:
        points_camera = _camera_points(points, calibration)
        points_camera = points_camera[points_camera[:, 2] > 0]
        if len(points_camera) > 1:
            pixels = _pixels(points_camera, calibration)
            for first, second in zip(pixels[:-1], pixels[1:]):
                _line(canvas[channel], first, second, 1, 4)
    ground_camera = _camera_points(ground_ego, calibration)
    ground_camera = ground_camera[ground_camera[:, 2] > 0]
    if len(ground_camera) > 1:
        pixels = _pixels(ground_camera, calibration).astype(int)
        # Circles entirely outside the canvas cannot affect the raster.
        pixels = pixels[(pixels[:, 0] >= -20) & (pixels[:, 0] < 1620) &
                        (pixels[:, 1] >= -20) & (pixels[:, 1] < 920)]
        for pixel in pixels:
            cv2.circle(canvas[3], tuple(pixel), 20, 1, -1)
    visible = []
    for corners, label in boxes_ego:
        points_camera = _camera_points(corners, calibration)
        if (points_camera[:, 2] > 0).all():
            visible.append((points_camera[:, 2].min(), _pixels(points_camera, calibration), label))
    for _, pixels, label in sorted(visible, key=lambda row: -row[0]):
        for value, edges in BOX_EDGES:
            for first, second in edges:
                _line(canvas[4 + label], pixels[first], pixels[second], value, 4, cv2.LINE_AA)
    return canvas[:, ::4, ::4][:, 1:, :]

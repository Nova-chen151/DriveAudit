"""GT2 geometry in the nuScenes conditioning style used by CTSD training.

GT2 does not provide lane polygons. We render only available drivable-area and
pedestrian-crossing contours; dividers are not mislabeled as lane polygons.
"""
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image, ImageDraw
from genbench_gt2.constants import IMAGE_SIZE, MAP_CHANNELS
from genbench_gt2.geometry import box_corners, scaled_calibration, transform_points

RENDERER_VERSION = 'opendwm-nuscenes-contours-v1'
BOX_COLORS = {
    'pedestrian': (255,0,0), 'bicycle': (128,255,0),
    'motorcycle': (0,255,128), 'bus': (128,0,255), 'car': (0,0,255),
    'construction_vehicle': (128,128,255), 'emergency': (255,128,128),
    'trailer': (255,255,255), 'truck': (255,255,0),
}
EDGES = ((0,1),(0,2),(1,3),(2,3),(0,4),(1,5),(2,6),(3,7),
         (4,5),(4,6),(5,7),(6,7),(6,3),(6,5))
NATIVE_SIZE = (1600,900)


def _line(draw, a, b, color, far=512.):
    # Clip in projected homogeneous coordinates, as native project_line does.
    delta = b-a
    low, high = 0., 1.
    for sign, bound in ((1., .05), (-1., -far)):
        origin, direction = sign*a[2], sign*delta[2]
        if abs(direction) < 1e-12:
            if origin < bound:
                return
        elif direction > 0:
            low = max(low, (bound-origin)/direction)
        else:
            high = min(high, (bound-origin)/direction)
    if low > high:
        return
    p, q = a+low*delta, a+high*delta
    draw.line((* (p[:2]/p[2]), * (q[:2]/q[2])), fill=color, width=8)


def prepare_map(map_condition):
    """Extract connected boundaries once per scene, retaining holes and height."""
    result = []
    x0, _, y0, _ = map_condition.extent_m
    for name, color in (('drivable_area',(0,0,255)), ('ped_crossing',(255,0,0))):
        if name not in map_condition.available_channels:
            continue
        mask = np.asarray(map_condition.mask[MAP_CHANNELS.index(name)] != 0, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cells = contour[:,0,:][:,::-1]  # OpenCV x=column=y_left, y=row=x_forward.
            if len(cells) < 3:
                continue
            heights = map_condition.ground_height_m[cells[:,0],cells[:,1]]
            xyz = np.column_stack((x0+(cells[:,0]+.5)*map_condition.resolution_m,
                                   y0+(cells[:,1]+.5)*map_condition.resolution_m,
                                   np.where(np.isfinite(heights),heights,0.)))
            result.append((np.concatenate((xyz,xyz[:1])),color))
    return result


def render_camera_conditions(inputs, frame_index, camera, size, map_geometry=None):
    calibration = scaled_calibration(inputs.calibrations[camera], IMAGE_SIZE, NATIVE_SIZE)
    camera_from_target = np.linalg.inv(calibration.camera_to_ego)
    def project(anchor_points):
        target = transform_points(anchor_points, inputs.ego_to_anchor[frame_index])
        # Avoid node-dependent BLAS GEMM corruption for long map contours.
        homogeneous = np.column_stack((target, np.ones(len(target))))
        cam = np.einsum("ij,nj->ni", camera_from_target, homogeneous, optimize=False)[:, :3]
        return np.einsum("ij,nj->ni", calibration.intrinsics, cam, optimize=False)
    boxes, maps = Image.new('RGB',NATIVE_SIZE), Image.new('RGB',NATIVE_SIZE)
    draw = ImageDraw.Draw(boxes)
    for agent in inputs.agents_by_frame.get(frame_index, ()):
        color = BOX_COLORS.get(agent.category)
        if color is None:
            continue
        corners = project(box_corners(agent))
        for first, second in EDGES:
            _line(draw,corners[first],corners[second],color)
    draw = ImageDraw.Draw(maps)
    geometry = prepare_map(inputs.map_condition) if map_geometry is None else map_geometry
    for points, color in geometry:
        projected = project(points)
        for first, second in zip(projected[:-1],projected[1:]):
            _line(draw,first,second,color,far=65.)
    return SimpleNamespace(box_rgb=np.asarray(boxes.resize(size,Image.Resampling.BILINEAR)),
                           hdmap_rgb=np.asarray(maps.resize(size,Image.Resampling.BILINEAR)))

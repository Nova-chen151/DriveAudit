"""GT2 adapter for Panacea's released nuScenes 19-channel representation.

Drawing primitives retain the released renderer conventions (including BGR map
channel values and box face colors). Camera transforms are camera-to-ego: the
upstream misleadingly named lidar2img also uses sensor2ego, not lidar coordinates.
"""
import copy
from typing import List, Tuple, Union
import cv2
import numpy as np
from shapely.geometry import MultiPoint, box, LineString
from genbench_gt2.constants import IMAGE_SIZE
from genbench_gt2.geometry import box_corners, scaled_calibration, transform_agent_to_target, transform_points
from genbench_gt2.raster import AGENT_CLASSES, CameraCondition, class_index

RENDERER_VERSION = 'panacea-native-v2'
COLORS = np.array([[255,255,255],[128,64,128],[244,35,232],[70,70,70],
    [102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],
    [107,142,35],[152,251,152]])
MAP_COLORS = {'divider': (0,0,255), 'boundary': (0,255,0), 'ped_crossing': (255,0,0)}

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (3840, 2160)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def draw_rect(img, selected_corners, color, linewidth):
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(img,
                    (int(prev[0]), int(prev[1])),
                    (int(corner[0]), int(corner[1])),
                    color, linewidth)
        prev = corner

class NativeRenderer:
    classes = AGENT_CLASSES

    def draw_bboxes(self, target, bboxes, labels, depths, colors, thickness=12):
        
        img = np.ones((target.shape[0], target.shape[1], len(self.classes))) * 255 
        img = img.copy().astype(np.uint8)
        if labels is None or len(labels) == 0:
            return img

        for i, name in enumerate(self.classes):
            mask = (labels == i)
            lab = labels[mask]
            dep = depths[mask]
            if bboxes is not None: bbox = bboxes[mask] 
            if bboxes is None or len(bbox) == 0:
                continue
            dep = dep * 3
            for j in range(len(bbox)):
                xmin,ymin,xmax,ymax = bbox[j]
                img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] = np.where(img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] > dep[j], dep[j], img[int(ymin) : int(ymax), int(xmin) : int(xmax), i])

        return img

    def draw_corners(self, target, corners, labels, depths2d, colors, linewidth=4):
        
        img = np.ones((target.shape[0], target.shape[1], 3)) * 255 
        img = img.copy().astype(np.uint8)

        if corners is None or len(corners) == 0:
            return img

        sort_indexes = np.argsort(depths2d)[::-1]
        corners = corners[sort_indexes]
        labels = labels[sort_indexes]
        depths2d = depths2d[sort_indexes]
        for j in range(len(corners)):
            color = colors[labels[j] + 1]
            color = (int(color[0]), int(color[1]), int(color[2]))

            points = corners[j, [0, 1, 2, 3]]
            # points =  np.array([[int(corner[j, 0, 0]), int(corner[j, 0, 1])], [int(corner[j, 1, 0]), int(corner[j, 1, 1])], [int(corner[j, 2, 0]), int(corner[j, 2, 1])], [int(corner[j, 3, 0]), int(corner[j, 3, 1])]])
            points =  np.array([[int(corners[j, 4, 0]), int(corners[j, 4, 1])], [int(corners[j, 5, 0]), int(corners[j, 5, 1])], [int(corners[j, 6, 0]), int(corners[j, 6, 1])], [int(corners[j, 7, 0]), int(corners[j, 7, 1])]])
            points = points.reshape(-1, 1, 2)
            points[..., 0] = np.clip(points[..., 0], 0, target.shape[1]) 
            points[..., 1] = np.clip(points[..., 1], 0, target.shape[0]) 
            ori_color = (int(color[0]*0.5 + 255*0.5), int(color[1]*0.5 + 255*0.5), int(color[2]*0.5 + 255*0.5))
            cv2.fillPoly(img, [points], ori_color)

            for i in range(4):
                cv2.line(img,
                    (int(corners[j][i][0]), int(corners[j][i][1])),
                    (int(corners[j][i + 4][0]), int(corners[j][i + 4][1])),
                    color[::-1], linewidth)

            draw_rect(img, corners[j][:4], color[::-1], linewidth)
            draw_rect(img, corners[j][4:], color[::-1], linewidth)
        
        return img

    def _get_2d_annos(self, shape, source_bbox_3d, source_corner_3d, source_label_3d, lidar2img):
        gt_bboxes_3d = source_bbox_3d
        gt_label_3d = source_label_3d
        corners_3d = source_corner_3d
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)

        gt_bbox2d = []
        gt_depth2d = []
        gt_label2d = []
        gt_corners3d = []
        for i in range(6):
            lidar2img_rt = np.array(lidar2img[i])
            pts_2d = np.einsum('ij,nj->ni', lidar2img_rt, pts_4d, optimize=False)
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=51.2)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            
            H, W = shape[0], shape[1]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
            imgfov_pts_depth = pts_2d[..., 2].reshape(num_bbox, 8)
            mask = imgfov_pts_depth.mean(1) > 0.1

            if mask.sum() == 0:
                gt_bbox2d.append([])
                gt_depth2d.append([])
                gt_label2d.append([]) 
                gt_corners3d.append([])
                continue

            imgfov_pts_2d = imgfov_pts_2d[mask]
            imgfov_pts_depth = imgfov_pts_depth[mask]
            imgfov_pts_label= gt_label_3d[mask]

            bbox = []
            label = []
            depth = []
            corners3d = []
            for j, corner_coord in enumerate(imgfov_pts_2d):
                final_coords = post_process_coords(corner_coord, imsize = (W,H))
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                    if ((max_x - min_x) >W-100) and ((max_y - min_y)>H-100):
                        continue
                    bbox.append([min_x, min_y, max_x, max_y])
                    label.append(imgfov_pts_label[j])
                    depth.append(imgfov_pts_depth[j].mean())
                    corners3d.append(copy.deepcopy(corner_coord))
            gt_bbox2d.append(np.array(bbox))
            gt_depth2d.append(np.array(depth))
            gt_label2d.append(np.array(label)) 
            gt_corners3d.append(np.array(corners3d)) 
        bbox2d_info = {
            'gt_bbox2d' : gt_bbox2d,
            'gt_depth2d' : gt_depth2d,
            'gt_label2d' : gt_label2d,
            'gt_corners3d': gt_corners3d
        }

        return bbox2d_info


def render_rays(calibrations, size):
    """Released min/max normalization spans all six cameras, all pixels/axes."""
    width, height = size
    # Preserve upstream meshgrid/reshape layout as well as its uint8 truncation.
    coords = np.stack(np.meshgrid(np.arange(width), np.arange(height), [1., 2.])).transpose(1, 2, 3, 0)
    coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
    coords[..., :2] *= coords[..., 2:3]
    coords = coords.reshape(1, width, height, 2, 4, 1)
    inverses = []
    for calibration in calibrations.values():
        project = np.eye(4)
        project[:3, :3] = calibration.intrinsics
        inverses.append(calibration.camera_to_ego @ np.linalg.inv(project))
    points = np.matmul(np.asarray(inverses).reshape(-1, 1, 1, 1, 4, 4), coords).squeeze(-1)[..., :3]
    points = points.transpose(0, 2, 1, 3, 4)
    directions = points[..., 1, :] - points[..., 0, :]
    low, high = directions.min(), directions.max()
    result = ((directions-low)/(high-low)*255).astype(np.uint8)
    return dict(zip(calibrations, result))


def render_map(inputs, frame, calibration, size):
    """Only the three vector classes seen in training, on a white canvas."""
    width, height = size
    # Native Renderer uses float64 canvases; OpenCV therefore does not AA blend.
    image = np.full((height, width, 3), 255., dtype=np.float64)
    vectors = inputs.map_condition.vectors
    classes = vectors.get('classes', ())
    camera_inverse = np.linalg.inv(calibration.camera_to_ego)
    roi = box(-30, -15, 30, 15)
    for vector in vectors.get('lines', ()):
        label = vector.get('label')
        label = classes[label] if isinstance(label, int) and 0 <= label < len(classes) else label
        if label not in MAP_COLORS:
            continue
        points = np.asarray(vector.get('points', ()), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        local = transform_points(points, inputs.ego_to_anchor[frame])
        # Native VectorizeMap is 2D ego-ground, clipped to its 60m x 30m ROI.
        clipped = LineString(local[:, :2]).intersection(roi)
        lines = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]
        for line in lines:
            if line.is_empty or line.geom_type != 'LineString' or line.length == 0:
                continue
            sampled = np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, 200)])
            points4 = np.column_stack((sampled, np.zeros(len(sampled)), np.ones(len(sampled))))
            # Avoid NumPy 1.23/OpenBLAS GEMM corruption on some CPU nodes.
            camera = np.einsum('ij,nj->ni', camera_inverse, points4, optimize=False)[:, :3]
            projected = np.einsum('ij,nj->ni', calibration.intrinsics, camera, optimize=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                uv = projected[:, :2] / projected[:, 2:3]
            valid = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width-1) & (uv[:, 1] >= 0) & (uv[:, 1] < height-1)
            uv = np.round(uv[valid]).astype(np.int32)
            for start, end in zip(uv[:-1], uv[1:]):
                cv2.line(image, tuple(start), tuple(end), MAP_COLORS[label], 4, lineType=cv2.LINE_AA)
    return image.astype(np.uint8)


def render_frame_conditions(inputs, frame, cameras, size=(512, 256), calibrations=None, rays=None):
    calibrations = calibrations or {name: scaled_calibration(inputs.calibrations[name], IMAGE_SIZE, size) for name in cameras}
    rays = rays or render_rays(calibrations, size)
    renderer = NativeRenderer()
    agents = [transform_agent_to_target(a, inputs.ego_to_anchor[frame]) for a in inputs.agents_by_frame.get(frame, ()) if class_index(a.category) is not None]
    # Match released inference CustomObjectRangeFilter in ego XY.
    agents = [a for a in agents if np.all(np.abs(a.center_xyz[:2]) < 51.2)]
    corners = np.asarray([box_corners(a)[[0,1,3,2,4,5,7,6]] for a in agents]).reshape(-1,8,3)
    labels = np.array([class_index(a.category) for a in agents], dtype=int)
    projections = []
    for name in cameras:
        project = np.eye(4); project[:3,:3] = calibrations[name].intrinsics
        projections.append(project @ np.linalg.inv(calibrations[name].camera_to_ego))
    width, height = size
    annotations = renderer._get_2d_annos((height,width), None, corners, labels, projections)
    blank = np.empty((height,width,3), dtype=np.uint8)
    result = {}
    for i, name in enumerate(cameras):
        bboxes, depths, labels2, corners2 = (annotations[key][i] for key in ('gt_bbox2d','gt_depth2d','gt_label2d','gt_corners3d'))
        result[name] = CameraCondition(
            renderer.draw_corners(blank, corners2, labels2, depths, COLORS, linewidth=2),
            renderer.draw_bboxes(blank, bboxes, labels2, depths, COLORS),
            render_map(inputs, frame, calibrations[name], size), rays[name])
    return result

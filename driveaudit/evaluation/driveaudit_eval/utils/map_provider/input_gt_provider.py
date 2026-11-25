"""Queries on exported input-gt drivable cells in the fixed history-last-ego frame."""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from driveaudit_eval.utils.map_provider.opendrive_provider import OpenDriveMapProvider
from driveaudit_eval.utils.map_provider.geometry import points_xy


class InputGTMapProvider(OpenDriveMapProvider):
    coordinate_system = 'history_last_ego'
    provider_version = 'input_gt_exported_raster_v1'

    def __init__(self, scene_path, native_config=None):
        self.scene_path = Path(scene_path)
        self.native = None
        folder = self.scene_path / 'map'
        self.meta = json.loads((folder / 'map_meta.json').read_text())
        if self.meta.get('array_axes') != 'channel,x_forward,y_left':
            raise ValueError('unsupported exported map axes')
        if self.meta.get('coordinate_frame') != 'history_last_ego_x_forward_y_left_meters':
            raise ValueError('unsupported exported map coordinate frame')
        if 'drivable_area' not in self.meta.get('available_channels', []):
            raise FileNotFoundError('drivable_area channel is unavailable')
        with np.load(folder / 'semantic_map.npz', allow_pickle=False) as data:
            if 'observed_surface' in data:
                raise FileNotFoundError('partial observed-surface map: unobserved cells are unknown; binary off-road metrics unavailable')
            mask = data['mask'][self.meta['channels'].index('drivable_area')].astype(bool)
        self.extent = np.asarray(self.meta['extent_m'], float)
        resolution = float(self.meta['resolution_m'])
        if not np.allclose(np.asarray(mask.shape) * resolution, self.extent[[1, 3]] - self.extent[[0, 2]]):
            raise ValueError('map raster shape, extent and resolution disagree')
        cells = []
        for row_index, row in enumerate(mask):
            edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                x = self.extent[0] + row_index * resolution
                cells.append(box(x, self.extent[2] + start * resolution, x + resolution, self.extent[2] + end * resolution))
        super().__init__({'drivable': unary_union(cells), 'lanes': [],
                          'semantic_support': {'drivable_area': True, 'directed_lane_centerlines': False}}, folder / 'semantic_map.npz')
        if native_config and native_config.get('enabled', True):
            if native_config.get('type', 'nuscenes') != 'nuscenes':
                raise ValueError('only native nuScenes lane supplementation is supported')
            context = json.loads((self.scene_path / 'scene_context.json').read_text())
            if context.get('dataset') == 'nuscenes':
                from driveaudit_eval.utils.map_provider.nuscenes_provider import NuScenesMapProvider
                from driveaudit_eval.utils.map_provider.cache import file_sha256
                from pyquaternion import Quaternion
                provenance_path = self.scene_path / 'generation_provenance.json'
                provenance = json.loads(provenance_path.read_text())
                anchor = np.asarray(provenance['anchor_ego_to_world'], dtype=float)
                if anchor.shape != (4, 4) or not np.isfinite(anchor).all() or not np.allclose(anchor[3], [0, 0, 0, 1]):
                    raise ValueError('invalid anchor_ego_to_world')
                # Exported maps use native quaternion yaw only.
                self.anchor_yaw = float(Quaternion(matrix=anchor[:3, :3]).yaw_pitch_roll[0])
                c, s = np.cos(self.anchor_yaw), np.sin(self.anchor_yaw)
                self.anchor_rotation = np.asarray([[c, -s], [s, c]])
                self.anchor_translation = anchor[:2, 3]
                self.native = NuScenesMapProvider(dataroot=native_config['dataroot'], location=context['location'],
                    cache_dir=native_config.get('cache_dir'),
                    discretization_m=float(native_config.get('discretization_m', .5)),
                    max_lane_distance_m=float(native_config.get('max_lane_distance_m', 10.0)))
                self.native_transform = {'anchor_ego_to_world': anchor.tolist(), 'anchor_yaw_radians': self.anchor_yaw,
                    'policy': 'native quaternion yaw-only XY map convention, matching exported semantic raster and generation vectors',
                    'anchor_provenance_sha256': file_sha256(provenance_path),
                    'scene_context_sha256': file_sha256(self.scene_path / 'scene_context.json'),
                    'GT_trajectory_used_to_fit_map': False}
                self.map_data['semantic_support']['directed_lane_centerlines'] = True

    def _check_extent(self, value):
        points = points_xy(value)
        if np.any((points[:, 0] < self.extent[0]) | (points[:, 0] >= self.extent[1]) |
                  (points[:, 1] < self.extent[2]) | (points[:, 1] >= self.extent[3])):
            raise ValueError('query outside exported map extent; outside cells are unknown')
        return points

    def is_drivable(self, value):
        return super().is_drivable(self._check_extent(value))

    def footprint_coverage(self, centers, headings, length, width):
        from driveaudit_eval.utils.map_provider.geometry import obb_polygon
        for center, heading in zip(points_xy(centers), headings):
            polygon = obb_polygon(center, heading, length, width)
            self._check_extent(np.asarray(polygon.exterior.coords))
        return super().footprint_coverage(centers, headings, length, width)

    def _world(self, value):
        return points_xy(value) @ self.anchor_rotation.T + self.anchor_translation

    def nearest_lane_center(self, value):
        if self.native is None:
            return [{'status': 'unavailable', 'reason': 'export contains no directed lane centerlines; native supplement not enabled'} for _ in points_xy(value)]
        result = self.native.nearest_lane_center(self._world(value))
        for item in result:
            if 'point' in item:
                item['point'] = ((np.asarray(item['point']) - self.anchor_translation) @ self.anchor_rotation).tolist()
            if 'heading' in item:
                item['heading'] = float((item['heading'] - self.anchor_yaw + np.pi) % (2 * np.pi) - np.pi)
        return result

    def lane_direction(self, value):
        return np.asarray([item.get('heading', np.nan) if item.get('status') == 'matched' else np.nan for item in self.nearest_lane_center(value)])

    def lane_id(self, value):
        return [item.get('lane_id') if item.get('status') == 'matched' else None for item in self.nearest_lane_center(value)]

    def distance_to_lane_boundary(self, value):
        return self.native.distance_to_lane_boundary(self._world(value)) if self.native else super().distance_to_lane_boundary(value)

    def road_segment_id(self, value):
        return self.native.road_segment_id(self._world(value)) if self.native else super().road_segment_id(value)

    def route_progress(self, trajectory_xy, route_context=None):
        if self.native:
            return self.native.route_progress(self._world(trajectory_xy), self._world(route_context) if route_context is not None else None)
        return super().route_progress(trajectory_xy, route_context)

    def query_metadata(self):
        metadata = dict(super().query_metadata(), coordinate_system=self.coordinate_system,
                    raster_resolution_m=self.meta['resolution_m'], geometry='exact union of exported drivable cell rectangles',
                    source_map_status=self.meta.get('status'), extent_m=self.meta['extent_m'],
                    unavailable_metrics=['route_completion'] if self.native else ['lane_direction', 'wrong_way', 'lane_center_distance', 'route_completion'])
        if self.native:
            metadata.update(native_lane_supplement=self.native.query_metadata(), native_coordinate_transform=self.native_transform,
                            drivable_policy='exported raster retained; native data supplements directed lanes only')
        return metadata

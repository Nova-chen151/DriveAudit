from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union

from driveaudit_eval.utils.map_provider.base import MapProvider
from driveaudit_eval.utils.map_provider.cache import cache_key, file_sha256, load_cache, save_cache
from driveaudit_eval.utils.map_provider.geometry import footprint_coverage, points_xy


MAP_LOCATIONS = ("boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown")
DEFAULT_DRIVABLE_LAYERS = ("drivable_area", "road_segment", "road_block", "lane", "lane_connector", "carpark_area")


class NuScenesMapProvider(MapProvider):
    provider_version = "nuscenes_map_provider_v1"

    def __init__(self, dataroot, location, cache_dir=None, discretization_m=0.5,
                 drivable_layers=DEFAULT_DRIVABLE_LAYERS, max_lane_distance_m=10.0):
        if location not in MAP_LOCATIONS:
            raise ValueError("unknown nuScenes map location: %s" % location)
        self.dataroot = Path(dataroot).resolve(); self.location = location
        self.discretization_m = float(discretization_m); self.drivable_layers = tuple(drivable_layers)
        self.max_lane_distance_m = float(max_lane_distance_m)
        source = self.dataroot / "maps/expansion" / (location + ".json")
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = file_sha256(source)
        try: devkit_version = version("nuscenes-devkit")
        except PackageNotFoundError: devkit_version = "not_available"
        identity = {"provider": self.provider_version, "location": location, "source_sha256": source_hash,
                    "version": "v1.0-trainval", "discretization_m": self.discretization_m,
                    "drivable_layers": self.drivable_layers, "devkit_version": devkit_version,
                    "code_sha256": file_sha256(Path(__file__))}
        cache_path = Path(cache_dir) / (location + "_" + cache_key(identity) + ".pkl") if cache_dir else None
        data = load_cache(cache_path) if cache_path else None
        self.cache_hit = data is not None
        if data is None:
            data = self._build()
            if cache_path:
                save_cache(cache_path, data)
        self.cache_path = str(cache_path) if cache_path else None
        self.identity = identity; self._install(data)

    def _polygons(self, api, layer):
        polygons, tokens = [], []
        for record in getattr(api, layer, []):
            polygon_tokens = record.get("polygon_tokens") or [record.get("polygon_token")]
            for token in polygon_tokens:
                if not token:
                    continue
                polygon = api.extract_polygon(token).buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon); tokens.append(record["token"])
        return polygons, tokens

    def _build(self):
        from nuscenes.map_expansion.map_api import NuScenesMap

        api = NuScenesMap(dataroot=str(self.dataroot), map_name=self.location)
        drivable_parts = []
        layer_counts = {}
        for layer in self.drivable_layers:
            polygons, _ = self._polygons(api, layer)
            drivable_parts.extend(polygons); layer_counts[layer] = len(polygons)
        lane_records = [("lane", record) for record in api.lane] + [("lane_connector", record) for record in api.lane_connector]
        lane_tokens = [record["token"] for _, record in lane_records]
        centerlines = api.discretize_lanes(lane_tokens, self.discretization_m)
        lane_points, lane_headings, lane_point_tokens, lane_point_layers = [], [], [], []
        lane_polygons = {}
        for layer, record in lane_records:
            token = record["token"]; line = np.asarray(centerlines.get(token, []), dtype=float)
            if len(line):
                lane_points.extend(line[:, :2]); lane_headings.extend(line[:, 2]); lane_point_tokens.extend([token] * len(line)); lane_point_layers.extend([layer] * len(line))
            polygon_token = record.get("polygon_token")
            if polygon_token:
                lane_polygons[token] = api.extract_polygon(polygon_token).buffer(0)
        road_polygons, road_tokens = self._polygons(api, "road_segment")
        return {"drivable": unary_union(drivable_parts).buffer(0), "lane_points": np.asarray(lane_points, dtype=float),
                "lane_headings": np.asarray(lane_headings, dtype=float), "lane_point_tokens": lane_point_tokens,
                "lane_point_layers": lane_point_layers, "lane_polygons": lane_polygons,
                "road_polygons": road_polygons, "road_tokens": road_tokens, "layer_counts": layer_counts}

    def _install(self, data):
        self.drivable = data["drivable"]; self.lane_points = np.asarray(data["lane_points"], dtype=float)
        self.lane_headings = np.asarray(data["lane_headings"], dtype=float); self.lane_point_tokens = list(data["lane_point_tokens"])
        self.lane_point_layers = list(data["lane_point_layers"]); self.lane_polygons = data["lane_polygons"]
        self.road_polygons = data["road_polygons"]; self.road_tokens = data["road_tokens"]; self.layer_counts = data["layer_counts"]
        self._lane_tree = cKDTree(self.lane_points) if len(self.lane_points) else None

    def is_drivable(self, value):
        return np.asarray([self.drivable.covers(Point(point)) for point in points_xy(value)], dtype=bool)

    def signed_distance_to_drivable(self, value):
        output = []
        for point in points_xy(value):
            shape = Point(point)
            output.append(shape.distance(self.drivable.boundary) if self.drivable.covers(shape) else -shape.distance(self.drivable))
        return np.asarray(output, dtype=float)

    def nearest_lane_center(self, value):
        pts = points_xy(value)
        if self._lane_tree is None:
            return [{"status": "unmatched_lane"} for _ in pts]
        distances, indices = self._lane_tree.query(pts, k=1)
        return [{"status": "matched" if float(distance) <= self.max_lane_distance_m else "unmatched_lane",
                 "distance": float(distance), "point": self.lane_points[int(index)].tolist(),
                 "heading": float(self.lane_headings[int(index)]), "lane_id": self.lane_point_tokens[int(index)],
                 "layer": self.lane_point_layers[int(index)]} for distance, index in zip(distances, indices)]

    def lane_direction(self, value):
        return np.asarray([item.get("heading", np.nan) if item.get("status") == "matched" else np.nan for item in self.nearest_lane_center(value)])

    def distance_to_lane_center(self, value):
        return np.asarray([item.get("distance", np.nan) for item in self.nearest_lane_center(value)], dtype=float)

    def distance_to_lane_boundary(self, value):
        output = []
        for point, item in zip(points_xy(value), self.nearest_lane_center(value)):
            polygon = self.lane_polygons.get(item.get("lane_id"))
            output.append(Point(point).distance(polygon.boundary) if polygon is not None else np.nan)
        return np.asarray(output, dtype=float)

    def distance_to_road_boundary(self, value):
        return np.asarray([Point(point).distance(self.drivable.boundary) for point in points_xy(value)], dtype=float)

    def lane_id(self, value):
        return [item.get("lane_id") if item.get("status") == "matched" else None for item in self.nearest_lane_center(value)]

    def road_segment_id(self, value):
        output = []
        for point in points_xy(value):
            shape = Point(point)
            output.append(next((token for polygon, token in zip(self.road_polygons, self.road_tokens) if polygon.covers(shape)), None))
        return output

    def route_progress(self, trajectory_xy, route_context=None):
        trajectory = points_xy(trajectory_xy)
        if len(trajectory) < 2:
            return {"progress_m": 0.0, "completion": 0.0, "source": "insufficient_points"}
        if route_context is not None and len(route_context) >= 2:
            from driveaudit_eval.agents.core import line_progress
            route = points_xy(route_context); start = line_progress(route, route[0], trajectory[0]); end = line_progress(route, route[0], trajectory[-1])
            total = max(line_progress(route, route[0], route[-1]), 1e-6)
            return {"progress_m": max(0.0, end - start), "completion": float(np.clip((end - start) / total, 0, 1)), "source": "route_context"}
        delta = np.diff(trajectory, axis=0); direction = self.lane_direction(trajectory[:-1])
        matched = np.isfinite(direction)
        signed = np.sum(delta[matched] * np.column_stack((np.cos(direction[matched]), np.sin(direction[matched]))), axis=1) if np.any(matched) else []
        progress = float(np.sum(signed)) if len(signed) else float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        return {"progress_m": progress, "completion": None, "source": "lane_direction" if len(signed) else "endpoint_displacement"}

    def footprint_coverage(self, centers, headings, length, width):
        return footprint_coverage(self.drivable, centers, headings, length, width)

    def query_metadata(self):
        return {**self.identity, "coordinate_system": self.coordinate_system, "cache_path": self.cache_path,
                "cache_hit": self.cache_hit, "layer_counts": self.layer_counts, "lane_sample_count": len(self.lane_points),
                "supported_locations": list(MAP_LOCATIONS)}

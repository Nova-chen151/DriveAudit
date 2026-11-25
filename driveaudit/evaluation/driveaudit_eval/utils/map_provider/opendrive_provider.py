from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from driveaudit_eval.utils.map_provider.base import MapProvider
from driveaudit_eval.utils.map_provider.geometry import footprint_coverage, points_xy


class OpenDriveMapProvider(MapProvider):
    provider_version = "opendrive_provider_v1"

    def __init__(self, map_data, source=None):
        self.map_data = map_data; self.drivable = map_data["drivable"]; self.source = str(source) if source else None

    @classmethod
    def from_file(cls, path, sample_step_m=0.5):
        from driveaudit_eval.trajectory.map_rule_metrics import parse_opendrive
        return cls(parse_opendrive(path, sample_step_m), source=path)

    def nearest_lane_center(self, value):
        from driveaudit_eval.trajectory.map_rule_metrics import _nearest_lane
        output = []
        for point in points_xy(value):
            nearest = _nearest_lane(point, self.map_data)
            if nearest is None:
                output.append({"status": "unmatched_lane"}); continue
            distance, lane, index = nearest
            output.append({"status": "matched", "distance": distance, "point": lane["center"][index].tolist(),
                           "heading": float(lane["heading"][index]), "lane_id": lane["key"], "layer": "driving_lane"})
        return output

    def is_drivable(self, value): return np.asarray([self.drivable.covers(Point(p)) for p in points_xy(value)])
    def signed_distance_to_drivable(self, value):
        return np.asarray([Point(p).distance(self.drivable.boundary) if self.drivable.covers(Point(p)) else -Point(p).distance(self.drivable) for p in points_xy(value)])
    def lane_direction(self, value): return np.asarray([x.get("heading", np.nan) for x in self.nearest_lane_center(value)])
    def distance_to_lane_center(self, value): return np.asarray([x.get("distance", np.nan) for x in self.nearest_lane_center(value)])
    def distance_to_lane_boundary(self, value): return np.asarray([np.nan] * len(points_xy(value)))
    def distance_to_road_boundary(self, value): return np.asarray([Point(p).distance(self.drivable.boundary) for p in points_xy(value)])
    def lane_id(self, value): return [x.get("lane_id") for x in self.nearest_lane_center(value)]
    def road_segment_id(self, value): return [None] * len(points_xy(value))
    def route_progress(self, trajectory_xy, route_context=None):
        pts = points_xy(trajectory_xy); return {"progress_m": float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0, "completion": None, "source": "endpoint_displacement"}
    def footprint_coverage(self, centers, headings, length, width): return footprint_coverage(self.drivable, centers, headings, length, width)
    def query_metadata(self): return {"provider": self.provider_version, "source": self.source, "coordinate_system": self.coordinate_system,
                                      "semantic_support": self.map_data.get("semantic_support", {})}

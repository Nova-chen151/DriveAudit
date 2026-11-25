from __future__ import annotations

from abc import ABC, abstractmethod


class MapProvider(ABC):
    """Coordinate-explicit map query contract shared by trajectory evaluators."""

    coordinate_system = "global_xy_meters"

    @abstractmethod
    def is_drivable(self, points_xy): ...

    @abstractmethod
    def signed_distance_to_drivable(self, points_xy): ...

    @abstractmethod
    def nearest_lane_center(self, points_xy): ...

    @abstractmethod
    def lane_direction(self, points_xy): ...

    @abstractmethod
    def distance_to_lane_center(self, points_xy): ...

    @abstractmethod
    def distance_to_lane_boundary(self, points_xy): ...

    @abstractmethod
    def distance_to_road_boundary(self, points_xy): ...

    @abstractmethod
    def lane_id(self, points_xy): ...

    @abstractmethod
    def road_segment_id(self, points_xy): ...

    @abstractmethod
    def route_progress(self, trajectory_xy, route_context=None): ...

    @abstractmethod
    def query_metadata(self): ...

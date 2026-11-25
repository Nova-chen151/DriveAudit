"""Optional dataset-specific extra-sensor loading, separate from canonical inputs."""
from .lidar import load_lidar_for_scene, read_points

__all__ = ['load_lidar_for_scene', 'read_points']

from .lidar_motion import CanonicalLidarFrame, compensate_lidar

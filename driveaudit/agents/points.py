"""Verified measured-point decoding for optional multimodal research."""
import numpy as np

def measured_lidar_xyz(payload, point_format='nuscenes_bin'):
    """Decode verified measured points, preserving the declared coordinate basis."""
    import io
    if point_format == 'xyz_npy':
        points = np.load(io.BytesIO(payload), allow_pickle=False)
    elif point_format == 'nuscenes_bin':
        raw = np.frombuffer(payload, dtype=np.float32)
        if raw.size % 5: raise ValueError('Invalid five-column LiDAR payload')
        points = raw.reshape(-1, 5)[:, :3]
    else: raise ValueError('Unsupported measured point format: '+point_format)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Invalid measured XYZ points')
    return points

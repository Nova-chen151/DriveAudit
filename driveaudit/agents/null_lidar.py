"""Fixed compatibility tensor; never imports or reads a sensor source."""

SENSOR_MODES = ('original_multimodal', 'zero_lidar', 'camera_only_native')


def validate_sensor_mode(mode):
    if mode not in SENSOR_MODES:
        raise ValueError('Unknown sensor mode: ' + str(mode))
    if mode == 'camera_only_native':
        raise ValueError('camera_only_native is not supported by the frozen checkpoint; use zero_lidar')
    return mode


def make_null_lidar_tensor(config):
    """Allocate the native histogram shape from config, without sensor data."""
    import torch
    channels = 2 if config.use_ground_plane else 1
    height = int((config.lidar_max_x-config.lidar_min_x) * int(config.pixels_per_meter))
    width = int((config.lidar_max_y-config.lidar_min_y) * int(config.pixels_per_meter))
    if config.lidar_seq_len != 1 or min(height,width) <= 0:
        raise ValueError('Unsupported native histogram configuration')
    return torch.zeros((channels,height,width), dtype=torch.float32)


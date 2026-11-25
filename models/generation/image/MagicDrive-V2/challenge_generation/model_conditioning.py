"""Normalize GT2 coordinate conventions to the nuScenes-trained checkpoint."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


def training_calibration(calibration: dict) -> dict:
    """Return model-only calibration, leaving published GT2 assets untouched.

    nuScenes already supplies native LIDAR_TOP coordinates, including its
    asynchronous camera transforms. Other GT2 datasets publish their ego frame
    as the renderer frame. Map/box/camera encoders are not rotation invariant,
    so use one nuScenes-style virtual LiDAR frame for those datasets. Consumers
    must derive boxes, maps and relative ego poses from the returned renderer.
    """
    output = deepcopy(calibration)
    renderer = output.get('renderer')
    if renderer is None and all('camera_to_ego_4x4' in cam for cam in output['cameras']):
        rig = np.array([[0,1,0,.985793],[-1,0,0,0],[0,0,1,1.84019],[0,0,0,1.]])
        output['renderer'] = dict(coordinate_frame='participant_virtual_lidar', sensor2ego_rotation_wxyz=[math.sqrt(.5),0,0,-math.sqrt(.5)], sensor2ego_translation_m=rig[:3,3].tolist())
        for camera in output['cameras']:
            camera['camera_to_renderer_4x4'] = (np.linalg.inv(rig) @ camera['camera_to_ego_4x4']).tolist()
            k = np.array(camera['challenge_intrinsics']).copy()
            k[0] *= 1600/400
            k[1] *= 900/224
            camera['native_intrinsics'] = k.tolist()
        return output
    if renderer is None or renderer.get('coordinate_frame') != 'history_last_ego_x_forward_y_left_meters':
        return output

    # These GT2 calibrations explicitly define the renderer as the ego frame.
    # Refuse an inconsistent declaration instead of silently moving a rig.
    quaternion = np.asarray(renderer['sensor2ego_rotation_wxyz'], dtype=np.float64)
    translation = np.asarray(renderer['sensor2ego_translation_m'], dtype=np.float64)
    if not (quaternion.shape == (4,) and translation.shape == (3,)
            and np.allclose(np.abs(quaternion[0]), 1.)
            and np.allclose(quaternion[1:], 0.) and np.allclose(translation, 0.)):
        raise ValueError('GT2 ego renderer must have identity sensor-to-ego calibration')

    renderer_to_ego = np.array([[0., 1., 0., .985793],
                                [-1., 0., 0., 0.],
                                [0., 0., 1., 1.84019],
                                [0., 0., 0., 1.]])
    ego_to_renderer = np.linalg.inv(renderer_to_ego)
    for camera in output['cameras']:
        old_transform = np.asarray(camera['camera_to_renderer_4x4'], dtype=np.float64)
        if old_transform.shape != (4, 4) or not np.isfinite(old_transform).all():
            raise ValueError(f"Invalid GT2 camera transform: {camera['camera']}")
        camera['camera_to_renderer_4x4'] = (ego_to_renderer @ old_transform).tolist()
        # GPU smoke tests support image-space K for these cross-dataset camera
        # embeddings. This compatibility policy is not a pixel-equivalent
        # reconstruction of native nuScenes training intrinsics. Keep both
        # published matrices intact; physical projection still uses challenge K.
        camera['model_intrinsics'] = deepcopy(camera['challenge_intrinsics'])
    renderer.update(
        coordinate_frame='magicdrivedit_virtual_lidar_top',
        sensor2ego_rotation_wxyz=[math.sqrt(.5), 0., 0., -math.sqrt(.5)],
        sensor2ego_translation_m=renderer_to_ego[:3, 3].tolist(),
    )
    return output

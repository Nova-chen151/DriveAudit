"""Build DiffusionDriveV2's measured XYZ input for multimodal research."""
import hashlib
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from driveaudit.agents.points import measured_lidar_xyz
from driveaudit.agents.lidar_motion import validate_compensated_entry


def validate_lidar_age(reference, timestamp):
    age = (reference - timestamp) / 1e6
    if not 0 <= age <= .1: raise ValueError('LiDAR must precede reference by at most 100 ms')
    return age


def transform_points(points, calibration, ego_pose, reference_pose):
    points = points @ Quaternion(calibration['rotation']).rotation_matrix.T + calibration['translation']
    points = points @ Quaternion(ego_pose['rotation']).rotation_matrix.T + ego_pose['translation']
    return (points - reference_pose['translation']) @ Quaternion(reference_pose['rotation']).rotation_matrix


def diffusion_points(model_input,index):
    state=model_input["ego_state"]
    key = model_input['camera_images']['CAM_FRONT']
    if key not in index['frames']:
        raise ValueError('NOT_AVAILABLE: no causal LIDAR_TOP for this frame')
    entry = index['frames'][key]
    if int(state['timestamp']) != entry['reference_timestamp']: raise ValueError('Sensor/reference timestamp mismatch')
    compensated = validate_compensated_entry(entry, int(state['timestamp']))
    age = ((entry['reference_timestamp']-entry['timestamp'])/1e6 if compensated
           else validate_lidar_age(entry['reference_timestamp'], entry['timestamp']))
    payload = Path(entry['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry['sha256']: raise ValueError('LiDAR content changed')
    xyz = transform_points(measured_lidar_xyz(payload,entry.get('point_format','nuscenes_bin')), entry['calibration'], entry['ego_pose'], entry['reference_pose'])
    if len(xyz) == 0 or not np.isfinite(xyz).all(): raise ValueError('Invalid point cloud')
    return xyz,entry,age

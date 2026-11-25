"""Build GoalFlow's measured LiDAR histogram for multimodal research."""
from types import SimpleNamespace
import hashlib
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from driveaudit.agents.points import measured_lidar_xyz
from driveaudit.agents.lidar_motion import validate_compensated_entry


def transform_lidar(points, entry):
    for name in ('lidar_calibration', 'lidar_ego_pose'):
        pose = entry[name]
        points = points @ Quaternion(pose['rotation']).rotation_matrix.T + np.asarray(pose['translation'])
    pose = entry['camera_ego_pose']
    return (points - np.asarray(pose['translation'])) @ Quaternion(pose['rotation']).rotation_matrix


def goalflow_feature(model_input,index,builder):
    key = model_input['camera_images']['CAM_FRONT']
    entry = index[key]
    timestamp = model_input['ego_state']['timestamp']
    compensated = validate_compensated_entry(entry, timestamp)
    if (entry['reference_timestamp'] != timestamp or
        (not compensated and not 0 <= timestamp-entry['lidar_timestamp'] <= 100000) or
        entry['camera_timestamp_delta_us'] != 0):
        raise ValueError('LiDAR must precede this exact camera frame by at most 100ms')
    payload=Path(entry['lidar_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=entry['sha256']: raise ValueError('LiDAR content changed')
    points=transform_lidar(measured_lidar_xyz(payload,entry.get('point_format','nuscenes_bin')),entry).T
    lidar = builder._get_lidar_feature(SimpleNamespace(lidars=[SimpleNamespace(lidar_pc=points)]))
    return lidar,entry

"""Measured native poses and point clouds; never download or synthesize inputs."""
import json
from pathlib import Path

import numpy as np

from driveaudit_eval.agents.core import quat_rotation
from driveaudit_eval.consistency.input_gt_full import _quat


def _matrix(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = quat_rotation(rotation)
    pose[:3, 3] = translation
    return pose


def _record(path, timestamp, pose, fmt):
    return dict(path=str(path), timestamp=int(timestamp), token=path.stem,
                format=fmt, lidar_calibration=dict(rotation=[1, 0, 0, 0], translation=[0, 0, 0]),
                lidar_ego_pose=dict(rotation=_quat(pose), translation=pose[:3, 3].tolist()))


def load_source(scene, source_root=None):
    import pandas as pd
    scene = Path(scene)
    provenance_path = scene / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    root = Path(source_root or provenance['source_root'])
    sources = [provenance_path]
    lidar = []
    if provenance['dataset'] == 'av2':
        path = root / 'city_SE3_egovehicle.feather'
        sources.append(path)
        table = pd.read_feather(path).sort_values('timestamp_ns')
        native_times = table.timestamp_ns.to_numpy(dtype=np.int64)
        times = native_times // 1000
        poses = np.array([_matrix(row[:4], row[4:]) for row in
                          table[['qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m']].to_numpy()])
        lookup = dict(zip(native_times.tolist(), poses))
        for path in sorted((root / 'sensors/lidar').glob('*.feather')):
            timestamp = int(path.stem)
            if timestamp not in lookup:
                raise ValueError(f'AV2 sweep has no exact native ego pose: {path}')
            lidar.append(_record(path, timestamp // 1000, lookup[timestamp], 'av2_ego_feather'))
    elif provenance['dataset'] == 'pandaset':
        paths = [root / 'lidar/timestamps.json', root / 'lidar/poses.json']
        sources.extend(paths)
        times = np.rint(np.asarray(json.loads(paths[0].read_text())) * 1e6).astype(np.int64)
        rows = json.loads(paths[1].read_text())
        # PandaSet native LiDAR axes: x right, y forward; published ego is x forward, y left.
        basis = np.eye(4)
        basis[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        poses = np.array([_matrix([r['heading'][k] for k in 'wxyz'],
                                 [r['position'][k] for k in 'xyz']) @ basis for r in rows])
        for path in sorted((root / 'lidar').glob('*.pkl.gz')):
            index = int(path.name.split('.')[0])
            if not 0 <= index < len(times):
                raise ValueError(f'Invalid PandaSet sweep index: {path}')
            lidar.append(_record(path, times[index], poses[index], 'pandaset_world_pickle'))
    else:
        raise ValueError(f"Unsupported native dataset: {provenance['dataset']}")
    if provenance['dataset'] == 'av2':
        keep = np.r_[np.diff(times) != 0, True]
        times, poses = times[keep], poses[keep]
    if len(times) != len(poses) or not np.all(np.diff(times) > 0) or not np.isfinite(poses).all():
        raise ValueError('Invalid native pose timeline')
    return dict(times=times, poses=poses, lidar=lidar, source_paths=[str(p) for p in sources])


def read_points(record):
    import pandas as pd
    if record['format'] == 'av2_ego_feather':
        points = pd.read_feather(record['path'])[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    elif record['format'] == 'pandaset_world_pickle':
        # Native trusted dataset pickle; coordinates are already motion-compensated world points.
        points = pd.read_pickle(record['path'])[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        pose = record['lidar_ego_pose']
        points = ((points - pose['translation']) @ quat_rotation(pose['rotation'])).astype(np.float32)
    else:
        raise ValueError(f"Unsupported measured point format: {record['format']}")
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Empty/nonfinite measured cloud: {record['path']}")
    return points

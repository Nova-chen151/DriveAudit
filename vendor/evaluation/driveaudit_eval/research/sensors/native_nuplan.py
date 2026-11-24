"""Read native nuPlan sensor metadata without requiring the nuPlan SDK."""
import io
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class _CalibrationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'nuplan.database.common.data_types' and name in ('Translation', 'Rotation', 'CameraIntrinsic'):
            return list
        if (module, name) in {('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Unexpected calibration type: {module}.{name}')


def load_source(scene, source_root=None):
    """Return absolute integer microseconds, native poses, and LiDAR records.

    Records can refer to absent payloads; callers must check their paths and
    select scans causally. No scene future annotations are read.
    """
    provenance_path = Path(scene) / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if provenance['dataset'] != 'nuplan' or provenance['source_time_unit'] != 'microseconds':
        raise ValueError('Expected native nuPlan microsecond provenance')
    root = Path(source_root or provenance['source_root'])
    db = root / 'log.db'
    if not db.is_file():
        raise FileNotFoundError(db)
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT timestamp,x,y,z,qw,qx,qy,qz FROM ego_pose ORDER BY timestamp').fetchall()
        records = connection.execute('''SELECT p.filename,p.timestamp,hex(p.token),
            l.rotation,l.translation,e.qw,e.qx,e.qy,e.qz,e.x,e.y,e.z
            FROM lidar_pc p JOIN lidar l ON l.token=p.lidar_token
            JOIN ego_pose e ON e.token=p.ego_pose_token ORDER BY p.timestamp''').fetchall()
    times = np.asarray([row[0] for row in rows], dtype=np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Native ego timestamps must be nonempty and strictly increasing')
    values = np.asarray([row[1:] for row in rows], dtype=float)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(values[:, [4, 5, 6, 3]]).as_matrix()
    poses[:, :3, 3] = values[:, :3]
    lidar = []
    for filename, timestamp, token, rotation, translation, *pose in records:
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid native LiDAR filename')
        paths = [root / relative, root.parent / relative, root / 'lidar' / relative.name]
        path = next((candidate for candidate in paths if candidate.is_file()), paths[1])
        lidar.append(dict(path=path, timestamp=int(timestamp), token=token.lower(), format='nuplan_pcd',
            lidar_calibration=dict(rotation=_CalibrationUnpickler(io.BytesIO(rotation)).load(),
                                   translation=_CalibrationUnpickler(io.BytesIO(translation)).load()),
            lidar_ego_pose=dict(rotation=pose[:4], translation=pose[4:])))
    return dict(times=times, poses=poses, lidar=lidar, source_paths=[provenance_path, db])


if __name__ == '__main__':
    import sys
    source = load_source(Path(sys.argv[1]))
    assert source['poses'].shape == (len(source['times']), 4, 4)
    assert np.allclose(np.linalg.det(source['poses'][:, :3, :3]), 1)
    assert all(row['timestamp'] <= source['lidar'][i + 1]['timestamp']
               for i, row in enumerate(source['lidar'][:-1]))
    print(len(source['times']), len(source['lidar']), 'native metadata rows verified')

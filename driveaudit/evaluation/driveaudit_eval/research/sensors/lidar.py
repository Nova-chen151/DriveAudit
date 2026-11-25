"""Optional measured LiDAR adapters. Canonical scene inputs never depend on these."""
import bisect
import json
from pathlib import Path

import numpy as np


def _provenance(scene):
    path = scene / 'provenance.json'
    if not path.is_file():
        path = scene / 'generation_provenance.json'
    return json.loads(path.read_text())


def _nuscenes(scene, config):
    from driveaudit_eval.consistency.input_gt_full import _native_records
    root = Path(config['dataroot'])
    table = root / config.get('version', 'v1.0-trainval')
    provenance = _provenance(scene)
    name = provenance.get('source_scene')
    if name is None:
        name = json.loads((scene / 'future/rgb_manifest.json').read_text())['source_scene']
    paths = [table / f'{name}.json' for name in ['scene', 'sample', 'sample_data', 'ego_pose', 'calibrated_sensor']]
    scene_token = next(r['token'] for r in json.loads(paths[0].read_text()) if r['name'] == name)
    samples = {r['token'] for r in json.loads(paths[1].read_text()) if r['scene_token'] == scene_token}
    selected = [r for r in _native_records(paths[2]) if r['sample_token'] in samples]
    scans = [r for r in selected if '/LIDAR_TOP/' in r['filename']]
    time_base = None
    if config.get('timestamps_relative'):
        frames = json.loads((scene / 'timeline.json').read_text())['frames']
        rgb = json.loads((scene / 'future/rgb_manifest.json').read_text())
        offsets = [(i, c, r['exposure_time_us']) for i, frame in enumerate(frames) for c, r in frame.get('cameras', {}).items()]
        offsets.extend((r['frame_index'], r['camera'], r['exposure_time_us']) for r in rgb['images'])
        by_camera = {}
        for row in selected:
            if '/CAM_' in row['filename']:
                by_camera.setdefault(row['filename'].split('/')[1], set()).add(row['timestamp'])
        first = next(offset for i, c, offset in offsets if i == 0 and c == 'CAM_FRONT')
        candidates = [stamp-first for stamp in by_camera['CAM_FRONT']]
        candidates = [base for base in candidates if all(base+offset in by_camera[c] for _, c, offset in offsets)]
        if len(candidates) != 1:
            raise ValueError('Native camera exposures do not uniquely align LiDAR time base')
        time_base = candidates[0]
    tokens = {r['ego_pose_token'] for r in scans}
    poses = {r['token']: r for r in _native_records(paths[3]) if r['token'] in tokens}
    calibrations = {r['token']: r for r in json.loads(paths[4].read_text())}
    records = [dict(path=str(root / r['filename']), timestamp=r['timestamp'], token=r['token'],
                    format='nuscenes_bin', lidar_calibration=calibrations[r['calibrated_sensor_token']],
                    lidar_ego_pose=poses[r['ego_pose_token']]) for r in scans]
    return dict(lidar=records, source_paths=[str(p) for p in paths], absolute_time_base_us=time_base)


def _availability(record):
    """A scan is available only after its latest measured return."""
    if record['format'] == 'av2_ego_feather':
        import pandas as pd
        offsets = pd.read_feather(record['path'], columns=['offset_ns']).offset_ns
        return max(record['timestamp'], (int(Path(record['path']).stem) + int(offsets.max()) + 999) // 1000)
    if record['format'] == 'pandaset_world_pickle':
        import pandas as pd
        returns = pd.read_pickle(record['path']).t
        return max(record['timestamp'], int(np.ceil(returns.max() * 1e6)))
    return record['timestamp']


def causal_point_mask(record):
    """Mask measured returns by their native acquisition timestamp, never XYZ."""
    import pandas as pd
    cutoff = int(record['point_time_cutoff_absolute_us'])
    if record['format'] == 'av2_ego_feather':
        data = pd.read_feather(record['path'], columns=['offset_ns'])
        acquisition = int(Path(record['path']).stem) + data.offset_ns.to_numpy(dtype=np.int64)
        return acquisition <= cutoff*1000, (acquisition+999)//1000
    if record['format'] == 'pandaset_world_pickle':
        acquisition = np.ceil(pd.read_pickle(record['path']).t.to_numpy()*1e6).astype(np.int64)
        return acquisition <= cutoff, acquisition
    raise ValueError('Per-point acquisition timing is unavailable for this source format')


def _partial_record(record, reference):
    base = record['absolute_timestamp']-record['timestamp']
    candidate = dict(record,point_time_cutoff_absolute_us=reference+base)
    mask, acquisition = causal_point_mask(candidate)
    if not mask.any():
        return None
    completion = max(record['timestamp'],int(acquisition[mask].max())-base)
    candidate.update(availability_timestamp=completion,absolute_availability_timestamp=completion+base,
        full_sweep_availability_timestamp=record['availability_timestamp'],partial_sweep=True,
        original_point_count=len(mask),retained_point_count=int(mask.sum()),retained_point_fraction=float(mask.mean()),
        point_mask_policy='native_acquisition_timestamp_lte_decision; no imputed returns')
    return candidate


def load_lidar_for_scene(scene_dir, reference_timestamps, source_config=None):
    """Return aligned records (None when absent) and explicit missing-data reasons.

    Timestamps are absolute integer microseconds. Caller supplies canonical
    reference poses independently. This function never fetches source data.
    """
    scene = Path(scene_dir)
    config = source_config or {}
    references = [int(t) for t in reference_timestamps]
    protocol = config.get('alignment_protocol', 'legacy')
    if protocol not in ('legacy', 'motion_compensated_v1'):
        raise ValueError('Unsupported LiDAR alignment protocol')
    provenance = _provenance(scene)
    dataset = provenance.get('dataset', 'nuscenes')
    try:
        if dataset == 'nuscenes':
            if not config.get('dataroot'):
                raise FileNotFoundError('nuScenes LiDAR requires source_config.dataroot')
            source = _nuscenes(scene, config)
        elif dataset in ('av2', 'pandaset'):
            from .native_av2_pandaset import load_source
            source = load_source(scene, source_root=config.get('source_root'))
        elif dataset == 'nuplan':
            from .native_nuplan import load_source
            source = load_source(scene, source_root=config.get('source_root'))
        else:
            raise ValueError(f'Unsupported LiDAR dataset: {dataset}')
    except FileNotFoundError as exc:
        return dict(status='NOT_AVAILABLE', records=[None] * len(references), source_paths=[],
                    missing=[dict(reference_timestamp=t, reason='required_lidar_not_available', detail=str(exc)) for t in references])
    base = 0
    if config.get('timestamps_relative'):
        if dataset == 'nuscenes':
            base = source['absolute_time_base_us']
        else:
            scale = {'nanoseconds': 0.001, 'microseconds': 1, 'seconds': 1e6}[provenance['source_time_unit']]
            origin = provenance['source_time_origin']
            base = (int(origin) // 1000 if provenance['source_time_unit'] == 'nanoseconds' else round(origin * scale))
            base += round(provenance.get('clip_start_seconds_from_origin', 0) * 1e6)
    anchor = provenance.get('anchor_ego_to_world') if config.get('timestamps_relative') else None
    cadence = None
    if protocol == 'motion_compensated_v1':
        from .lidar_motion import native_cadence_evidence
        cadence = native_cadence_evidence(dataset, source)
    scans = []
    for row in source['lidar']:
        record = dict(row, path=str(row['path']))
        if not Path(record['path']).is_file():
            continue
        record['availability_timestamp'] = _availability(record)
        record['absolute_timestamp'] = record['timestamp']
        record['absolute_availability_timestamp'] = record['availability_timestamp']
        record['timestamp'] -= base
        record['availability_timestamp'] -= base
        if anchor is not None:
            from .native_av2_pandaset import _matrix
            from scipy.spatial.transform import Rotation
            native = record['lidar_ego_pose']
            relative = np.linalg.inv(np.asarray(anchor)) @ _matrix(native['rotation'], native['translation'])
            record['native_ego_pose'] = native
            q = Rotation.from_matrix(relative[:3, :3]).as_quat()
            record['lidar_ego_pose'] = dict(rotation=q[[3,0,1,2]].tolist(), translation=relative[:3,3].tolist())
        if protocol == 'motion_compensated_v1':
            from .lidar_motion import NATIVE_SCAN_PERIOD_US
            record.update(alignment_protocol=protocol,native_scan_period_us=NATIVE_SCAN_PERIOD_US[dataset],
                native_timestamp_jitter_us=cadence['measured_max_jitter_us'],cadence_evidence=cadence)
        record['calibration'] = record['lidar_calibration']
        record['ego_pose'] = record['lidar_ego_pose']
        scans.append(record)
    scans.sort(key=lambda r: (r['availability_timestamp'], r['timestamp']))
    available = [r['availability_timestamp'] for r in scans]
    records, missing = [], []
    for reference in references:
        index = bisect.bisect_right(available, reference) - 1
        record = scans[index] if index >= 0 else None
        age_valid = record is not None and 0 <= reference - record['timestamp'] <= config.get('max_age_us', 100000)
        if record is not None and protocol == 'motion_compensated_v1':
            from .lidar_motion import source_age_valid
            age_valid = source_age_valid(record, reference)
            if not age_valid:
                pending = sorted((r for r in scans if r['timestamp'] <= reference < r['availability_timestamp']
                    and r['format'] in ('av2_ego_feather','pandaset_world_pickle')),key=lambda r:r['timestamp'],reverse=True)
                for candidate in pending:
                    partial = _partial_record(candidate, reference)
                    if partial is not None and source_age_valid(partial, reference):
                        record, age_valid = partial, True
                        break
            native_times = source.get('times', np.asarray([r['timestamp'] for r in source['lidar']], dtype=np.int64))
            native_times = np.sort(native_times)
            query = reference+base
            ix = int(np.searchsorted(native_times, query))
            pose_support = bool(len(native_times) and native_times[0] <= query <= native_times[-1] and
                (ix < len(native_times) and native_times[ix] == query or
                 0 < ix < len(native_times) and native_times[ix]-native_times[ix-1] <= 250000))
            record = dict(record,pose_support_valid=pose_support)
            age_valid = age_valid and pose_support
        if not age_valid:
            records.append(None)
            missing.append(dict(reference_timestamp=reference,
                reason='invalid_sensor_alignment' if scans else 'required_lidar_not_available',
                detail='No measured causal scan within max_age_us',
                nearest_causal_age_us=None if record is None else reference-record['timestamp'],
                max_observed_return_offset_us=max((s['availability_timestamp']-s['timestamp'] for s in scans), default=None)))
        else:
            records.append(dict(record, reference_timestamp=reference, age_us=reference-record['timestamp']))
    return dict(status='AVAILABLE' if not missing else 'NOT_AVAILABLE', records=records,
                source_paths=[str(p) for p in source['source_paths']], missing=missing, absolute_time_base_us=base,
                max_observed_return_offset_us=max((s['availability_timestamp']-s['timestamp'] for s in scans), default=None),
                cadence_evidence=cadence)


def read_points(record):
    """Read actual XYZ in the sensor basis described by record['calibration']."""
    fmt = record['format']
    if fmt in ('av2_ego_feather', 'pandaset_world_pickle'):
        from .native_av2_pandaset import read_points as read_native
        native_record = dict(record)
        if 'native_ego_pose' in record:
            native_record['lidar_ego_pose'] = record['native_ego_pose']
        points = read_native(native_record)
        if 'point_time_cutoff_absolute_us' in record:
            mask, _ = causal_point_mask(record)
            if len(mask) != len(points):
                raise ValueError('Causal mask/native point count mismatch')
            points = points[mask]
        return points
    if fmt == 'nuscenes_bin':
        points = np.fromfile(record['path'], dtype=np.float32)
        if points.size % 5:
            raise ValueError('Invalid nuScenes five-column LiDAR payload')
        points = points.reshape(-1, 5)[:, :3]
    elif fmt == 'nuplan_pcd':
        points = _read_pcd(record['path'])
    else:
        raise ValueError(f'Unsupported LiDAR format: {fmt}')
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Empty/nonfinite measured LiDAR')
    return points


def _read_pcd(path):
    """Read native binary or ASCII PCD XYZ; reject unsupported compression."""
    with Path(path).open('rb') as handle:
        header = {}
        for _ in range(64):
            parts = handle.readline().decode('ascii').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0]] = parts[1:]
            if parts[0] == 'DATA':
                break
        else:
            raise ValueError('Missing bounded PCD header')
        fields = header['FIELDS']
        if any(int(c) != 1 for c in header.get('COUNT', ['1'] * len(fields))):
            raise ValueError('Unsupported vector PCD field')
        count = int(header['POINTS'][0])
        mode = header['DATA'][0]
        if mode == 'binary':
            types = {'F': 'f', 'U': 'u', 'I': 'i'}
            dtype = np.dtype([(name, '<'+types[kind]+size) for name, kind, size in zip(fields, header['TYPE'], header['SIZE'])])
            payload = handle.read()
            if len(payload) != count*dtype.itemsize:
                raise ValueError('Truncated/oversized binary PCD')
            data = np.frombuffer(payload, dtype=dtype)
            return np.column_stack([data[k] for k in 'xyz']).astype(np.float32)
        if mode == 'ascii':
            data = np.loadtxt(handle, ndmin=2)
            if len(data) != count or data.shape[1] != len(fields):
                raise ValueError('Invalid ASCII PCD shape')
            return data[:, [fields.index(k) for k in 'xyz']].astype(np.float32)
        raise ValueError(f'Unsupported PCD encoding: {mode}')

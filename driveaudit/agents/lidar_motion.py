"""Measured sweep-to-decision compensation; no dynamic-object motion model."""
from dataclasses import dataclass

import numpy as np

from driveaudit.agents.runtime import quat_rotation

PROTOCOL = 'motion_compensated_v1'
# Native sensor rates, not model planning rate. AV2 official 10 Hz; nuPlan/nuScenes 20 Hz.
# Frozen DriveAudit freshness budget; a policy, not an official model threshold.
COMPLETION_FRESHNESS_BUDGET_US = 100000
NATIVE_SCAN_PERIOD_US = {'av2': 100000, 'pandaset': 100000, 'nuplan': 50000, 'nuscenes': 50000}


def native_cadence_evidence(dataset, source):
    """Calibrate clock/sampling jitter from the full native metadata timeline."""
    period = NATIVE_SCAN_PERIOD_US[dataset]
    times = (source['times'] if dataset == 'pandaset' else
             np.asarray([r['timestamp'] for r in source['lidar']], dtype=np.int64))
    gaps = np.diff(np.sort(np.unique(times)))
    one_period = gaps[np.rint(gaps/period) == 1]
    full_metadata = dataset != 'av2'
    jitter = max(0,int(one_period.max())-period) if len(one_period) and full_metadata else 0
    return dict(nominal_period_us=period,measured_max_jitter_us=jitter,
        interval_count=len(gaps),one_period_interval_count=len(one_period),
        excluded_multi_period_or_invalid_count=int(len(gaps)-len(one_period)),
        interval_min_us=int(gaps.min()) if len(gaps) else None,
        interval_max_us=int(gaps.max()) if len(gaps) else None,
        interval_median_us=float(np.median(gaps)) if len(gaps) else None,
        interval_p01_us=float(np.percentile(gaps,1)) if len(gaps) else None,
        interval_p99_us=float(np.percentile(gaps,99)) if len(gaps) else None,
        interval_mean_us=float(np.mean(gaps)) if len(gaps) else None,
        metadata_scope='full_native_sensor_timeline' if full_metadata else 'local_AV2_scan_listing; no jitter allowance calibrated',
        method='maximum positive deviation among nearest-integer-one-period intervals; multi-period drops excluded')


def source_age_valid(record, target_timestamp):
    """Preserve the frozen 100 ms budget, measured from actual availability."""
    period = int(record['native_scan_period_us'])
    source = int(record['timestamp'])
    completion = int(record['availability_timestamp'])
    target = int(target_timestamp)
    duration = completion-source
    if period <= 0 or duration < 0:
        raise ValueError('Invalid native LiDAR timing evidence')
    return 0 <= target-completion <= COMPLETION_FRESHNESS_BUDGET_US and 0 <= target-source <= duration+COMPLETION_FRESHNESS_BUDGET_US


def _pose(value):
    result = np.eye(4)
    result[:3, :3] = quat_rotation(value['rotation'])
    result[:3, 3] = value['translation']
    if not np.isfinite(result).all() or not np.allclose(result[:3,:3].T @ result[:3,:3], np.eye(3), atol=1e-6):
        raise ValueError('Invalid measured rigid pose')
    return result


@dataclass
class CanonicalLidarFrame:
    points_xyz: np.ndarray
    intensity: object
    source_timestamp: int
    target_timestamp: int
    temporal_offset_s: float
    motion_compensated: bool
    calibration: dict
    valid: bool
    metadata: dict


def compensate_lidar(record, target_timestamp, target_ego_pose):
    """Transform source sensor XYZ -> source ego -> shared world -> decision ego.

    AV2 XYZ already includes the publisher's per-point ego-motion deskew. Applying
    point-time poses again would deskew twice; only the residual rigid transform
    from the supplied sweep reference pose to the decision pose is applied here.
    """
    from .lidar import read_points
    if record.get('alignment_protocol') != PROTOCOL or not source_age_valid(record, target_timestamp):
        raise ValueError('Invalid motion-compensated LiDAR age/protocol')
    if record.get('pose_support_valid') is not True:
        raise ValueError('Measured source/target pose support has not been verified')
    xyz = read_points(record).astype(np.float64)
    source_pose = _pose(record['ego_pose'])
    calibration = _pose(record['calibration'])
    target_pose = _pose(target_ego_pose)
    transform = np.linalg.inv(target_pose) @ source_pose @ calibration
    points = (xyz @ transform[:3,:3].T + transform[:3,3]).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite compensated measured points')
    intensity = None
    if record['format'] == 'av2_ego_feather':
        import pandas as pd
        intensity = pd.read_feather(record['path'], columns=['intensity']).intensity.to_numpy()
    elif record['format'] == 'pandaset_world_pickle':
        import pandas as pd
        intensity = pd.read_pickle(record['path']).i.to_numpy()
    elif record['format'] == 'nuscenes_bin':
        intensity = np.fromfile(record['path'], dtype=np.float32).reshape(-1,5)[:,3]
    if intensity is not None and 'point_time_cutoff_absolute_us' in record:
        from .lidar import causal_point_mask
        mask, _ = causal_point_mask(record)
        intensity = intensity[mask]
    if intensity is not None and (len(intensity) != len(points) or not np.isfinite(intensity).all()):
        raise ValueError('Invalid native intensity')
    raw_offset = int(target_timestamp)-int(record['timestamp'])
    metadata = dict(alignment_protocol=PROTOCOL,alignment_status='compensated_valid',
        compensation_type='sweep_rigid',
        native_deskew='publisher_per_point_ego_compensated' if record['format']=='av2_ego_feather' else 'native_coordinates_preserved',
        dynamic_object_compensation=False,source_timestamp=int(record['timestamp']),target_timestamp=int(target_timestamp),
        availability_timestamp=int(record['availability_timestamp']),raw_lidar_offset_ms=raw_offset/1000.,
        source_age_ms=raw_offset/1000.,completion_age_ms=(int(target_timestamp)-record['availability_timestamp'])/1000.,
        compensated_lidar_offset_ms=0.,compensated_offset_meaning='coordinate_reference_time_only; point measurements retain source age',
        max_source_age_ms=(record['availability_timestamp']-record['timestamp']+COMPLETION_FRESHNESS_BUDGET_US)/1000.,
        completion_freshness_budget_us=COMPLETION_FRESHNESS_BUDGET_US,
        freshness_policy='inherited_DriveAudit_100ms_budget_applied_to_measured_availability',
        native_scan_period_us=record['native_scan_period_us'],
        native_timestamp_jitter_us=record.get('native_timestamp_jitter_us',0),
        cadence_evidence=record.get('cadence_evidence'),point_count=len(points),pose_support_valid=True,
        source_to_target_4x4=transform.tolist(),source_pose=record['ego_pose'],target_pose=target_ego_pose,
        partial_sweep=record.get('partial_sweep',False),
        full_sweep_availability_timestamp=record.get('full_sweep_availability_timestamp',record['availability_timestamp']),
        original_point_count=record.get('original_point_count',len(points)),retained_point_count=len(points),
        retained_point_fraction=record.get('retained_point_fraction',1.),
        point_mask_policy=record.get('point_mask_policy','all acquired returns in completed sweep'),
        point_time_cutoff_absolute_us=record.get('point_time_cutoff_absolute_us'))
    return CanonicalLidarFrame(points,intensity,int(record['timestamp']),int(target_timestamp),raw_offset/1e6,True,
        dict(rotation=[1,0,0,0],translation=[0,0,0]),True,metadata)


def validate_compensated_entry(entry, target_timestamp):
    """Service-side validation of the canonical sidecar, retaining raw age checks."""
    evidence = entry.get('lidar_alignment', {})
    if evidence.get('alignment_protocol') != PROTOCOL:
        return False
    source = int(evidence['source_timestamp'])
    completion = int(evidence['availability_timestamp'])
    period = int(evidence['native_scan_period_us'])
    if (entry.get('point_format') != 'xyz_npy' or evidence.get('alignment_status') != 'compensated_valid' or evidence.get('pose_support_valid') is not True or
        evidence.get('dynamic_object_compensation') is not False or evidence.get('target_timestamp') != target_timestamp or
        evidence.get('compensation_type') != 'sweep_rigid' or evidence.get('compensated_lidar_offset_ms') != 0 or
        not source_age_valid(dict(timestamp=source,availability_timestamp=completion,native_scan_period_us=period,
            native_timestamp_jitter_us=evidence.get('native_timestamp_jitter_us',0)),target_timestamp) or
        not np.isclose(evidence['raw_lidar_offset_ms'], (target_timestamp-source)/1000., atol=1e-9)):
        raise ValueError('Invalid canonical LiDAR compensation evidence')
    return True

"""Validate the measured sweep evidence accepted by fusion agents."""
import numpy as np

PROTOCOL = 'motion_compensated_v1'
COMPLETION_FRESHNESS_BUDGET_US = 100000


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

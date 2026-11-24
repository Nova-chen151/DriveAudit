"""Metric inputs are distinct from agent sensor requirements."""

REQUIREMENTS = {
    'FVD': ('video', 'reference_video'),
    'CMMD': ('video', 'reference_video'),
    'CLIP_IQA': ('video',),
    'DINO_T': ('video', 'timeline'),
    'Epi_3px': ('camera', 'calibration'),
    'TVC': ('trajectory', 'camera', 'calibration'),
    'TRS': ('trajectory',),
    'TRS_road': ('trajectory', 'drivable_area'),
    'TRS_direction': ('trajectory', 'directed_lane_centerlines'),
    'DAC': ('agent_plan', 'drivable_area'),
    'PDMS': ('agent_plan', 'drivable_area', 'TTC', 'Comfort', 'EP'),
    'RPDMS': ('agent_plan', 'drivable_area', 'TTC', 'Comfort', 'EP'),
    'RouteCompletion': ('agent_plan', 'reference_route_proxy'),
}
MAP_REASON = 'insufficient_drivable_area_map_support'


def map_capabilities(scene):
    """Inspect the exported map's declared support, including unknown surfaces."""
    import json
    from pathlib import Path
    import numpy as np
    folder = Path(scene) / 'map'
    try:
        meta = json.loads((folder / 'map_meta.json').read_text())
        with np.load(folder / 'semantic_map.npz', allow_pickle=False) as data:
            full = ('drivable_area' in meta.get('available_channels', [])
                    and 'observed_surface' not in data
                    and meta.get('status') != 'RECONSTRUCTED_PARTIAL_OBSERVATIONS'
                    and not meta.get('generation_conditions', {}).get('requires_partial_map_support', False))
        return dict(drivable_area=full, directed_lane_centerlines=False)
    except (FileNotFoundError, ValueError, KeyError):
        return dict(drivable_area=False, directed_lane_centerlines=False)


def metric_requirements(capabilities, collision=False):
    """Only explicit True establishes availability; missing inputs stay unknown."""
    result = {}
    for metric, required in REQUIREMENTS.items():
        required = required + (('responsibility',) if metric == 'RPDMS' and collision else ())
        missing = [key for key in required if capabilities.get(key) is not True]
        reason = MAP_REASON if 'drivable_area' in missing else 'required_' + missing[0] + '_not_available' if missing else None
        result[metric] = dict(status='unavailable' if missing else 'available',
                              requirements=list(required), missing=missing, reason=reason)
    return result


def apply_metric_requirements(row, capabilities):
    """Mask unsupported outputs without substituting or reweighting multipliers."""
    result = dict(row)
    gates = metric_requirements(capabilities, collision=row.get('NC') == 0)
    fields = {'DAC': ('DAC', 'offroad_rate'), 'PDMS': ('PDMS_open_loop', 'PDMS_TC_open_loop', 'Driving_Score_open_loop'),
              'RPDMS': ('rpdms', 'RPDMS'), 'RouteCompletion': ('route_completion_ratio',)}
    for metric, names in fields.items():
        if gates[metric]['status'] == 'unavailable':
            for name in names:
                if name in result:
                    result[name] = None
            result[metric + '_reason'] = gates[metric]['reason']
    result['metric_capabilities'] = gates
    return result

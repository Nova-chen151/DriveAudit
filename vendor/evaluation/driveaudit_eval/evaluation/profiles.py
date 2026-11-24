"""Execution policy; unchanged numeric kernels remain available in legacy mode."""
from copy import deepcopy

from driveaudit_eval.agents.registry import FORMAL_AGENTS
CONTROL_KEYS = {'profile', 'agents', 'core_metrics', 'diagnostics', 'legacy_export', 'leaderboard', 'reuse_results', 'testing_effectiveness'}
from driveaudit_eval.metrics.registry import DEPRECATED_ALIASES


def prepare_config(config):
    cfg = deepcopy(config)
    settings = cfg.setdefault('evaluation', {})
    diagnostic_config = cfg.get('diagnostic_config')
    if diagnostic_config:
        from pathlib import Path
        import yaml
        path = Path(diagnostic_config)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        defaults = yaml.safe_load(path.read_text())
        for section, values in defaults.items():
            # Overrides select runtime inputs; large backend defaults live separately.
            merged = deepcopy(values)
            merged.update(settings.get(section, {}))
            settings[section] = merged
    profile = settings.get('profile', 'legacy')
    if profile not in ('leaderboard', 'research', 'legacy'):
        raise ValueError('evaluation.profile must be leaderboard or research')
    if profile != 'legacy':
        from driveaudit_eval.metrics.registry import CORE_IDS, OFFICIAL_IDS
        if not settings.get('core_metrics', {}).get('enabled', True):
            raise ValueError('Core metrics are required in core/research profiles; use legacy for original-only evaluation')
        board = settings.get('leaderboard', {})
        if tuple(board.get('metrics', OFFICIAL_IDS)) not in (CORE_IDS, OFFICIAL_IDS) or not board.get('group_balanced_rank', True):
            raise ValueError('Formal metrics require the registry order and supplementary group-balanced ranks')
        settings.setdefault('leaderboard', {}).update(metrics=list(OFFICIAL_IDS), final_ranking_metric='qtes')
    settings.setdefault('diagnostics', {})['enabled'] = profile == 'research'
    retention = cfg.get('output', {}).get('retention', 'full')
    if retention not in ('full', 'metrics'):
        raise ValueError('output.retention must be full or metrics')
    if retention == 'metrics':
        for section in ('video', 'trajectory', 'consistency', 'agent'):
            settings.setdefault(section, {}).update(retain_intermediates=False, resume=False)
        settings['video'].setdefault('output', {}).update(retention='metrics', save_cache=False,
            save_visualizations=False, save_curves=False)
        testing = settings.setdefault('testing_effectiveness', {})
        testing['retain_intermediates'] = False
        testing.setdefault('quality_retention', {})['retain_intermediates'] = False
    if profile == 'legacy':
        settings.setdefault('legacy_export', {})['enabled'] = True
        settings.setdefault('core_metrics', {})['enabled'] = False
    included = settings.get('agents', {}).get('include', cfg.get('agents', {}).get('include'))
    if included is not None:
        if not isinstance(included, list) or not included or len(set(included)) != len(included):
            raise ValueError('agents.include must be a nonempty unique list')
        if profile != 'legacy' and any(a not in FORMAL_AGENTS for a in included):
            raise ValueError('Formal profiles only support the six frozen agents')
        cfg['agents'] = {'include': included}
        settings.setdefault('agent', {})['agents'] = included
    if profile == 'leaderboard':
        agent = settings.setdefault('agent', {})
        if agent.get('sensor_protocol','camera_only') != 'camera_only' or agent.get('allow_real_lidar',False):
            raise RuntimeError('Real LiDAR is forbidden in camera-only leaderboard mode')
        agent.update(sensor_protocol='camera_only',allow_real_lidar=False)
        experiments = cfg.setdefault('responsibility_v2', {}).setdefault('experiments', {})
        for name in ('ablation', 'weight_sensitivity', 'window_sensitivity', 'size_sensitivity'):
            experiments[name] = False
        video = settings.setdefault('video', {})
        for name, item in video.get('metrics', {}).items():
            if isinstance(item, dict):
                item['enabled'] = name in ('fvd', 'epipolar', 'loftr')
        for name, item in video.get('visual_quality', {}).items():
            if isinstance(item, dict):
                item['enabled'] = name in ('clipiqa', 'cmmd', 'dino_temporal')
        trajectory = settings.setdefault('trajectory', {})
        for name in ('integrity', 'kinematics', 'map_compliance', 'collision', 'interaction', 'reference', 'distribution'):
            trajectory.setdefault('metrics', {}).setdefault(name, {})['enabled'] = name in ('integrity', 'kinematics', 'map_compliance')
        consistency = settings.setdefault('consistency', {})
        consistency['profile'] = 'core'
        for name, item in consistency.get('backends', {}).items():
            item['enabled'] = name == 'mtpc'
    return cfg


def public_values(value, legacy_export=False):
    """Remove deprecated aliases only at the public boundary, never from evidence."""
    if isinstance(value, dict):
        return {k: public_values(v, legacy_export) for k, v in value.items()
                if legacy_export or k not in DEPRECATED_ALIASES}
    if isinstance(value, list):
        return [public_values(v, legacy_export) for v in value]
    return value

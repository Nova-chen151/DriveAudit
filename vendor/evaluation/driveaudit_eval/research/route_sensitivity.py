"""Existing route-threshold sensitivity, applied only after frozen termination."""
from copy import deepcopy
from pathlib import Path


def attach_route_sensitivity(result, config, root):
    from driveaudit_eval.agents.input_gt import load_scene
    from driveaudit_eval.route_projection import correct_executed_route
    rows = []
    for relative in config['evaluation']['agent']['scene_paths']:
        path = Path(root)/relative
        _, _, route, _, _ = load_scene(path)
        for episode in result.get('episodes', []):
            if not episode['scene_id'].endswith(path.name):
                continue
            for threshold in (1.5, 2.5, 3.5):
                one = deepcopy({'episode': episode, 'windows': [], 'terminal_responsibility': [],
                                'route_timeseries': [r for r in result['route_timeseries'] if r['episode_id'] == episode['episode_id']]})
                options = dict(config['episode_rollout']['route_projection'], max_lateral_distance_m=threshold)
                correct_executed_route(one, route[:, :2], options)
                row = {k: one['episode'][k] for k in ('episode_id', 'scene_id', 'agent_name', 'route_completion_ratio',
                       'route_projection_coverage', 'route_completion_status', 'terminal_projection_reason')}
                rows.append(dict(row, max_lateral_distance_m=threshold))
    result['sensitivity'] = rows

"""Score saved predictions only; all reported driving scores are offline proxies."""
from pathlib import Path
import importlib.util
import json
import pickle


def evaluate(config, context):
    if config.get('mode') == 'input_gt_agents':
        from driveaudit_eval.agents.real_agents import evaluate as evaluate_real_agents
        return evaluate_real_agents(config, context)
    if config.get('mode') == 'input_gt':
        from driveaudit_eval.agents.input_gt import evaluate as evaluate_input_gt
        return evaluate_input_gt(config, context)
    root = Path(context['root'])
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else root / p
    def enabled(name):
        value = config.get(name, config.get('metrics', {}).get(name, True))
        return value.get('enabled', True) if isinstance(value, dict) else bool(value)
    groups = ('collision', 'dac', 'ttc', 'comfort', 'progress', 'open_loop_score',
              'trajectory_validity', 'lane_direction', 'supplemental')
    states = {name: {'status': 'PENDING' if enabled(name) else 'SKIPPED',
                     'reason': None if enabled(name) else 'disabled'} for name in groups}
    result = {'status': 'SKIPPED', 'metric_type': 'open_loop_proxy',
              'evaluation_mode': 'offline_open_loop', 'metrics': states}
    if not config.get('enabled', True) or not any(enabled(name) for name in groups):
        result['reason'] = 'disabled'
        for state in states.values(): state.update(status='SKIPPED', reason='disabled')
        return result
    missing = [key for key in ('manifest', 'predictions') if not config.get(key) or not path(config[key]).is_file()]
    deps = [name for name in ('numpy',) if importlib.util.find_spec(name) is None]
    if missing or deps:
        result['reason'] = 'Missing inputs: %s; dependencies: %s' % (missing, deps)
        for state in states.values():
            if state['status'] == 'PENDING': state.update(status='SKIPPED', reason=result['reason'])
        return result
    manifest = json.loads(path(config['manifest']).read_text())
    if not isinstance(manifest.get('scenes'), list):
        raise ValueError('Agent manifest must contain scenes')
    map_paths = [path(config['map_cache_root']) / s['scene_id'] / 'map/map_cache.h5' for s in manifest['scenes']] if config.get('map_cache_root') else ([path(config['map_cache'])] if config.get('map_cache') else [])
    map_ready = bool(map_paths) and all(p.is_file() for p in map_paths) and importlib.util.find_spec('h5py') is not None
    provider_config = config.get('map_provider', {})
    provider_ready = bool(provider_config.get('dataroot')) and path(provider_config['dataroot']).is_dir() and importlib.util.find_spec('nuscenes') is not None
    if context.get('dry_run'):
        for name, state in states.items():
            if state['status'] == 'PENDING':
                needs_map = name in ('dac', 'open_loop_score')
                needs_lane = name == 'lane_direction'
                ready = not ((needs_map and not (map_ready or provider_ready)) or (needs_lane and not provider_ready))
                state.update(status='READY' if ready else 'SKIPPED', reason=None if ready else 'Map data/provider unavailable')
        result.update(status='READY', inputs={'manifest': str(path(config['manifest'])), 'predictions': str(path(config['predictions']))}, heavy_models_executed=False)
        return result
    import numpy as np
    from driveaudit_eval.agents.core import stable_hash, write_csv, write_json
    from driveaudit_eval.agents.open_loop_trajectory_evaluator import MapCache, evaluate_open_loop_trajectory, aggregate
    from driveaudit_eval.agents.scoring import plan_metrics
    source = path(config['predictions'])
    if source.suffix in ('.pkl', '.pickle'):
        # Existing local agent outputs use pickle; input must be trusted.
        with source.open('rb') as handle: records = pickle.load(handle)
    elif source.suffix == '.jsonl': records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else: records = json.loads(source.read_text())
    if not isinstance(records, list): raise ValueError('Predictions must be a list of frame records')
    by_key = {}
    for record in records:
        key = (record['scene_id'], record['frame_id'])
        if key in by_key: raise ValueError('Duplicate prediction key: %r' % (key,))
        by_key[key] = record
    dt = float(config.get('dt', .5))
    if dt <= 0: raise ValueError('dt must be positive')
    rows, scenes = [], []
    bounds = config.get('vehicle_bounds', dict(max_speed=55., max_accel=2.6, max_decel=-4.5, R=1.5))
    for scene in manifest['scenes']:
        lookup = {f['frame_id']: f for f in scene['frames']}
        cache_path = path(config['map_cache_root']) / scene['scene_id'] / 'map/map_cache.h5' if config.get('map_cache_root') else (path(config['map_cache']) if config.get('map_cache') else None)
        cache = MapCache(cache_path, {'scenes': [scene]}) if cache_path and cache_path.is_file() else None
        lane_provider = None
        if provider_ready:
            try:
                from driveaudit_eval.utils.map_provider.nuscenes_provider import NuScenesMapProvider
                location = provider_config.get('scene_locations', {}).get(scene['scene_id'], scene.get('location', provider_config.get('location')))
                if location:
                    lane_provider = NuScenesMapProvider(str(path(provider_config['dataroot'])), location,
                        discretization_m=provider_config.get('discretization_m', .5),
                        drivable_layers=provider_config.get('drivable_layers', ['drivable_area', 'road_segment', 'road_block', 'lane', 'lane_connector', 'carpark_area']),
                        max_lane_distance_m=provider_config.get('max_lane_distance_m', 10.))
            except (ImportError, FileNotFoundError, OSError) as exc:
                states['lane_direction'].update(status='SKIPPED', reason=str(exc))
        tracks = None
        if enabled('supplemental') and all('timestamp' in frame for frame in scene['frames']):
            from driveaudit_eval.agents.scoring import scene_tracks
            tracks = scene_tracks(scene)
        part = []
        try:
            for index, frame in enumerate(scene['frames']):
                if not frame.get('evaluation_only', {}).get('scoreable', True): continue
                record = dict(by_key.get((scene['scene_id'], frame['frame_id']), dict(scene_id=scene['scene_id'], frame_id=frame['frame_id'], inference_success=False)))
                if record.get('inference_success'):
                    offsets = np.asarray(record.get('time_offsets') or [(i+1)*dt for i in range(len(record.get('trajectory', [])))], float)
                    if len(offsets) != len(record.get('trajectory', [])) or not np.isfinite(offsets).all() or np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0):
                        record.update(inference_success=False, error='invalid_time_offsets')
                row = evaluate_open_loop_trajectory(record, frame, lookup, cache, dt)
                if row.get('valid_trajectory'):
                    row['Comfort'] = row['comfort_score']
                    if enabled('supplemental') and 'velocity' in frame['model_input']['ego_state']:
                        state = frame['model_input']['ego_state']
                        row.update(plan_metrics(np.asarray(record['trajectory'])[:, :2], dt, state.get('velocity', [0, 0])[:2], state['yaw'], bounds))
                    if lane_provider is not None:
                        from driveaudit_eval.agents.score_audit.frame_audit import evaluate_with_provider
                        mapped = evaluate_with_provider(record, frame, lookup, lane_provider, dt=dt)
                        row.update({k: mapped[k] for k in ('lane_direction_error_mean', 'offroad_point_ratio', 'map_query_status')})
                        row['DAC_nuscenes_map_open_loop'] = mapped.get('map_provider_DAC')
                        # Keep raster and vector map protocols explicitly distinct.
                        row['PDMS_nuscenes_map_open_loop'] = mapped.get('PDMS_open_loop')
                        _lane_metrics(row, record, frame, lane_provider, dt)
                    if enabled('supplemental') and tracks is not None:
                        _timestamp_replay(row, record, scene, index, tracks, dt)
                    if enabled('supplemental') or enabled('ttc'):
                        _causal_metrics(row, record, scene, index, dt)
                part.append(row)
        finally:
            if cache is not None and cache.h5 is not None: cache.h5.close()
        rows.extend(part)
        scenes.append(aggregate(part, scene['scene_id']))
    summary = aggregate(rows, '__dataset__')
    # Supplemental scalar summaries use the existing scene-equal convention.
    supplement = ('EP_current_speed_proxy', 'EP_target_distance_m', 'wrong_way_point_ratio',
                  'reverse_driving_ratio', 'lane_direction_error_mean', 'wrong_way_lane_coverage',
                  'longitudinal_acceleration_violation', 'lateral_acceleration_violation', 'jerk_violation',
                  'ttc_plan_cv_capped3_s', 'ttc_plan_cv_under1s', 'ttc_plan_cv_under2s',
                  'collision_replay_1s', 'collision_replay_2s', 'collision_replay_3s')
    for scene_summary in scenes:
        selected = [row for row in rows if row['scene_id'] == scene_summary['scene_id']]
        for key in supplement:
            values = [row[key] for row in selected if row.get(key) is not None]
            scene_summary[key] = float(np.mean(values)) if values else None
    for key in supplement:
        values = [row[key] for row in scenes if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary['trajectory_valid_rate'] = sum(bool(r.get('valid_trajectory')) for r in rows) / len(rows) if rows else None
    summary['source_manifest_hash'] = stable_hash(manifest)
    summary['replay_protocol'] = 'legacy_12hz_frame_id; causal_cv uses native timestamps'
    for name, state in states.items():
        if state['status'] == 'PENDING': state.update(status='COMPLETED' if summary.get('valid_frames') else 'SKIPPED', reason=None if summary.get('valid_frames') else 'No valid replay windows')
    if not any(r.get('DAC') is not None for r in rows):
        for name in ('dac', 'open_loop_score'):
            if enabled(name): states[name].update(status='SKIPPED', reason='Raster map unavailable; vector-map variant reported separately if available')
    if not any(r.get('lane_direction_error_mean') is not None for r in rows) and enabled('lane_direction'):
        states['lane_direction'].update(status='SKIPPED', reason='Lane map unavailable or no matched lanes')
    for collection in (rows, scenes, [summary]):
        for row in collection:
            for key in list(row):
                group = _field_group(key)
                if group and not enabled(group): del row[key]
    output = Path(context['output_dir']) / 'agent_details'; output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_frame_metrics.csv', rows)
    write_csv(output / 'per_scene_metrics.csv', scenes)
    write_json(output / 'summary.json', summary)
    incomplete = any(enabled(name) and state['status'] != 'COMPLETED' for name, state in states.items())
    status = ('PARTIAL' if incomplete else 'COMPLETED') if summary.get('valid_frames') else 'SKIPPED'
    result.update(status=status, summary=summary,
                  artifacts=[str(output / name) for name in ('per_frame_metrics.csv', 'per_scene_metrics.csv', 'summary.json')])
    return result


def _field_group(key):
    if key in ('trajectory_valid_rate', 'valid_trajectory'): return 'trajectory_validity'
    if key.startswith(('PDMS', 'RC_', 'Driving_Score')): return 'open_loop_score'
    if key.startswith(('NC', 'collision', 'first_collision')): return 'collision'
    if key.startswith(('DAC', 'offroad', 'first_offroad', 'minimum_drivable', 'mean_drivable')): return 'dac'
    if key.startswith(('TTC', 'ttc', 'min_ttc')): return 'ttc'
    if key.startswith(('Comfort', 'comfort', 'longitudinal_acceleration', 'lateral_acceleration', 'jerk', 'yaw_', 'curvature')): return 'comfort'
    if key.startswith(('EP', 'valid_progress', 'expected_progress', 'lane_forward_progress')): return 'progress'
    if key.startswith(('lane_direction', 'wrong_way', 'reverse_driving')): return 'lane_direction'
    return None


def _lane_metrics(row, record, frame, provider, dt):
    import numpy as np
    from driveaudit_eval.utils.map_provider.geometry import trajectory_headings, wrap_angle
    from driveaudit_eval.agents.scoring import ego_progress_proxy
    state = frame['model_input']['ego_state']
    world = np.asarray(row['trajectory_world']); points = np.vstack((state['position'][:2], world))
    lane = provider.nearest_lane_center(world)
    matched = np.array([x['status'] == 'matched' for x in lane])
    statuses = [item.get('status') for item in lane]
    support = row['ep_spd_support'] = dict(status='unavailable', reason='lane_query_support_unverified',
        query_points=len(lane), matched_points=int(matched.sum()),
        lane_query_status_counts={status: statuses.count(status) for status in set(statuses)},
        evidence='executed_nearest_lane_center_query')
    if not matched.any():
        row['lane_direction_error_mean'] = None
        if len(lane) == len(world) and len(lane):
            absent = (getattr(provider, 'map_data', {}).get('semantic_support', {}).get('directed_lane_centerlines') is False
                and all(item.get('status') == 'unavailable' and item.get('reason') ==
                    'export contains no directed lane centerlines; native supplement not enabled' for item in lane))
            unmatched = all(status in ('unmatched', 'unmatched_lane') for status in statuses)
            if absent or unmatched:
                support.update(status='unobserved', reason='no_directed_lane_support' if absent else 'no_matched_directed_lanes')
        return
    headings = trajectory_headings(points, state['yaw'])[1:]
    direction = np.array([x.get('heading', np.nan) for x in lane])
    errors = np.full(len(lane), np.nan)
    errors[matched] = abs(wrap_angle(headings[matched] - direction[matched]))
    row['lane_direction_error_mean'] = float(np.mean(errors[matched]))
    wrong = (errors > np.pi/2) & (np.linalg.norm(np.diff(points, axis=0), axis=1)/dt > .5)
    row.update(wrong_way_point_ratio=float(np.mean(wrong[matched])), wrong_way_lane_coverage=float(np.mean(matched)))
    row['reverse_driving_ratio'] = row['wrong_way_point_ratio']
    progress = provider.route_progress(points)['progress_m']
    if 'velocity' not in state:
        support['reason'] = 'current_velocity_unavailable'
        return
    ep, target = ego_progress_proxy(progress, float(np.linalg.norm(state['velocity'][:2])))
    row.update(lane_forward_progress_proxy_m=progress, EP_current_speed_proxy=ep, EP_target_distance_m=target)
    support.update(status='available', reason=None)


def _causal_metrics(row, record, scene, index, dt):
    import numpy as np
    from driveaudit_eval.agents.scoring import causal_boxes, cv_ttc
    if index == 0 or any('timestamp' not in f for f in scene['frames'][index-1:index+1]): return
    if float(scene['frames'][index]['timestamp']) <= float(scene['frames'][index-1]['timestamp']): return
    frame = scene['frames'][index]; state = frame['model_input']['ego_state']
    origin = np.asarray(state['position'][:2]); world = np.asarray(row['trajectory_world'])
    velocity = (world[0] - origin)/dt; heading = np.arctan2(velocity[1], velocity[0])
    boxes = causal_boxes(scene, index); minimum = np.inf
    for box, pos, vel, matched in boxes:
        minimum = min(minimum, cv_ttc(origin, heading, velocity, pos, box['yaw'], vel, box['length'], box['width']))
    row.update(ttc_plan_cv_capped3_s=float(min(minimum, 3.)), ttc_plan_cv_finite_s=float(minimum) if np.isfinite(minimum) else None,
               ttc_plan_cv_under1s=float(minimum < 1), ttc_plan_cv_under2s=float(minimum < 2),
               causal_velocity_coverage=float(np.mean([b[3] for b in boxes])) if boxes else 1.)


def _timestamp_replay(row, record, scene, index, tracks, dt):
    import numpy as np
    from driveaudit_eval.agents.scoring import interpolate, overlap, replay_boxes, rates
    from driveaudit_eval.utils.map_provider.geometry import trajectory_headings
    frame = scene['frames'][index]; start = float(frame['timestamp'])/1e6
    if float(scene['frames'][-1]['timestamp'])/1e6 < start+3.: return
    world = np.asarray(row['trajectory_world'])
    offsets = np.asarray(record.get('time_offsets') or [(i+1)*dt for i in range(len(world))])
    # Supplement preserves the declared six-step, half-second protocol.
    if len(offsets) != 6 or not np.allclose(offsets, np.arange(1, 7)*.5): return
    state = frame['model_input']['ego_state']
    points = np.vstack((state['position'][:2], world)); grid = np.linspace(0, 3, 61)
    path = interpolate(points, np.arange(7)*.5, grid)
    headings = trajectory_headings(points, state['yaw'])
    ph = headings[np.clip(np.ceil(grid/.5).astype(int), 0, 6)]
    hit = np.zeros(len(grid), bool)
    for ident, data, valid in replay_boxes(tracks, start+grid):
        hit |= valid & overlap(path, ph, data[:, :2], data[:, 2], data[:, 3], data[:, 4])
    row.update(rates(hit, grid, 'collision_replay_'))

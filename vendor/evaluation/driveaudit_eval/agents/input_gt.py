"""Native-time saved-model or GT-oracle scoring; never executes model inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from driveaudit_eval.agents.core import (EGO_LENGTH, EGO_WIDTH, comfort, dac_from_ratios, route_progress,
                   world_to_ego, ego_to_world, write_csv, write_json, stable_hash)
from driveaudit_eval.agents.scoring import cv_ttc, overlap, plan_metrics
from driveaudit_eval.agents.score_audit.score_reconstructor import pdms_open_loop, driving_score_open_loop

SOURCE = 'ground_truth_oracle'
MODEL_SOURCE = 'model_prediction'
FILES = ('timeline.json', 'history/ego.csv', 'history/agents.csv', 'future/ego_trajectory.csv',
         'future/future_trajectories.csv', 'calibration/reference_ego_poses.json', 'map/map_meta.json', 'map/semantic_map.npz', 'generation_provenance.json', 'scene_context.json')


def sample(times, values, query, max_gap=.25):
    """Interpolate at actual timestamps, without extrapolation or bridging gaps."""
    times, values, query = np.asarray(times), np.asarray(values), np.asarray(query)
    if len(times) == 0: return np.empty((len(query), values.shape[-1])), np.zeros(len(query), bool)
    if np.any(np.diff(times) <= 0): raise ValueError('Source timestamps must be strictly increasing')
    data = np.column_stack([np.interp(query, times, values[:, i]) for i in range(values.shape[1])])
    exact = np.min(abs(query[:, None]-times[None, :]), axis=1) <= 1e-8
    ix = np.clip(np.searchsorted(times, query), 1, max(len(times)-1, 1))
    valid = (query >= times[0]-1e-8) & (query <= times[-1]+1e-8)
    if len(times) > 1: valid &= exact | ((times[ix]-times[ix-1]) <= max_gap+1e-8)
    else: valid &= exact
    return data, valid


def load_scene(path):
    timeline = json.loads((path/'timeline.json').read_text())
    if timeline['unit'] != 'microseconds_relative_to_frame_0': raise ValueError('Unsupported timeline units')
    timestamp = {int(row['frame_index']): float(row['reference_time_us'])/1e6 for row in timeline['frames']}
    poses = json.loads((path/'calibration/reference_ego_poses.json').read_text())
    if poses['coordinate_frame'] != 'history_frame_9_ego': raise ValueError('Unsupported scene coordinate frame')
    def read(name):
        with (path/name).open() as handle: return list(csv.DictReader(handle))
    ego = read('history/ego.csv') + read('future/ego_trajectory.csv')
    valid_ego = [r for r in ego if r['valid'].lower() == 'true']
    times = np.array([timestamp[int(r['frame_index'])] for r in valid_ego])
    values = np.array([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in valid_ego])
    values[:, 2] = np.unwrap(values[:, 2])
    actors = {}
    for row in read('history/agents.csv') + read('future/future_trajectories.csv'):
        if row['valid'].lower() != 'true': continue
        dimensions = [float(row[k]) for k in ('length', 'width')]
        if min(dimensions) <= 0: raise ValueError('Nonpositive actor dimensions')
        actors.setdefault(row['agent_id'], []).append([timestamp[int(row['frame_index'])],
            float(row['x']), float(row['y']), float(row['yaw']), *dimensions])
    for key, rows in actors.items():
        array = np.array(rows); array[:, 3] = np.unwrap(array[:, 3]); actors[key] = array
    return timestamp, times, values, actors, timeline


def _lane_provider_support(scene, error, canonical_map):
    """Only a validated partial export's expected rejection proves lane absence."""
    result = dict(status='unavailable', reason='lane_provider_unavailable', query_points=0,
                  matched_points=0, evidence='provider_not_available',
                  provider_error_type=type(error).__name__ if error is not None else None,
                  provider_error=str(error) if error is not None else None)
    expected = 'partial observed-surface map: unobserved cells are unknown; binary off-road metrics unavailable'
    if (type(error) is FileNotFoundError and str(error) == expected and canonical_map is not None
            and canonical_map.metadata.get('source_status') == 'RECONSTRUCTED_PARTIAL_OBSERVATIONS'
            and canonical_map.lane_direction_if_available is None):
        context = json.loads((Path(scene)/'scene_context.json').read_text())
        if context.get('dataset') in ('av2', 'nuplan', 'pandaset'):
            result.update(status='unobserved', reason='no_directed_lane_support',
                evidence='validated_observed_surface_export_without_native_directed_lanes',
                dataset=context['dataset'])
    return result


def score_window(start, times, ego, actors, provider, config, prediction=None, canonical_map=None, lane_support=None):
    dt, horizon = float(config.get('dt', .5)), float(config.get('horizon_s', 3.))
    if dt <= 0 or not np.isclose(horizon, 3.) or not np.isclose(horizon/dt, round(horizon/dt)) or horizon/dt < 3:
        raise ValueError('Native GT metrics require 3-second horizon and at least three equal positive steps')
    offsets = np.arange(1, round(horizon/dt)+1)*dt
    query = np.r_[start, start+offsets]
    values, valid = sample(times, ego, query, config.get('max_interpolation_gap_s', .25))
    row = {'source': MODEL_SOURCE if prediction is not None else SOURCE, 'model_inference_executed': False, 'metric_type': 'open_loop_proxy',
           'start_time_s': float(start), 'time_offsets_s': offsets.tolist(), 'valid_trajectory': bool(valid.all()),
           'ep_spd_support': dict(status='unavailable', reason='causal_ego_or_lane_support_unavailable',
                                  query_points=0, matched_points=0)}
    if not valid.all(): return dict(row, reason='incomplete_ego_time_support')
    origin, reference_world = values[0, :2], values[1:, :2]
    world, yaw = reference_world, values[1:, 2]
    if prediction is not None:
        from driveaudit_eval.utils.map_provider.geometry import trajectory_headings
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != (6, 2) or not np.isfinite(prediction).all() or not np.isclose(dt, .5):
            raise ValueError('Model prediction must be finite 6x2 at 0.5s steps')
        world = ego_to_world(prediction, [*origin, 0.], float(values[0, 2]))
        yaw = trajectory_headings(np.vstack((origin, world)), float(values[0, 2]))[1:]
        errors = np.linalg.norm(world-reference_world, axis=1)
        for h in (1, 2, 3):
            row[f'ADE_{h}s'] = float(np.mean(errors[offsets <= h]))
            row[f'FDE_{h}s'] = float(errors[np.flatnonzero(offsets <= h)[-1]])
        row['upstream_inference_success'] = True
    hit = np.zeros(len(offsets), bool); first_object = None; minimum_ttc = np.inf
    causal_current, causal_matched = 0, 0
    ix = np.searchsorted(times, start)
    previous = times[max(0, ix-1)]
    for ident, track in actors.items():
        other, observed = sample(track[:, 0], track[:, 1:], start+offsets, config.get('max_interpolation_gap_s', .25))
        collided = observed & overlap(world, yaw, other[:, :2], other[:, 2], other[:, 3], other[:, 4])
        if collided.any() and (not hit.any() or np.flatnonzero(collided)[0] < np.flatnonzero(hit)[0]): first_object = ident
        hit |= collided
        current, available = sample(track[:, 0], track[:, 1:], np.array([previous, start]), config.get('max_interpolation_gap_s', .25))
        if available[-1] and start > previous:
            causal_current += 1
            causal_matched += int(available[0])
            velocity = (current[1, :2]-current[0, :2])/(start-previous) if available[0] else np.zeros(2)
            minimum_ttc = min(minimum_ttc, cv_ttc(origin, values[0, 2], (world[0]-origin)/dt,
                current[1, :2], current[1, 2], velocity, current[1, 3], current[1, 4], horizon=horizon))
    first = float(offsets[np.flatnonzero(hit)[0]]) if hit.any() else None
    common = config.get('dac_protocol') == 'observed_common_v1'
    if common:
        if canonical_map is None: raise ValueError('observed_common_v1 requires CanonicalMapContext')
        support = canonical_map.dac_common(world,yaw,EGO_LENGTH,EGO_WIDTH,
            min_coverage=float(config.get('dac_min_coverage',.8)))
        row.update(support); row['dac_protocol'] = row.pop('protocol')
        row['dac_status'] = row.pop('status')
        row['DAC_reason'] = row.pop('reason')
        ratios = np.asarray([value if coverage >= support['coverage_threshold'] else np.nan
            for value,coverage in zip(support['per_step_drivable_fraction'],support['per_step_coverage'])],float)
        dac = support['DAC_common']
    else:
        try:
            ratios = np.asarray(provider.footprint_coverage(world, yaw, EGO_LENGTH, EGO_WIDTH)) if provider else np.array([])
        except ValueError as exc:
            ratios = np.array([]); row['map_skip_reason'] = str(exc)
        dac = dac_from_ratios(ratios.tolist()) if len(ratios) else None
    valid_steps = int(np.flatnonzero(hit)[0]) if hit.any() else len(world)
    if len(ratios) and (ratios < .2).any(): valid_steps = min(valid_steps, int(np.flatnonzero(ratios < .2)[0]))
    route = np.vstack((origin, ego[(times > start) & (times < start+horizon), :2], reference_world[-1]))
    progress, _ = route_progress(route, origin, world, valid_steps)
    expected, _ = route_progress(route, origin, reference_world, len(reference_world))
    ep = float(np.clip(progress/max(expected, 1e-3), 0, 1))
    local = world_to_ego(world, [*origin, 0.], float(values[0, 2])) if prediction is None else prediction
    c = comfort(local, dt)
    ttc = 0. if first is not None and first <= 1. else 1.
    score = pdms_open_loop(float(not hit.any()), dac, ttc, c['comfort_score'], ep)
    velocity = (values[1, :2]-origin)/dt
    row.update(NC=float(not hit.any()), collision_rate=float(hit.any()), first_collision_time=first,
        collision_object_id=first_object, collision_count=int(hit.sum()), DAC=dac,
        offroad_rate=(float(np.mean(ratios[np.isfinite(ratios)] < .2)) if np.isfinite(ratios).any() else None) if common else (float(np.mean(ratios < .2)) if len(ratios) else None),
        TTC_score=ttc, min_ttc=first, ttc_1s_risk=float(first is not None and first < 1),
        ttc_2s_risk=float(first is not None and first < 2), ttc_3s_risk=float(first is not None and first < 3),
        ttc_plan_cv_capped3_s=float(min(minimum_ttc, horizon)),
        ttc_plan_cv_finite_s=float(minimum_ttc) if np.isfinite(minimum_ttc) else None,
        Comfort=c['comfort_score'], comfort_violation_rate=len(c['comfort_violation_types'])/6., **{k: v for k, v in c.items() if k != 'kinematics'},
        EP=ep, EP_current_speed_proxy=None, valid_progress=progress, expected_progress=expected, PDMS_open_loop=score,
        lane_direction_error_mean=None, reverse_driving_ratio=None,
        causal_velocity_coverage=causal_matched/causal_current if causal_current else None, oracle_identity_comparison=prediction is None)
    current_ego, ok = sample(times, ego, np.array([previous, start]), config.get('max_interpolation_gap_s', .25))
    if ok.all() and start > previous:
        velocity = np.linalg.norm(current_ego[1, :2]-current_ego[0, :2])/(start-previous)
        row.update(plan_metrics(local, dt, [velocity, 0.], values[0, 2], config.get('vehicle_bounds',
            dict(max_speed=55., max_accel=2.6, max_decel=-4.5, R=1.5))))
        if provider is not None:
            from driveaudit_eval.agents.evaluator import _lane_metrics
            frame = {'model_input': {'ego_state': {'position': [*origin, 0.], 'yaw': float(values[0, 2]), 'velocity': [velocity, 0.]}}}
            row['trajectory_world'] = world.tolist()
            _lane_metrics(row, {}, frame, provider, dt)
            del row['trajectory_world']
        else:
            row['ep_spd_support'] = dict(lane_support or dict(status='unavailable',
                reason='lane_provider_unavailable', query_points=0, matched_points=0))
    return row


def summarize(rows):
    valid = [row for row in rows if row['valid_trajectory']]
    result = {'source': rows[0]['source'] if rows else SOURCE, 'model_inference_executed': False, 'metric_type': 'open_loop_proxy',
              'windows': len(rows), 'valid_windows': len(valid),
              'trajectory_valid_rate': len(valid)/len(rows) if rows else None}
    for key in ('NC', 'collision_rate', 'DAC', 'offroad_rate', 'TTC_score', 'Comfort', 'EP', 'PDMS_open_loop',
                'ttc_1s_risk', 'ttc_2s_risk', 'ttc_3s_risk', 'ttc_plan_cv_capped3_s',
                'longitudinal_acceleration_violation', 'lateral_acceleration_violation', 'jerk_violation',
                'comfort_violation_rate', 'EP_current_speed_proxy', 'EP_target_distance_m',
                'lane_forward_progress_proxy_m', 'lane_direction_error_mean', 'reverse_driving_ratio',
                'wrong_way_point_ratio', 'wrong_way_lane_coverage', 'kinematic_feasible_proxy',
                'ADE_1s', 'FDE_1s', 'ADE_2s', 'FDE_2s', 'ADE_3s', 'FDE_3s'):
        values = [row[key] for row in valid if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    events = [row['first_collision_time'] for row in valid if row.get('first_collision_time') is not None]
    result['first_collision_time'] = min(events) if events else None
    result['collision_event_windows'] = len(events)
    result['first_collision_time_status'] = 'observed' if events else 'no_collision_observed'
    windows = []; end = -np.inf
    for row in valid:
        if row['start_time_s'] >= end-1e-8:
            windows.append(row); end = row['start_time_s']+row['time_offsets_s'][-1]
    result['RC_open_loop'] = float(np.clip(sum(r['valid_progress'] for r in windows)/max(1e-6, sum(r['expected_progress'] for r in windows)), 0, 1)) if windows else None
    result['Driving_Score_open_loop'] = driving_score_open_loop(result['PDMS_open_loop'], result['RC_open_loop']) if windows else None
    if any(row.get('dac_protocol') == 'observed_common_v1' for row in valid):
        coverage = [row['dac_map_coverage'] for row in valid]
        result.update(dac_protocol='observed_common_v1',dac_map_coverage=float(np.mean(coverage)),
            dac_map_coverage_min=min(coverage),dac_map_coverage_p10=float(np.percentile(coverage,10)),
            dac_valid_windows=sum(row['DAC'] is not None for row in valid),
            dac_window_coverage=sum(row['DAC'] is not None for row in valid)/len(valid))
        for key in ('drivable_samples','observable_samples','total_samples','unknown_samples','non_drivable_samples'):
            result[key] = sum(row[key] for row in valid)
    support = [row.get('ep_spd_support', dict(status='unavailable', reason='support_not_reported')) for row in rows]
    observed = sum(item.get('status') == 'available' and row.get('EP_current_speed_proxy') is not None
                   for row, item in zip(rows, support))
    unobserved = sum(item.get('status') == 'unobserved' and row.get('EP_current_speed_proxy') is None
                     for row, item in zip(rows, support))
    unavailable = len(rows)-observed-unobserved
    reasons = [item.get('reason') or 'support_not_reported' for row, item in zip(rows, support)
               if not (item.get('status') == 'available' and row.get('EP_current_speed_proxy') is not None)]
    result['ep_spd_support'] = dict(status='unavailable' if unavailable or not rows else
        'available' if observed == len(rows) else 'partial_support' if observed else 'unobserved',
        total_windows=len(rows), observable_windows=observed, unobserved_windows=unobserved,
        unavailable_windows=unavailable, coverage=observed/len(rows) if rows else 0.,
        query_points=sum(item.get('query_points', 0) for item in support),
        matched_points=sum(item.get('matched_points', 0) for item in support),
        reason_counts={reason: reasons.count(reason) for reason in sorted(set(reasons))})
    return result


def evaluate(config, context, _records=None):
    if not config.get('enabled', True): return {'status': 'SKIPPED', 'reason': 'disabled', 'source': SOURCE}
    root, output = Path(context['root']), Path(context['output_dir'])/'agent_details'
    if config.get('dac_protocol') not in (None, 'observed_common_v1'):
        raise ValueError('Unsupported DAC protocol')
    if config.get('prediction_records'):
        if _records is not None: raise ValueError('Provide records or prediction_records, not both')
        record_path = Path(config['prediction_records'])
        if not record_path.is_absolute(): record_path = root/record_path
        _records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    predictions = _prediction_index(config, context, _records) if _records is not None else None
    source = MODEL_SOURCE if predictions is not None else SOURCE
    if predictions is not None:
        agent = config['agent']
        if Path(agent).name != agent or agent in ('.', '..'): raise ValueError('agent must be a plain name')
        output = output/agent
    scenes = [Path(p) if Path(p).is_absolute() else root/p for p in config.get('scene_paths', [])]
    if not scenes: return {'status': 'SKIPPED', 'reason': 'scene_paths required', 'source': SOURCE}
    required = FILES[:6]
    missing = [str(p/f) for p in scenes for f in required if not (p/f).is_file()]
    if missing: return {'status': 'SKIPPED', 'reason': 'Missing GT inputs', 'missing': missing, 'source': SOURCE}
    if context.get('dry_run'): return {'status': 'READY', 'source': source, 'scene_count': len(scenes), 'model_inference_executed': False}
    output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, evidence = [], [], []
    implementation = [Path(__file__), Path(__file__).with_name('core.py'), Path(__file__).with_name('scoring.py'),
                      root/'driveaudit_eval/utils/map_provider/input_gt_provider.py', root/'driveaudit_eval/utils/map_provider/opendrive_provider.py', root/'driveaudit_eval/utils/map_provider/nuscenes_provider.py',
                      root/'driveaudit_eval/metrics/capabilities.py', root/'driveaudit_eval/utils/map_provider/canonical_map.py', Path(__file__).with_name('evaluator.py')]
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation if p.is_file()}
    for scene in scenes:
        sid = scene.parent.parent.name+'__'+scene.name
        hashes = {name: hashlib.sha256((scene/name).read_bytes()).hexdigest() for name in FILES if (scene/name).is_file()}
        native = config.get('native_map', {})
        if native.get('dataroot'):
            location = json.loads((scene/'scene_context.json').read_text()).get('location')
            native_root = Path(native['dataroot'])
            if not native_root.is_absolute(): native_root = root/native_root
            source_map = native_root/'maps/expansion'/f'{location}.json'
            if source_map.is_file(): hashes[str(source_map)] = hashlib.sha256(source_map.read_bytes()).hexdigest()
        signature = stable_hash({'files': hashes, 'config': config, 'code': code_hashes, 'predictions': _records})
        cache = output/(sid+'.json')
        if config.get('resume', True) and cache.is_file() and (cached := json.loads(cache.read_text())).get('signature') == signature:
            rows, summary, provenance = cached['rows'], cached['summary'], cached['provenance']
        else:
            timestamp, times, ego, actors, timeline = load_scene(scene)
            provider = None; map_reason = None; map_error = None
            try:
                from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
                provider = InputGTMapProvider(scene, native_config=config.get('native_map')) if config.get('native_map') else InputGTMapProvider(scene)
            except (ImportError, FileNotFoundError, ValueError) as exc:
                map_reason = str(exc); map_error = exc
            from driveaudit_eval.metrics.capabilities import map_capabilities, apply_metric_requirements
            map_support = map_capabilities(scene)
            canonical_map = None
            if config.get('dac_protocol') == 'observed_common_v1':
                from driveaudit_eval.utils.map_provider.canonical_map import load_map_context
                canonical_map = load_map_context(scene)
            lane_support = _lane_provider_support(scene, map_error, canonical_map) if provider is None else None
            rows = []
            for frame_id, start in _windows(timestamp, times, config, sid):
                record = predictions[(sid, frame_id)] if predictions is not None else None
                row = score_window(start, times, ego, actors, provider, config, record['planning_traj'] if record else None,
                                   canonical_map=canonical_map, lane_support=lane_support)
                if canonical_map is None and not map_support['drivable_area']:
                    row = apply_metric_requirements(row, dict(map_support, trajectory=True, agent_plan=True, reference_route_proxy=True,
                        TTC=row.get('TTC_score') is not None, Comfort=row.get('Comfort') is not None, EP=row.get('EP') is not None))
                if record and not row['valid_trajectory']: raise ValueError(f'Incomplete GT time support: {sid}/{frame_id}')
                row.update(scene_id=sid, frame_id=frame_id)
                if record: row['agent'] = config['agent']
                rows.append(row)
            summary = dict(summarize(rows), scene_id=sid)
            provenance = {'scene_path': str(scene), 'source_sha256': hashes, 'implementation_sha256': code_hashes, 'source': source,
                'model_inference_executed': False, 'coordinate_frame': 'history_frame_9_ego',
                'time_policy': 'source reference_time_us; bounded interpolation at configured dt; no extrapolation', 'plan_dt_s': config.get('dt', .5), 'maximum_interpolation_gap_s': config.get('max_interpolation_gap_s', .25),
                'source_timing_metadata': {k: v for k, v in timeline.items() if k != 'frames'},
                'ego_footprint': {'length_m': EGO_LENGTH, 'width_m': EGO_WIDTH, 'source': 'existing evaluator fixed proxy; not measured source vehicle dimensions'},
                'collision_protocol': ('predicted tangent yaw from supplied trajectory incl origin; actual actor dimensions; existing OBB SAT' if predictions is not None else 'sampled native-time oracle ego yaw, actual actor dimensions, existing OBB SAT'),
                'EP_reference': 'fixed GT future route and GT progress denominator',
                'prediction_records_sha256': stable_hash(_records) if predictions is not None else None,
                'agent': config.get('agent') if predictions is not None else None,
                'upstream_inference_success': predictions is not None,
                'map_metadata': provider.query_metadata() if provider else None, 'map_skip_reason': map_reason,
                'lane_metrics_status': 'COMPLETED' if summary.get('lane_direction_error_mean') is not None else 'SKIPPED: no matched directed lane centerlines'}
            if canonical_map is not None:
                provenance.update(dac_protocol='observed_common_v1', map_metadata=canonical_map.metadata,
                    map_skip_reason=None, lane_map_skip_reason=map_reason,
                    offroad_termination_policy='step drivable fraction below 0.2 only when that step has sufficient observed coverage; unknown does not terminate progress')
            if config.get('retain_intermediates', True):
                temporary = cache.with_suffix('.tmp'); write_json(temporary, dict(signature=signature, rows=rows, summary=summary, provenance=provenance)); temporary.replace(cache)
        all_rows.extend(rows); summaries.append(summary); evidence.append(provenance)
    from driveaudit_eval.agents.evaluator import _field_group
    def enabled(group):
        value = config.get(group, config.get('metrics', {}).get(group, True))
        return value.get('enabled', True) if isinstance(value, dict) else bool(value)
    for row in all_rows + summaries:
        for key in list(row):
            group = _field_group(key)
            if group and not enabled(group): del row[key]
    write_csv(output/'per_frame_metrics.csv', all_rows); write_csv(output/'per_scene_metrics.csv', summaries)
    result = {'status': 'PARTIAL', 'source': source, 'metric_type': 'open_loop_proxy', 'model_inference_executed': False,
        'scene_count': len(summaries), 'valid_windows': sum(s['valid_windows'] for s in summaries), 'per_scene': summaries,
        'provenance': evidence}
    requirements = {'collision': ('NC', 'collision_rate'), 'dac': ('DAC', 'offroad_rate'),
        'ttc': ('TTC_score', 'ttc_1s_risk', 'ttc_2s_risk', 'ttc_3s_risk'),
        'comfort': ('Comfort', 'comfort_violation_rate'),
        'progress': ('EP', 'EP_current_speed_proxy'),
        'open_loop_score': ('PDMS_open_loop', 'RC_open_loop', 'Driving_Score_open_loop'),
        'trajectory_validity': ('trajectory_valid_rate',),
        'lane_direction': ('lane_direction_error_mean', 'reverse_driving_ratio'),
        'supplemental': ('longitudinal_acceleration_violation', 'lateral_acceleration_violation', 'jerk_violation')}
    if predictions is not None:
        result['agent'] = config['agent']
        result['upstream_inference_success'] = True
        requirements['planning_error'] = ('ADE_1s', 'FDE_1s', 'ADE_2s', 'FDE_2s', 'ADE_3s', 'FDE_3s')
    result['metrics'] = {}
    for group, fields in requirements.items():
        count = sum(all(row.get(key) is not None for key in fields) for row in summaries)
        status = ('COMPLETED' if count == len(summaries) else 'PARTIAL' if count else 'SKIPPED') if enabled(group) else 'SKIPPED'
        result['metrics'][group] = {'status': status, 'available_scenes': count, 'required_fields': list(fields)}
    incomplete = [group for group, state in result['metrics'].items() if enabled(group) and state['status'] != 'COMPLETED']
    result['status'] = 'PARTIAL' if incomplete else 'COMPLETED'
    if incomplete: result['reason'] = 'Unavailable requested metric groups: '+', '.join(incomplete)
    result['artifacts'] = [str(output/name) for name in ('per_frame_metrics.csv', 'per_scene_metrics.csv', 'summary.json')]
    write_json(output/'summary.json', result)
    return result


def _windows(timestamp, times, config, scene_id):
    horizon = float(config.get('horizon_s', 3.))
    selected = config.get('score_frame_ids')
    if isinstance(selected, dict): selected = selected.get(scene_id)
    selected = set(selected) if selected is not None else None
    result = []; last = -np.inf
    for frame_id, start in sorted(timestamp.items(), key=lambda item: item[1]):
        if start < timestamp[9]-1e-8 or start+horizon > times[-1]+1e-8 or start not in times: continue
        if selected is not None and frame_id not in selected: continue
        if start-last < float(config.get('window_stride_s', 0))-1e-8: continue
        result.append((frame_id, start)); last = start
    if selected is not None and {key for key, _ in result} != selected:
        raise ValueError(f'Requested score_frame_ids not scoreable: {scene_id}')
    return result


def prediction_windows(config, context):
    """The frozen scoring key set; never inferred from prediction availability."""
    root = Path(context['root']); result = []
    for name in config.get('scene_paths', []):
        scene = Path(name) if Path(name).is_absolute() else root/name
        timestamp, times, _, _, _ = load_scene(scene)
        sid = scene.parent.parent.name+'__'+scene.name
        result.extend(dict(scene_id=sid, frame_id=f, start_time_s=t) for f, t in _windows(timestamp, times, config, sid))
    return result


def _prediction_index(config, context, records):
    if not config.get('agent'): raise ValueError('agent is required for model prediction scoring')
    required = {(r['scene_id'], r['frame_id']) for r in prediction_windows(config, context)}
    if not required: raise ValueError('No scoreable prediction windows')
    legal = set(required)
    for name in config.get('scene_paths', []):
        scene = Path(name) if Path(name).is_absolute() else Path(context['root'])/name
        sid = scene.parent.parent.name+'__'+scene.name
        legal.update((sid, row['frame_index']) for row in json.loads((scene/'timeline.json').read_text())['frames'])
    by_key = {}
    for record in records:
        if not isinstance(record.get('frame_id'), int) or isinstance(record['frame_id'], bool):
            raise ValueError('Prediction frame_id must be an integer')
        key = (record['scene_id'], record['frame_id'])
        if key not in legal: raise ValueError(f'Unknown prediction scene/frame: {key}')
        if key in by_key: raise ValueError(f'Duplicate prediction key: {key}')
        by_key[key] = record
        if key not in required: continue  # Warmup records do not alter the frozen scoring set.
        if record.get('agent') != config['agent'] or record.get('inference_success') is not True:
            raise ValueError(f'Missing successful inference for {key}')
        if record.get('coordinate_system') != 'ego_x_forward_y_left': raise ValueError(f'Wrong prediction coordinates for {key}')
        trajectory = np.asarray(record.get('planning_traj'), float)
        offsets = np.asarray(record.get('time_offsets'), float)
        if trajectory.shape != (6, 2) or not np.isfinite(trajectory).all(): raise ValueError(f'Invalid planning_traj for {key}')
        if offsets.shape != (6,) or not np.allclose(offsets, np.arange(1, 7)*.5, rtol=0, atol=1e-8):
            raise ValueError(f'Invalid time_offsets for {key}')
    missing = required-by_key.keys()
    if missing: raise ValueError(f'Missing required predictions: {sorted(missing)}')
    return by_key


def evaluate_predictions(config, context, records):
    """Score caller-supplied real model outputs; this function performs no inference."""
    return evaluate(config, context, _records=records)

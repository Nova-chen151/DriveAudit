"""CARS-derived, finite-horizon counterfactual diagnostics, not legal liability.

The original SAT collision metric triggers evaluation and is never overwritten.
CARS source is compiled directly into a cached module without writing bytecode.
"""
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
import hashlib

import numpy as np
from shapely.geometry import Point, LineString

from driveaudit_eval.agents.input_gt import sample
from driveaudit_eval.traffic_compliance import evaluate_actor_compliance

DEFAULTS = dict(source_path='driveaudit_eval/vendor/cars/responsibility_models.py',
                desired_pre_collision_s=5., minimum_pre_collision_s=1.,
                post_collision_s=1., dt=.1, max_gap_s=.25, consensus_votes=2,
                bidirectional=True, lane_cross_section_half_length_m=20.)


@lru_cache(maxsize=4)
def _load(path, digest):
    module = ModuleType('_driveaudit_eval_cars_replay')
    module.__file__ = path
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError('CARS source changed while loading')
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def lane_measurement(provider, position, half_length):
    if provider is None:
        raise ValueError('lane_geometry_unavailable')
    match = provider.nearest_lane_center(np.asarray(position)[None])[0]
    if match.get('status') != 'matched':
        raise ValueError('target_lane_unmatched')
    native = getattr(provider, 'native', None)
    if native is not None:
        point = provider._world(np.asarray(position)[None])[0]
        yaw = match['heading']+provider.anchor_yaw
        polygon = native.lane_polygons.get(match['lane_id'])
    else:
        point, yaw = np.asarray(position), match['heading']
        polygon = next((x['polygon'] for x in provider.map_data.get('lanes', []) if x['key'] == match['lane_id']), None)
    if polygon is None or not polygon.covers(Point(point)):
        raise ValueError('target_not_inside_observed_lane_polygon')
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    cross = polygon.intersection(LineString([point-half_length*normal, point+half_length*normal]))
    segments = list(cross.geoms) if hasattr(cross, 'geoms') else [cross]
    segment = next((s for s in segments if s.distance(Point(point)) < 1e-6), None)
    if segment is None or segment.length <= 0 or segment.length >= 2*half_length-1e-6:
        raise ValueError('lane_cross_section_not_bounded')
    return dict(lane_id=match['lane_id'], lane_width_m=float(segment.length),
                distance_to_boundary_m=float(Point(point).distance(polygon.boundary)),
                method='matched_lane_polygon_normal_cross_section_at_episode_start')


def evaluate_collision_responsibility(events, trajectories_by_id, map_context, config, context):
    cfg = {**DEFAULTS, **(config or {})}
    if set(cfg.get('controllers', ['FSM', 'CCJP', 'RSS'])) != {'FSM', 'CCJP', 'RSS'}:
        raise ValueError('Responsibility consensus requires FSM, CCJP and RSS')
    if cfg.get('responsibility_adjusted_pdms', {}).get('enabled', False):
        raise ValueError('Responsibility-adjusted PDMS is not implemented; original NC/PDMS must remain unchanged')
    if not events:
        return dict(status='not_applicable', reason='no_original_collision', events=[], summary=_summary([]))
    if cfg['dt'] <= 0 or cfg['consensus_votes'] not in (1, 2, 3):
        raise ValueError('invalid replay dt/consensus threshold')
    path = Path(context['root'])/cfg['source_path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    load_reason = 'cars_source_missing'
    try:
        cars = _load(str(path), digest) if digest else None
    except ImportError as exc:
        cars = None
        load_reason = f'cars_dependency_missing: {exc}'
    output = []
    for number, source in enumerate(events):
        pair = source.get('collision_pair', source.get('agent_ids'))
        a, b = [trajectories_by_id.get(str(key)) for key in pair]
        first = float(source['first_collision_time'])
        scene_id = source.get('scene_id', a.scene_id if a is not None else b.scene_id if b is not None else None)
        row = dict(source, scene_id=scene_id, collision_event_id=source.get('collision_event_id', f'{scene_id}/collision/{number}'),
                   actor_a=str(pair[0]), actor_b=str(pair[1]), is_ego_involved=any(x is not None and x.agent_type == 'ego' for x in (a, b)),
                   first_collision_frame=source.get('first_collision_frame', int(np.argmin(abs(a.timestamps-first))) if a is not None else None),
                   responsibility_class='unscorable', responsibility_status='unscorable', directions={},
                   source_collision_geometry='original_DriveAudit_SAT; retained without CARS replacement')
        output.append(row)
        if a is None or b is None:
            row['reason'] = 'unsupported_actor_category_or_missing_canonical_trajectory'
            row['controllers_executed'] = False
            continue
        ranges = [x.timestamps[x.valid_mask] for x in (a, b)]
        if all(len(x) for x in ranges):
            history_start = max(first-cfg['desired_pre_collision_s'], *(x[0] for x in ranges))
            history_end = min(first+cfg['post_collision_s'], *(x[-1] for x in ranges))
            row['pre_collision_window'] = dict(requested_seconds=cfg['desired_pre_collision_s'],
                available_seconds=float(max(0., first-history_start)), coverage=float(max(0., first-history_start)/cfg['desired_pre_collision_s']),
                start_time=float(history_start), end_time=float(history_end), post_collision_seconds=float(max(0., history_end-first)))
        if cars is None:
            row['reason'] = load_reason
            continue
        if a.agent_type not in ('ego', 'vehicle') or b.agent_type not in ('ego', 'vehicle'):
            row['reason'] = 'pair_not_supported_by_vehicle_reference_driver_models'
            continue
        if a.coordinate_system != b.coordinate_system:
            row['reason'] = 'coordinate_frame_mismatch'
            continue
        ia, ib = np.flatnonzero(a.valid_mask), np.flatnonzero(b.valid_mask)
        if not len(ia) or not len(ib):
            row['reason'] = 'no_valid_trajectory'
            continue
        start = max(first-cfg['desired_pre_collision_s'], a.timestamps[ia[0]], b.timestamps[ib[0]])
        end = min(first+cfg['post_collision_s'], a.timestamps[ia[-1]], b.timestamps[ib[-1]])
        row['pre_collision_window'] = dict(requested_seconds=cfg['desired_pre_collision_s'],
            available_seconds=float(max(0., first-start)), coverage=float(max(0., first-start)/cfg['desired_pre_collision_s']),
            start_time=float(start), end_time=float(end), post_collision_seconds=float(max(0., end-first)))
        if first-start < cfg['minimum_pre_collision_s'] or end < first:
            row['reason'] = 'insufficient_pre_collision_history'
            continue
        query = start+np.arange(int(np.floor((end-start)/cfg['dt']))+1)*cfg['dt']
        arrays = []
        for track, indices in ((a, ia), (b, ib)):
            values = np.column_stack([track.positions[indices], np.unwrap(track.headings[indices])])
            data, valid = sample(track.timestamps[indices], values, query,
                                 max_gap=float(track.metadata.get('replay_max_gap_s', cfg['max_gap_s'])))
            arrays.append(data if valid.all() else None)
        if any(x is None for x in arrays) or len(query) < 2 or query[-1] < first:
            row['reason'] = 'trajectory_gap_or_collision_outside_uniform_replay_horizon'
            continue
        row['replay_times'] = query.tolist()
        row['traffic_before_collision'] = {}
        for track in (a, b):
            pre = track.clone()
            pre.valid_mask &= (pre.timestamps >= start) & (pre.timestamps < first)
            pre.metadata['history_frames'] = 0
            row['traffic_before_collision'][track.agent_id] = evaluate_actor_compliance(
                track.agent_id, pre, list(trajectories_by_id.values()), map_context, config.get('traffic_config', {}))
        for target, adversary, x, y in ((a, b, *arrays), (b, a, *arrays[::-1])):
            if target is b and not cfg['bidirectional']:
                continue
            direction = dict(target=target.agent_id, adversary=adversary.agent_id, controllers={}, status='unscorable')
            row['directions'][target.agent_id] = direction
            ext = [target.shape['length'], target.shape['width']]
            adv_ext = [adversary.shape['length'], adversary.shape['width']]
            baseline = cars.check_raw_collision(x[:, :2], x[:, 2], y[:, :2], y[:, 2], ext, adv_ext)
            direction['cars_baseline'] = dict(collision=bool(baseline[0]), collision_step=baseline[1], min_distance=float(baseline[2]))
            geometry_valid = baseline[0] and baseline[1] != 0
            try:
                lane = lane_measurement(map_context.provider, x[0, :2], cfg['lane_cross_section_half_length_m'])
                direction['lane_evidence'] = lane
            except ValueError as exc:
                lane = None
                direction['lane_reason'] = str(exc)
            for name, cls in [('FSM', cars.FSMController), ('CCJP', cars.CCJPLaneAwareController), ('RSS', cars.RSSController)]:
                if name == 'CCJP' and lane is None:
                    direction['controllers'][name] = dict(status='unavailable', reason=direction['lane_reason'], raw=None)
                    continue
                controller = cls(dt=cfg['dt'], **cfg.get('controller_parameters', {}).get(name, {}))
                if name == 'CCJP':
                    dtr = cars.compute_dist_to_react(lane['lane_width_m'], adv_ext[1], lane['distance_to_boundary_m'])
                    controller.set_dist_to_react(dtr)
                    direction['lane_evidence']['dist_to_react_m'] = dtr
                raw = cars.replay_episode(x[:, :2], x[:, 2], y[:, :2], y[:, 2], ext, adv_ext, controller, dt=cfg['dt'])
                direction['controllers'][name] = dict(status='valid' if geometry_valid else 'unscorable',
                    reason=None if geometry_valid else 'baseline_collision_missing_or_initial_overlap', raw=raw,
                    raw_non_finite_fields={k: str(v) for k, v in raw.items()
                        if isinstance(v, (float, np.floating)) and not np.isfinite(v)})
            valid = [v for v in direction['controllers'].values() if v['status'] == 'valid']
            direction['avoidable_votes'] = sum(bool(v['raw']['preventable']) for v in valid) if len(valid) == 3 else None
            direction['status'] = 'valid' if len(valid) == 3 else 'unscorable'
        votes = [row['directions'].get(track.agent_id, {}).get('avoidable_votes') for track in (a, b)]
        if None not in votes:
            ma, mb = [v >= cfg['consensus_votes'] for v in votes]
            row['responsibility_class'] = ('shared_or_ambiguous' if ma and mb else 'actor_a_attributed' if ma else
                                          'actor_b_attributed' if mb else 'unavoidable_or_model_limit')
            row['responsibility_status'] = 'valid'
            row['reason'] = None
        else:
            row['reason'] = 'three_valid_controller_votes_required_in_both_directions'
        row['actor_a_avoidable_votes'], row['actor_b_avoidable_votes'] = votes
    return dict(status='completed', events=output, summary=_summary(output),
                provenance=dict(source_path=str(path), source_sha256=digest, config=cfg,
                    method='CARS-derived model-based counterfactual attribution',
                    scope='finite-horizon pairwise longitudinal braking with fixed adversary; not legal liability',
                    min_distance_definition='CARS signed corner margin, not Euclidean box distance',
                    fuzzy_metric_scope='max_pfs/max_cfs and fuzzy criticality meaningful for FSM only; other raw zeros retained'))


def _summary(events):
    ego = [x for x in events if x['is_ego_involved']]
    non = [x for x in events if not x['is_ego_involved']]
    result = dict(ego_collision_count=len(ego), non_ego_collision_count=len(non),
                ego_responsibility_distribution=dict(Counter(x['responsibility_class'] for x in ego)),
                non_ego_responsibility_distribution=dict(Counter(x['responsibility_class'] for x in non)),
                unscorable_collision_count=sum(x['responsibility_class'] == 'unscorable' for x in events))
    attributed = sum((x['responsibility_class'] == 'actor_a_attributed' and x['actor_a'] == 'ego') or
                     (x['responsibility_class'] == 'actor_b_attributed' and x['actor_b'] == 'ego') for x in ego)
    other = sum((x['responsibility_class'] == 'actor_a_attributed' and x['actor_a'] != 'ego') or
                (x['responsibility_class'] == 'actor_b_attributed' and x['actor_b'] != 'ego') for x in ego)
    result.update(ego_attributed_collision_count=attributed, other_attributed_collision_count=other,
        shared_collision_count=sum(x['responsibility_class'] == 'shared_or_ambiguous' for x in ego),
        unavoidable_collision_count=sum(x['responsibility_class'] == 'unavoidable_or_model_limit' for x in ego),
        ego_attributed_collision_rate=attributed/len(ego) if ego else None,
        rate_denominator='all_original_ego_collision_events; unscorable counts separately reported')
    for name in ('FSM', 'CCJP', 'RSS'):
        rows = [x.get('directions', {}).get('ego', {}).get('controllers', {}).get(name, {}) for x in ego]
        valid = [r['raw'] for r in rows if r.get('status') == 'valid']
        rate = float(np.mean([r['preventable'] for r in valid])) if valid else None
        result[name.lower()+'_preventable_rate'] = rate
        result[name.lower()+'_unpreventable_rate'] = 1-rate if rate is not None else None
        result[name.lower()+'_valid_event_count'] = len(valid)
        for key in ('min_ttc', 'speed_at_collision', 'collision_angle', 'max_braking_deficit', 'bd_at_trigger'):
            values = [float(r[key]) for r in valid if r.get(key) is not None and np.isfinite(r[key])]
            result[name.lower()+'_'+key] = dict(min=min(values), mean=float(np.mean(values)), max=max(values)) if values else None
        result[name.lower()+'_criticality_distribution'] = dict(Counter(r.get('criticality') for r in valid)) if name == 'FSM' else None
    return result

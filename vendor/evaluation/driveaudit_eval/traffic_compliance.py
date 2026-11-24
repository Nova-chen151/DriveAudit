"""Shared vehicle rule diagnostics on the existing Trajectory schema.

Sources: third_party/Onsite_UT2SG_Testing-main/av_ego_scoring/
EvaluationSystem-onsite2025/evaluateUtils/{standard_parameter.py,
TrafficRule_compliance/{detect_behaviors.py,rules_dict.py,pipeline.py,
get_speed_limit_function.py}}. Port rule selection by observed context and
geometry, not the legacy cwd-dependent evaluator or its overall score.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from shapely.geometry import Point, LineString

from driveaudit_eval.trajectory.map_rule_metrics import obb_polygon

RULES = ('offroad', 'lane_departure', 'wrong_way', 'solid_line_crossing', 'prohibited_area',
         'special_lane_misuse', 'emergency_lane_misuse', 'wrong_lane_usage', 'route_direction',
         'overspeed', 'underspeed', 'red_light', 'stop_sign', 'stop_line_encroachment',
         'failure_to_yield_vehicle', 'failure_to_yield_pedestrian', 'unsafe_following',
         'unsafe_lane_change', 'roundabout_yield', 'turn_vs_straight_yield',
         'illegal_reverse', 'illegal_uturn', 'illegal_stop', 'illegal_overtaking')
# Active standard_parameter.Parameter.weightData.indexWeightDict entries, not commented alternatives.
ONSITE_WEIGHTS = dict(offroad=10., lane_departure=10., wrong_way=2.5, solid_line_crossing=2.5,
                     overspeed=2.5, prohibited_area=2.5, red_light=5., wrong_lane_usage=2.5,
                     stop_line_encroachment=2.5, stop_sign=2.5)
DEFAULTS = dict(minimum_occupancy=.99, wrong_way_angle_rad=math.pi/2,
                minimum_moving_speed_mps=.5, speed_tolerance_mps=0., max_gap_s=.25,
                stopped_speed_mps=.1, stop_duration_s=1., approach_distance_m=10.,
                uturn_window_s=5., uturn_heading_change_rad=2.6, overtaking_heading_tolerance_rad=math.pi/4)


@dataclass
class MapContext:
    """All supplied geometry and trajectories must share a fixed XY frame.

None means not observed; an empty region list means observed and absent.
Regions carry polygon plus explicit restrictions; signals carry line geometry,
forward_normal, and observed state intervals [start,end,state].
"""
    provider: object = None
    regions: dict = field(default_factory=dict)
    solid_lines: object = None
    speed_limit_mps: object = None
    minimum_speed_mps: object = None
    signals: object = None
    stop_signs: object = None
    lane_area: object = None
    route_heading: object = None
    following_minimum_distance_m: object = None
    priority_zones: object = None
    provenance: dict = field(default_factory=dict)


def _series(value, track):
    if callable(value):
        values = [value(float(t), p) for t, p in zip(track.timestamps, track.positions)]
    else:
        values = np.broadcast_to(value, (len(track.timestamps),))
    return np.asarray(values, dtype=float)


def _rule(track, name, flags, observed, config, evidence=None):
    t = np.asarray(track.timestamps)
    valid = np.asarray(track.valid_mask, bool).copy()
    start = int(track.metadata.get('history_frames', 0))
    valid[:start] = False
    dt = np.r_[np.diff(t), 0.]
    consecutive = np.r_[valid[:-1] & valid[1:], False]
    weights = np.where(consecutive & (dt <= config['max_gap_s']), dt, 0.)
    if flags is None:
        return dict(actor_id=track.agent_id, rule_id=name, applicable=True, observable=False,
                    violated=None, violation_count=None, duration=None, exposure_duration=0.,
                    violation_ratio=None, severity=None, rule_score=None, status='unavailable',
                    reason=(evidence or {}).get('reason', name+'_context_not_provided'), coverage=0., evidence=evidence or {}, events=[])
    observed = np.asarray(observed, bool) & valid
    flags = np.asarray(flags, bool) & observed
    exposure = float(weights[observed].sum())
    duration = float(weights[flags].sum())
    events, active = [], None
    for index in np.flatnonzero(flags):
        if active is None or index != active[-1]+1 or t[index]-t[active[-1]] > config['max_gap_s']:
            if active is not None:
                events.append(active)
            active = [int(index)]
        else:
            active.append(int(index))
    if active is not None:
        events.append(active)
    event_rows = [dict(scene_id=track.scene_id, actor_id=track.agent_id, rule_id=name,
                       start_time=float(t[ix[0]]), end_time=float(t[ix[-1]]+weights[ix[-1]]),
                       frame_ids=ix, duration=float(weights[ix].sum())) for ix in events]
    ratio = duration/exposure if exposure > 0 else None
    discrete = name in ('red_light', 'stop_sign', 'illegal_uturn', 'illegal_overtaking')
    scorable = bool(observed.any()) if discrete else exposure > 0
    score = float(not flags.any()) if discrete and scorable else 1-ratio if ratio is not None else None
    return dict(actor_id=track.agent_id, rule_id=name, applicable=True, observable=scorable,
                violated=bool(flags.any()) if scorable else None,
                violation_count=len(events) if scorable else None,
                start_time=event_rows[0]['start_time'] if events else None,
                end_time=event_rows[-1]['end_time'] if events else None,
                duration=duration if scorable else None, exposure_duration=exposure,
                violation_ratio=ratio, severity=1-score if score is not None else None, rule_score=score,
                status='valid' if scorable else 'unscorable',
                reason=None if scorable else 'no_observed_time_exposure',
                coverage=exposure/weights.sum() if weights.sum() > 0 else 0.,
                evidence=dict(evidence or {}, observed_frame_ids=np.flatnonzero(observed).tolist(),
                              violation_frame_ids=np.flatnonzero(flags).tolist(), duration_policy='left_endpoint_native_intervals_no_gap_bridging',
                              score_policy='any_discrete_event_violation' if discrete else 'one_minus_violation_time_ratio'),
                events=event_rows)


def evaluate_actor_compliance(actor_id, trajectory, all_actor_trajectories, map_context, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    track, context = trajectory, map_context
    if str(actor_id) != str(track.agent_id):
        raise ValueError('actor identity mismatch')
    if track.agent_type not in ('ego', 'vehicle'):
        return dict(actor_id=actor_id, scene_id=track.scene_id, status='not_applicable', reason='vehicle_rules_only', rules={})
    errors = track.validate()
    if errors:
        raise ValueError(str(errors))
    n, p, t = len(track.timestamps), track.positions, track.timestamps
    speed = np.asarray(track.velocities, float)
    moving = np.isfinite(speed) & (speed > cfg['minimum_moving_speed_mps'])
    valid = np.asarray(track.valid_mask, bool)
    rules = {name: _rule(track, name, None, valid, cfg) for name in RULES}
    polygons = [obb_polygon(x, yaw, track.shape['length'], track.shape['width']) for x, yaw in zip(p, track.headings)]
    def put(name, flags, observed=valid, **evidence):
        rules[name] = _rule(track, name, flags, observed, cfg, evidence)
    provider = context.provider
    if provider is not None:
        map_coordinates = getattr(provider, 'coordinate_system', 'world')
        if map_coordinates == 'global_xy_meters':
            map_coordinates = 'world'
        if track.coordinate_system != map_coordinates:
            raise ValueError('map/trajectory coordinate mismatch')
        coverage, observed = np.full(n, np.nan), np.zeros(n, bool)
        for index in np.flatnonzero(valid):
            try:
                coverage[index] = provider.footprint_coverage(p[index:index+1], track.headings[index:index+1],
                    track.shape['length'], track.shape['width'])[0]
                observed[index] = np.isfinite(coverage[index])
            except ValueError:
                pass  # Outside an exported map is unknown, not offroad.
        put('offroad', coverage < cfg['minimum_occupancy'], observed,
            occupancy=coverage.tolist(), minimum_occupancy=cfg['minimum_occupancy'])
        lanes = provider.nearest_lane_center(p)
        matched = np.array([x.get('status') == 'matched' and np.isfinite(x.get('heading', np.nan)) for x in lanes])
        angle = np.abs((track.headings-np.array([x.get('heading', np.nan) for x in lanes])+np.pi)%(2*np.pi)-np.pi)
        put('wrong_way', moving & (angle > cfg['wrong_way_angle_rad']), matched & valid,
            lane_ids=[x.get('lane_id') for x in lanes], heading_error_rad=angle.tolist(),
            minimum_speed_mps=cfg['minimum_moving_speed_mps'])
    if context.lane_area is not None:
        put('lane_departure', [not context.lane_area.covers(poly) for poly in polygons],
            definition='outside_union_of_permitted_lane_areas; ordinary_lane_change_not_penalized')
    if context.route_heading is not None:
        route = _series(context.route_heading, track)
        angle = np.abs((track.headings-route+np.pi)%(2*np.pi)-np.pi)
        put('route_direction', moving & (angle > cfg['wrong_way_angle_rad']), valid & np.isfinite(route),
            prescribed_route_headings=route.tolist())
    for name, attribute, sign in [('overspeed', 'speed_limit_mps', 1), ('underspeed', 'minimum_speed_mps', -1)]:
        limit = getattr(context, attribute)
        if limit is None:
            continue
        limits = _series(limit, track)
        observed = valid & np.isfinite(speed) & np.isfinite(limits) & (limits > 0)
        exceed = sign*(speed-limits)
        positive = np.maximum(exceed, 0)
        put(name, exceed > cfg['speed_tolerance_mps'], observed, limits_mps=limits.tolist(),
            max_exceedance_mps=float(positive[observed].max()) if observed.any() else None,
            max_exceedance_kmh=float(positive[observed].max()*3.6) if observed.any() else None,
            mean_positive_exceedance_mps=float(positive[observed & (positive > 0)].mean()) if np.any(observed & (positive > 0)) else 0. if observed.any() else None,
            relative_exceedance=np.divide(positive, limits, out=np.full(n, np.nan), where=limits>0).tolist())
    if context.solid_lines is not None:
        put('solid_line_crossing', [any(poly.intersects(line) for line in context.solid_lines) for poly in polygons],
            definition='footprint_touch_or_cross_solid_boundary; dashed_lines_excluded')
    for name, regions in context.regions.items():
        if name not in RULES:
            raise ValueError('unknown restriction: '+name)
        if regions is None:
            continue
        flags = np.zeros(n, bool)
        for region in regions:
            inside = np.array([poly.intersects(region['polygon']) for poly in polygons])
            if name == 'illegal_stop':
                inside &= speed <= cfg['stopped_speed_mps']
            elif name == 'illegal_uturn':
                yaw = np.unwrap(track.headings)
                turns = np.zeros(n, bool)
                for index in range(1, n):
                    begin = int(np.searchsorted(t, t[index]-cfg['uturn_window_s']))
                    if valid[begin:index+1].all() and np.all(np.diff(t[begin:index+1]) <= cfg['max_gap_s']):
                        turns[index] = abs(yaw[index]-yaw[begin]) >= cfg['uturn_heading_change_rad']
                inside &= turns & moving
            elif name == 'illegal_reverse':
                delta = np.vstack([np.zeros((1, 2)), np.diff(p, axis=0)])
                inside &= np.sum(delta*np.column_stack([np.cos(track.headings), np.sin(track.headings)]), axis=1) < 0
            elif name == 'illegal_overtaking':
                if provider is None or 'overtaking_lane_pairs' not in region:
                    continue
                from driveaudit_eval.agents.input_gt import sample
                passes = np.zeros(n, bool)
                own_lanes = provider.lane_id(p)
                allowed_pairs = {tuple(pair) for pair in region['overtaking_lane_pairs']}
                for other in all_actor_trajectories:
                    if other.agent_id == actor_id or other.agent_type not in ('vehicle', 'ego'):
                        continue
                    indices = np.flatnonzero(other.valid_mask)
                    other_state, supported = sample(other.timestamps[indices],
                        np.column_stack([other.positions[indices], np.unwrap(other.headings[indices])]), t, cfg['max_gap_s'])
                    other_lanes = provider.lane_id(other_state[:, :2])
                    compatible = np.array([(a, b) in allowed_pairs for a, b in zip(own_lanes, other_lanes)])
                    relative = np.sum((other_state[:, :2]-p)*np.column_stack([np.cos(track.headings), np.sin(track.headings)]), axis=1)
                    angle = np.abs((other_state[:, 2]-track.headings+np.pi)%(2*np.pi)-np.pi)
                    passes[1:] |= supported[1:] & supported[:-1] & valid[1:] & valid[:-1] & compatible[1:] & (
                        np.diff(t) <= cfg['max_gap_s']) & (relative[:-1] > 0) & (relative[1:] <= 0) & (
                        angle[1:] <= cfg['overtaking_heading_tolerance_rad'])
                inside &= passes & moving
            elif name in ('special_lane_misuse', 'emergency_lane_misuse', 'wrong_lane_usage'):
                allowed = region.get('allowed_actor_ids')
                if allowed is None:
                    continue
                if actor_id in allowed:
                    continue
            elif name not in ('prohibited_area',):
                continue
            flags |= inside
        supported = name in ('prohibited_area', 'illegal_stop', 'illegal_reverse', 'illegal_uturn') or (
            name in ('special_lane_misuse', 'emergency_lane_misuse', 'wrong_lane_usage') and all('allowed_actor_ids' in r for r in regions)) or (
            name == 'illegal_overtaking' and provider is not None and all('overtaking_lane_pairs' in r for r in regions))
        if supported:
            put(name, flags, restriction_count=len(regions))
    if context.signals is not None:
        flags, observed = np.zeros(n, bool), np.zeros(n, bool)
        for signal in context.signals:
            line, normal = signal['line'], np.asarray(signal['forward_normal'], float)
            for index in range(1, n):
                if not valid[index-1:index+1].all() or t[index]-t[index-1] > cfg['max_gap_s']:
                    continue
                front = p[index]+track.shape['length']/2*np.array([np.cos(track.headings[index]), np.sin(track.headings[index])])
                previous = p[index-1]+track.shape['length']/2*np.array([np.cos(track.headings[index-1]), np.sin(track.headings[index-1])])
                crossing = LineString([previous, front]).intersects(line) and np.dot(front-previous, normal) > 0
                crossing_time = t[index]
                if crossing:
                    alpha = np.dot(np.asarray(line.coords[0])-previous, normal)/np.dot(front-previous, normal)
                    crossing_time = t[index-1]+np.clip(alpha, 0., 1.)*(t[index]-t[index-1])
                state = next((r[2] for r in signal.get('states', []) if r[0] <= crossing_time < r[1]), None)
                if state is None:
                    continue
                observed[index] = True
                flags[index] |= crossing and str(state).lower() == 'red'
        put('red_light', flags, observed, signal_observable=True, signal_count=len(context.signals))
    if context.stop_signs is not None:
        flags, observed, encroachment = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        approaches = []
        for sign in context.stop_signs:
            line, normal = sign['line'], np.asarray(sign['forward_normal'], float)
            anchor = np.asarray(line.coords[0])
            front = p+track.shape['length']/2*np.column_stack([np.cos(track.headings), np.sin(track.headings)])
            side = (front-anchor)@normal
            near = valid & np.array([Point(x).distance(line) <= cfg['approach_distance_m'] for x in front])
            observed |= near
            encroachment |= near & (speed <= cfg['stopped_speed_mps']) & (side > 0)
            for index in range(1, n):
                if not valid[index-1:index+1].all() or t[index]-t[index-1] > cfg['max_gap_s']:
                    continue
                if side[index-1] < 0 <= side[index] and LineString(front[index-1:index+1]).intersects(line):
                    # Only the current contiguous approach counts, never a previous stop elsewhere.
                    beginning = index-1
                    while beginning > 0 and near[beginning-1] and t[beginning]-t[beginning-1] <= cfg['max_gap_s']:
                        beginning -= 1
                    best, current = 0., 0.
                    for j in range(beginning, index):
                        current = current+(t[j+1]-t[j]) if speed[j] <= cfg['stopped_speed_mps'] and side[j] <= 0 else 0.
                        best = max(best, current)
                    flags[index] |= best < cfg['stop_duration_s']
                    approaches.append(dict(crossing_frame=index, stop_duration_s=float(best),
                        minimum_approach_speed_mps=float(np.min(speed[beginning:index+1]))))
        put('stop_sign', flags, observed, approaches=approaches, minimum_stop_duration_s=cfg['stop_duration_s'])
        put('stop_line_encroachment', encroachment, observed)
    if context.following_minimum_distance_m is not None and provider is not None:
        from driveaudit_eval.agents.input_gt import sample
        limits = _series(context.following_minimum_distance_m, track)
        own_lanes = provider.lane_id(p)
        flags, observed = np.zeros(n, bool), np.zeros(n, bool)
        measurements = []
        for other in all_actor_trajectories:
            if other.agent_id == actor_id or other.agent_type not in ('vehicle', 'ego'):
                continue
            if other.coordinate_system != track.coordinate_system:
                raise ValueError('interaction coordinate mismatch')
            ix = np.flatnonzero(other.valid_mask)
            other_positions, support = sample(other.timestamps[ix], other.positions[ix], t, cfg['max_gap_s'])
            lane_ids = provider.lane_id(other_positions)
            delta = other_positions-p
            ahead = (delta*np.column_stack([np.cos(track.headings), np.sin(track.headings)])).sum(1)
            gap = ahead-(track.shape['length']+other.shape['length'])/2
            same_lane = np.array([a is not None and a == b for a, b in zip(own_lanes, lane_ids)])
            applies = valid & support & same_lane & (ahead > 0) & np.isfinite(limits)
            observed |= applies
            flags |= applies & (gap < limits)
            measurements.append(dict(other_actor_id=other.agent_id, gap_m=gap.tolist()))
        put('unsafe_following', flags, observed, explicitly_supplied_minimum_distance_m=limits.tolist(), pairs=measurements)
    if context.priority_zones is not None:
        from driveaudit_eval.agents.input_gt import sample
        for name in ('failure_to_yield_vehicle', 'failure_to_yield_pedestrian', 'roundabout_yield', 'turn_vs_straight_yield', 'unsafe_lane_change'):
            zones = [z for z in context.priority_zones if z['rule_id'] == name and actor_id in z['yield_actor_ids']]
            if not zones:
                continue
            flags, observed = np.zeros(n, bool), np.zeros(n, bool)
            for zone in zones:
                for other in all_actor_trajectories:
                    if other.agent_id not in zone['priority_actor_ids']:
                        continue
                    ix = np.flatnonzero(other.valid_mask)
                    other_p, support = sample(other.timestamps[ix], other.positions[ix], t, cfg['max_gap_s'])
                    own_inside = np.array([zone['polygon'].covers(Point(x)) for x in p])
                    priority_inside = np.array([zone['polygon'].covers(Point(x)) for x in other_p])
                    observed |= valid & support
                    flags |= valid & support & own_inside & priority_inside & moving
            put(name, flags, observed, definition='simultaneous_conflict_zone_occupation_despite_explicit_priority', zone_count=len(zones))
    profiles = {}
    absent = set(name for name, regions in context.regions.items() if regions == [])
    if context.solid_lines == []:
        absent.add('solid_line_crossing')
    if context.signals == []:
        absent.add('red_light')
    if context.stop_signs == []:
        absent.update(('stop_sign', 'stop_line_encroachment'))
    for name in absent:
        rules[name].update(applicable=False, observable=False, status='not_applicable',
            reason='relevant_map_element_observed_absent', rule_score=None, violated=None,
            violation_count=None, duration=None, violation_ratio=None, severity=None, coverage=0.)
    applicable_count = sum(r['applicable'] for r in rules.values())
    for profile, weights in [('normalized', cfg.get('rule_weights', {})), ('onsite_compat', ONSITE_WEIGHTS)]:
        eligible = [(name, r) for name, r in rules.items() if r['applicable'] and r['observable']
                    and (profile != 'onsite_compat' or name in weights)]
        total = sum(float(weights.get(name, 1.)) for name, _ in eligible)
        if any(float(weights.get(name, 1.)) < 0 for name, _ in eligible):
            raise ValueError('rule weights must be nonnegative')
        score = sum(float(weights.get(name, 1.))*r['rule_score'] for name, r in eligible)/total if total > 0 else None
        profiles[profile] = dict(traffic_compliance_score=score,
            traffic_compliance_coverage=sum(r['coverage'] for _, r in eligible)/applicable_count if applicable_count else 0.,
            num_applicable_rules=applicable_count, num_observable_rules=len(eligible),
            status='valid' if score is not None else 'unavailable',
            reason=None if score is not None else 'no_observable_weighted_rules',
            definition='weighted_observable_rule_compliance; continuous rules use time ratios, discrete rules use event violations; OnSite weights only, not original overall score')
    return dict(scene_id=track.scene_id, actor_id=actor_id, actor_type=track.agent_type,
                status=profiles['normalized']['status'], profiles=profiles, rules=rules,
                map_provenance=context.provenance)


def aggregate_actors(actors):
    valid = [x for x in actors if x['profiles']['normalized']['traffic_compliance_score'] is not None]
    values = [x['profiles']['normalized']['traffic_compliance_score'] for x in valid]
    worst = min(valid, key=lambda x: x['profiles']['normalized']['traffic_compliance_score']) if valid else None
    rules = {}
    for name in RULES:
        observable = [x['rules'][name] for x in actors if x['rules'][name]['observable']]
        rules[name] = dict(observable_actor_count=len(observable),
            violation_count=sum(r['violation_count'] for r in observable) if observable else None,
            violation_duration=sum(r['duration'] for r in observable) if observable else None,
            violation_actor_ratio=float(np.mean([r['violated'] for r in observable])) if observable else None)
    return dict(actor_count=len(actors), scorable_actor_count=len(valid),
                compliance_mean=float(np.mean(values)) if values else None,
                compliance_median=float(np.median(values)) if values else None,
                compliance_p10=float(np.percentile(values, 10)) if values else None,
                compliance_min=min(values) if values else None,
                violation_actor_ratio=float(np.mean([any(r['violated'] for r in x['rules'].values()) for x in valid])) if valid else None,
                worst_actor_id=worst['actor_id'] if worst else None,
                worst_actor_scene_id=worst['scene_id'] if worst else None,
                worst_compliance_score=min(values) if values else None, rules=rules)


def evaluate_scene_compliance(tracks, map_context, config=None):
    config = config or {}
    actors = [evaluate_actor_compliance(t.agent_id, t, tracks, map_context, config)
              for t in tracks if t.agent_type in ('ego', 'vehicle')
              and config.get('evaluate_ego' if t.agent_type == 'ego' else 'evaluate_non_ego', True)
              and np.any(t.valid_mask[int(t.metadata.get('history_frames', 0)):])]
    return dict(scene_id=tracks[0].scene_id if tracks else None, actors=actors,
                ego_traffic_compliance=aggregate_actors([x for x in actors if x['actor_type'] == 'ego']),
                non_ego_traffic_compliance=aggregate_actors([x for x in actors if x['actor_type'] != 'ego']),
                all_actor_traffic_compliance=aggregate_actors(actors))


def load_map_context(scene, native_config=None):
    """Missing spatial support disables map rules without disabling other rules."""
    from driveaudit_eval.metrics.capabilities import map_capabilities, MAP_REASON
    from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
    if not map_capabilities(scene)['drivable_area']:
        return MapContext(provenance=dict(status='unavailable',reason=MAP_REASON))
    provider=InputGTMapProvider(scene,native_config)
    return MapContext(provider=provider,provenance=provider.query_metadata())

"""Continuous distance attribution on executed states; never changes termination."""
from copy import deepcopy
import numpy as np

DEFAULTS = dict(enabled=False, max_lateral_distance_m=2.5, endpoint_tolerance_m=.5,
    local_arc_window_m=10., minimum_projection_coverage=.8,
    continuity=dict(enabled=True, forward_jump_scale=1.5, forward_jump_buffer_m=.05, backward_tolerance_m=2.))


def settings(config=None):
    cfg=deepcopy(DEFAULTS)
    for key,value in (config or {}).items():
        if isinstance(value,dict) and isinstance(cfg.get(key),dict):cfg[key].update(value)
        else:cfg[key]=value
    for key in ('max_lateral_distance_m','endpoint_tolerance_m','local_arc_window_m'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('Invalid route projection parameter: '+key)
    if not 0<=cfg['minimum_projection_coverage']<=1:raise ValueError('Invalid projection coverage threshold')
    for key in ('forward_jump_scale','forward_jump_buffer_m','backward_tolerance_m'):
        if not np.isfinite(cfg['continuity'][key]) or cfg['continuity'][key]<0:raise ValueError('Invalid route continuity parameter')
    return cfg


def prepare_route(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Invalid route polyline')
    # Remove consecutive duplicate vertices, preserving the route geometry.
    points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10]]
    if len(points)<2:raise ValueError('Zero-length route')
    delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1)
    return dict(points=points,delta=delta,length=length,s=np.r_[0.,np.cumsum(length)])


def project_to_route_continuous(position,route,previous_projection=None,previous_position=None,dt=None,config=None):
    """Prefer nearby route segments; fallback still obeys all distance/jump gates.

    previous_projection/position refer to the last accepted state. Invalid states
    never advance this anchor. dt is diagnostic; the gate bounds arc displacement
    by observed XY displacement, not a fabricated velocity from logged resets.
    """
    cfg=settings(config);p=np.asarray(position,float)
    if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Invalid projection position')
    if dt is not None and (not np.isfinite(dt) or dt<0):raise ValueError('Invalid projection time difference')
    a=route['points'][:-1];delta=route['delta'];length=route['length']
    raw=np.sum((p-a)*delta,axis=1)/(length*length);fraction=np.clip(raw,0.,1.)
    projected=a+fraction[:,None]*delta;distance=np.linalg.norm(p-projected,axis=1)
    arc=route['s'][:-1]+fraction*length
    reasons=np.full(len(length),'valid',dtype=object)
    reasons[distance>cfg['max_lateral_distance_m']]='lateral_distance_exceeded'
    if raw[0]<0 and np.linalg.norm(p-route['points'][0])>cfg['endpoint_tolerance_m']:reasons[0]='start_clamp_outside_tolerance'
    if raw[-1]>1 and np.linalg.norm(p-route['points'][-1])>cfg['endpoint_tolerance_m']:reasons[-1]='endpoint_clamp_outside_goal_tolerance'
    prev=previous_projection if previous_projection and previous_projection.get('projection_valid') else None
    indices=np.arange(len(length));local=indices
    if prev is not None:
        s=float(prev['route_s']);window=cfg['local_arc_window_m']
        local=indices[(route['s'][:-1]<=s+window)&(route['s'][1:]>=s-window)]
        if cfg['continuity']['enabled']:
            if previous_position is None:raise ValueError('Continuity requires last accepted position')
            motion=float(np.linalg.norm(p-np.asarray(previous_position,float)))
            forward=cfg['continuity']['forward_jump_scale']*motion+cfg['continuity']['forward_jump_buffer_m']
            jump=(arc-s>forward+1e-9)|(arc-s < -cfg['continuity']['backward_tolerance_m']-1e-9)
            reasons[(reasons=='valid')&jump]='route_continuity_jump_exceeded'
    def choose(candidates):
        if not len(candidates):return None
        # Gate first, then nearest valid candidate, with arc continuity as tie-break.
        nearest=min(candidates,key=lambda i:distance[i])
        if reasons[nearest] in ('endpoint_clamp_outside_goal_tolerance','start_clamp_outside_tolerance'):return nearest
        valid=candidates[reasons[candidates]=='valid']
        pool=valid if len(valid) else candidates
        return min(pool,key=lambda i:(distance[i],abs(arc[i]-prev['route_s']) if prev else arc[i]))
    index=choose(local);fallback=False
    if index is None or reasons[index]!='valid':
        fallback=prev is not None
        index=choose(indices)
    valid=bool(reasons[index]=='valid')
    return dict(segment_index=int(index),segment_fraction=float(fraction[index]),
        segment_fraction_raw=float(raw[index]),segment_fraction_clamped=float(fraction[index]),
        route_s=float(arc[index]),projection_point=projected[index].tolist(),lateral_distance=float(distance[index]),
        projection_valid=valid,projection_reason=str(reasons[index]),projection_fallback_used=fallback)


def correct_executed_route(result,points,config):
    """Replace route-distance aliases only, on existing executed samples in order."""
    cfg=settings(config);episode=result['episode'];rows=result['route_timeseries']
    prepared=prepare_route(points) if points is not None and rows else None
    previous=None;previous_position=None;previous_time=None;start=None;last=None;maximum=None;valid_count=0;by_frame={};terminal=None
    for index,row in enumerate(rows):
        p=[row['x'],row['y']]
        projection=project_to_route_continuous(p,prepared,previous,previous_position,
            row['timestamp']-previous_time if previous_time is not None else None,cfg)
        if index==0 and projection['projection_valid']:start=projection['route_s'];last=start;maximum=start
        if projection['projection_valid']:
            valid_count+=1;previous=projection;previous_position=p;previous_time=row['timestamp']
            if start is not None:last=projection['route_s'];maximum=max(maximum,last)
        total=prepared['s'][-1]-start if start is not None else None
        ratio=float(np.clip((maximum-start)/total,0,1)) if total is not None and total>0 else None
        row.update(unconstrained_route_s=row['route_s'],route_s=last,route_progress_ratio=ratio,
            route_completion_ratio=ratio,route_projection=projection,route_projection_valid=projection['projection_valid'])
        by_frame.setdefault(row['decision_frame_id'],[]).append(row);terminal=projection
    total=prepared['s'][-1]-start if prepared is not None and start is not None else None
    available=total is not None and total>0
    ratio=float(np.clip((maximum-start)/total,0,1)) if available else None
    coverage=valid_count/len(rows) if rows else 0.
    success=int(bool(episode['goal_reached']) and not episode['collision'])
    episode.update(unconstrained_route_progress_ratio=episode['route_progress_ratio'],
        route_completion_ratio=ratio,route_progress_ratio=ratio,proxy_route_progress_ratio=ratio,
        route_progress_distance=maximum-start if available else None,completed_route_distance=maximum-start if available else None,
        max_route_progress=maximum-start if available else None,net_route_progress=last-start if available else None,
        route_total_distance=total,total_route_distance=total,s_start=start,s_current=last,episode_start_route_s=start,
        task_success=success,task_outcome='SUCCESS' if success else 'COLLISION_FAILURE' if episode['collision'] else 'INCOMPLETE',
        safe_route_completion_ratio=0. if episode['collision'] else ratio,
        valid_route_projection_frames=valid_count,total_executed_route_frames=len(rows),route_projection_coverage=coverage,
        route_projection_invalid_count=len(rows)-valid_count,
        route_completion_status='unavailable' if not available else 'low_projection_coverage' if coverage<cfg['minimum_projection_coverage'] else 'valid',
        terminal_position_x=rows[-1]['x'] if rows else None,terminal_position_y=rows[-1]['y'] if rows else None,
        terminal_route_s=last,terminal_route_projection_x=terminal['projection_point'][0] if terminal else None,
        terminal_route_projection_y=terminal['projection_point'][1] if terminal else None,
        terminal_route_projection_distance=terminal['lateral_distance'] if terminal else None,
        terminal_endpoint_distance=float(np.linalg.norm(np.array([rows[-1]['x'],rows[-1]['y']])-prepared['points'][-1])) if rows else None,
        terminal_projection_segment_index=terminal['segment_index'] if terminal else None,
        terminal_projection_fraction_raw=terminal['segment_fraction_raw'] if terminal else None,
        terminal_projection_fraction_clamped=terminal['segment_fraction_clamped'] if terminal else None,
        terminal_projection_valid=terminal['projection_valid'] if terminal else False,
        terminal_projection_reason=terminal['projection_reason'] if terminal else 'route_unavailable',
        route_projection_config=cfg)
    last_ratio=None;last_s=start
    for window in result['windows']:
        states=by_frame.get(window['decision_frame_id'],[])
        if states:
            before=last_ratio
            window.update(route_s_before=states[0]['route_s'],route_s_after=states[-1]['route_s'],route_progress_ratio_after=states[-1]['route_completion_ratio'])
            last_ratio=states[-1]['route_completion_ratio'];last_s=states[-1]['route_s']
            if window['decision_frame_id']==episode['terminal_decision_frame']:
                episode.update(route_progress_before_terminal_commit=before,route_progress_after_terminal_commit=last_ratio,
                    progress_gained_in_terminal_commit=last_ratio-before if last_ratio is not None and before is not None else None)
        else:window.update(route_s_before=last_s,route_s_after=last_s,route_progress_ratio_after=last_ratio)
        window['route_completion_ratio_after']=window['route_progress_ratio_after']
    return result

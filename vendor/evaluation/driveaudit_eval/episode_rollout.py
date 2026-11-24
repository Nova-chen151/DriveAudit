"""Logged-state execution-prefix episodes, not state-feedback closed-loop simulation.

Every plan retains its logged decision anchor. Prefixes are separate motion
segments: pose resets are audited and never counted as executed distance. Other
actors are logged background and do not react. Responsibility algorithms remain
frozen; only the terminal event and its observed pre-event inputs are changed.
"""
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from driveaudit_eval.agents.core import EGO_LENGTH, EGO_WIDTH, ego_to_world, line_progress
from driveaudit_eval.agents.scoring import overlap
from driveaudit_eval.agents.input_gt import load_scene as load_logged, sample
from driveaudit_eval.utils.map_provider.geometry import trajectory_headings
from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
from driveaudit_eval.traffic_compliance import MapContext, evaluate_actor_compliance
from driveaudit_eval.traffic_evaluator import ego_track
from driveaudit_eval.trajectory.input_gt import load_scene as load_tracks

DEFAULTS = dict(enabled=True, execution_mode='logged_state_prefix',
    terminal_execution_mode='nominal_prefix', nominal_replan_interval_s=.1, collision_check_dt_s=.01,
    time_refinement_tolerance_s=.0001, max_prediction_gap_s=.5,
    max_actor_gap_s=.25, terminate_on_offroad=False,
    offroad_occupancy_threshold=.2, timeout_s=None,
    goal=dict(longitudinal_tolerance_m=.1, euclidean_tolerance_m=.5),
    route=dict(source='logged_ego_path_proxy'), experimental_score=dict(enabled=False))


def _config(config):
    cfg=deepcopy(DEFAULTS)
    for key,value in (config or {}).items():
        if isinstance(value,dict) and isinstance(cfg.get(key),dict): cfg[key].update(value)
        else: cfg[key]=value
    if cfg['execution_mode']!='logged_state_prefix':
        raise ValueError('Only logged_state_prefix is supported; reanchored_proxy is not implemented')
    if cfg['terminal_execution_mode'] not in ('nominal_prefix','horizon_commit'):
        raise ValueError('Unsupported terminal_execution_mode')
    for key in ('nominal_replan_interval_s','collision_check_dt_s','time_refinement_tolerance_s','max_prediction_gap_s','max_actor_gap_s'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError('Invalid positive episode time parameter: '+key)
    if cfg['timeout_s'] is not None and (not np.isfinite(cfg['timeout_s']) or cfg['timeout_s']<=0): raise ValueError('Invalid episode timeout')
    if any(not np.isfinite(v) or v<0 for v in cfg['goal'].values()): raise ValueError('Invalid goal tolerance')
    if not 0<=cfg['offroad_occupancy_threshold']<=1: raise ValueError('Invalid offroad threshold')
    if cfg['experimental_score'].get('enabled'): raise ValueError('Experimental episode composite score is not implemented')
    return cfg


def _jsonable(value):
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(np.ndarray,list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,np.bool_): return bool(value)
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def _plan(window):
    times=np.asarray(window['timestamps'],float)
    p=np.asarray(window['positions'],float); yaw=np.asarray(window['headings'],float)
    if len(times)<2 or p.shape!=(len(times),2) or yaw.shape!=(len(times),) or np.any(np.diff(times)<=0):
        raise ValueError('Invalid plan shape or timestamp order')
    values=np.column_stack([p,np.unwrap(yaw)])
    if not np.isfinite(times).all() or not np.isfinite(values).all(): raise ValueError('Nonfinite plan state')
    if not np.isclose(times[0],float(window['decision_time']),rtol=0,atol=1e-8): raise ValueError('Plan lacks logged decision anchor')
    return times,values


def _query(window,query,cfg):
    times,values=_plan(window)
    return sample(times,values,np.asarray(query,float),cfg['max_prediction_gap_s'])


def _grid(start,end,dt,knots=()):
    if end<=start: return np.array([start])
    regular=np.linspace(start,end,int(np.ceil((end-start)/dt))+1)
    return np.unique(np.r_[regular,np.asarray(knots)[(np.asarray(knots)>=start)&(np.asarray(knots)<=end)]])


def _collision(window, actors, start, end, cfg):
    """Same SAT boxes as original agent metric, dense support plus contact bisection.

    Dense sampling bounds the reported temporal search resolution; it is not an
    analytic certificate against contacts shorter than collision_check_dt_s.
    """
    grid=_grid(start,end,cfg['collision_check_dt_s'],window['timestamps'])
    ego,valid=_query(window,grid,cfg)
    if not valid.all(): return None,dict(status='invalid',reason='execution_prefix_time_support_gap')
    first=None; observed_count=0
    for actor,data in actors.items():
        data=np.asarray(data,float)
        if data.ndim!=2 or data.shape[1]!=6 or len(data)==0: raise ValueError('Actor must provide t,x,y,yaw,length,width')
        if not np.isfinite(data).all() or np.any(data[:,4:6]<=0): raise ValueError('Invalid native actor geometry')
        other,support=sample(data[:,0],data[:,1:],grid,cfg['max_actor_gap_s'])
        observed_count+=int(support.any())
        hit=support&overlap(ego[:,:2],ego[:,2],other[:,:2],other[:,2],other[:,3],other[:,4])
        found=np.flatnonzero(hit)
        if not len(found): continue
        index=int(found[0]); upper=float(grid[index]); lower=float(grid[max(0,index-1)])
        # Refine only supported non-contact -> contact brackets; no extrapolation.
        if index and support[index-1] and not hit[index-1]:
            while upper-lower>cfg['time_refinement_tolerance_s']:
                mid=(upper+lower)/2
                ep,ev=_query(window,[mid],cfg); op,ov=sample(data[:,0],data[:,1:],[mid],cfg['max_actor_gap_s'])
                if not ev[0] or not ov[0]: break
                touch=bool(overlap(ep[:,:2],ep[:,2],op[:,:2],op[:,2],op[:,3],op[:,4])[0])
                if touch: upper=mid
                else: lower=mid
        candidate=dict(time=upper,collision_pair=['ego',str(actor)],collision_time_bracket=[lower,upper],
            collision_geometry='original_agent_SAT_no_core_inset_native_sampled_dimensions',
            collision_check_dt_s=cfg['collision_check_dt_s'])
        if first is None or (candidate['time'],str(actor))<(first['time'],first['collision_pair'][1]): first=candidate
    return first,dict(status='valid',observed_actor_count=observed_count,search_samples=len(grid))



def _first_event_time(grid, flags, predicate, tolerance):
    indices=np.flatnonzero(flags)
    if not len(indices): return None
    index=int(indices[0]);upper=float(grid[index]);lower=float(grid[max(0,index-1)])
    if index:
        while upper-lower>tolerance:
            mid=(upper+lower)/2
            if predicate(mid): upper=mid
            else: lower=mid
    return upper

def _route_s(route,point):
    return float(line_progress(route,route[0],np.asarray(point)))


def _compliance(scene, times, values, context, cfg):
    if len(times)<2 or context is None: return None
    track=ego_track(scene,np.asarray(times),np.asarray(values),0,'generated')
    result=evaluate_actor_compliance('ego',track,[track],context,dict(cfg.get('traffic_config',{}),max_gap_s=cfg['max_prediction_gap_s']))
    return result


def run_episode(scene_id, agent_name, windows, actors, route=None, config=None,
                map_context=None, terminal_callback=None, scene_end_time=None, route_provenance=None):
    """Execute sorted absolute-time plans; suitable for both synthetic and saved data.

    Windows contain decision_frame_id, decision_time, timestamps (including the
    logged anchor), positions, headings and optional original collision metadata.
    terminal_callback(event, original_window, executed_history) is invoked once.
    """
    cfg=_config(config); ordered=sorted(windows,key=lambda w:float(w['decision_time']))
    if not ordered: raise ValueError('Episode has no common valid decision windows')
    decisions=np.asarray([w['decision_time'] for w in ordered],float)
    if not np.isfinite(decisions).all() or np.any(np.diff(decisions)<=0): raise ValueError('Duplicate/nonfinite episode decision timestamps')
    ident=f'{scene_id}/{agent_name}'; start=float(decisions[0]); start_frame=ordered[0]['decision_frame_id']
    if route is not None:
        route=np.asarray(route,float)
        if route.ndim!=2 or route.shape[1]!=2 or len(route)<2 or not np.isfinite(route).all(): raise ValueError('Invalid route polyline')
        if np.linalg.norm(np.diff(route,axis=0),axis=1).sum()<=0: route=None
    try: origin=_plan(ordered[0])[1][0,:2]
    except (ValueError,KeyError,IndexError): origin=None
    s_start=_route_s(route,origin) if route is not None and origin is not None else None
    s_goal=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()) if route is not None else None
    total=s_goal-s_start if s_goal is not None and s_start is not None else None
    route_ok=total is not None and total>0
    if not route_ok: route=None
    max_progress=0.; last_s=s_start; rows=[]; route_rows=[]; executed=[]; traffic=[]
    done=None; end=start; last_frame=start_frame; executed_distance=0.; executed_time=0.; resets=[]; prior_endpoint=None
    terminal=[]; offroad=False
    commit=cfg['terminal_execution_mode']=='horizon_commit'
    tail_started=False; tail_duration=0.; progress_before=None
    for index,window in enumerate(ordered):
        decision=float(window['decision_time']); frame=window['decision_frame_id']
        pred_end=float(window['timestamps'][-1]); requested=float(decisions[index+1]-decision) if index+1<len(ordered) else cfg['nominal_replan_interval_s']
        is_tail=index==len(ordered)-1
        if commit and is_tail: requested=pred_end-decision
        prefix_end=min(decision+requested,pred_end,float(scene_end_time) if scene_end_time is not None else np.inf)
        open_time=window.get('open_loop_first_collision_time')
        row=dict(episode_id=ident,scene_id=scene_id,agent_name=agent_name,decision_frame_id=frame,decision_time=decision,
            window_execution_mode='terminal_horizon_commit' if commit and is_tail else 'replan_prefix',
            prediction_horizon_s=pred_end-decision,execution_horizon_s=max(0.,prefix_end-decision),
            open_loop_collision=bool(window.get('open_loop_collision',False)),open_loop_first_collision_time=open_time,
            executed_prefix_collision=False,executed_prefix_first_collision_time=None,
            window_execution_status='executed',episode_execution_status='executed',episode_done_after_window=False,
            route_s_before=last_s,route_s_after=last_s,route_progress_ratio_after=float(np.clip(max_progress/total,0,1)) if route_ok else None,
            executed_prefix_duration=0.,interpolation_used=False,collision_beyond_execution_prefix=False)
        rows.append(row)
        if done is not None:
            suffix='goal' if done['reason']=='goal_reached' else done['reason']
            row.update(window_execution_status='not_executed_after_'+suffix,episode_execution_status='not_executed_after_'+suffix,
                       episode_done_after_window=True)
            continue
        if is_tail:
            tail_started=True
            progress_before=float(np.clip(max_progress/total,0,1)) if route_ok else None
        candidates=[]
        if index+1<len(ordered) and pred_end+1e-8<decisions[index+1] and (scene_end_time is None or pred_end<scene_end_time):
            candidates.append(dict(time=decision,reason='invalid_state',detail='prediction_does_not_cover_next_replan'))
            states=np.empty((0,3)); grid=np.empty(0)
        elif prefix_end<=decision:
            candidates.append(dict(time=decision,reason='invalid_state',detail='nonpositive_execution_prefix'))
            states=np.empty((0,3)); grid=np.empty(0)
        else:
            try:
                collision,search=_collision(window,actors,decision,prefix_end,cfg)
                grid=_grid(decision,prefix_end,cfg['collision_check_dt_s'],window['timestamps'])
                states,support=_query(window,grid,cfg)
                if not support.all() or search['status']!='valid': candidates.append(dict(time=decision,reason='invalid_state',detail='unsupported_execution_state'))
                if collision is not None: candidates.append(dict(collision,reason='collision'))
                if route_ok:
                    route_values=np.array([_route_s(route,p) for p in states[:,:2]])
                    goal=(s_goal-route_values<=cfg['goal']['longitudinal_tolerance_m'])&(np.linalg.norm(states[:,:2]-route[-1],axis=1)<=cfg['goal']['euclidean_tolerance_m'])
                    if goal.any():
                        def goal_at(stamp):
                            state,valid=_query(window,[stamp],cfg)
                            return bool(valid[0] and s_goal-_route_s(route,state[0,:2])<=cfg['goal']['longitudinal_tolerance_m'] and np.linalg.norm(state[0,:2]-route[-1])<=cfg['goal']['euclidean_tolerance_m'])
                        goal_time=_first_event_time(grid,goal,goal_at,cfg['time_refinement_tolerance_s'])
                        candidates.append(dict(time=goal_time,reason='goal_reached'))
                if cfg['terminate_on_offroad'] and map_context is not None and map_context.provider is not None:
                    occupancies=[]
                    for p,yaw in zip(states[:,:2],states[:,2]):
                        try: occupancies.append(float(map_context.provider.footprint_coverage(p[None],np.array([yaw]),EGO_LENGTH,EGO_WIDTH)[0]))
                        except ValueError: occupancies.append(np.nan)
                    bad=np.isfinite(occupancies)&(np.asarray(occupancies)<cfg['offroad_occupancy_threshold'])
                    if cfg['terminate_on_offroad'] and bad.any():
                        def offroad_at(stamp):
                            state,valid=_query(window,[stamp],cfg)
                            if not valid[0]: return False
                            try: occupancy=map_context.provider.footprint_coverage(state[:,:2],state[:,2],EGO_LENGTH,EGO_WIDTH)[0]
                            except ValueError: return False
                            return bool(np.isfinite(occupancy) and occupancy<cfg['offroad_occupancy_threshold'])
                        offroad_time=_first_event_time(grid,bad,offroad_at,cfg['time_refinement_tolerance_s'])
                        candidates.append(dict(time=offroad_time,reason='offroad'))
            except (ValueError,KeyError,IndexError) as exc:
                candidates.append(dict(time=decision,reason='invalid_state',detail=str(exc))); grid=np.empty(0);states=np.empty((0,3))
        deadline=start+cfg['timeout_s'] if cfg['timeout_s'] is not None else np.inf
        if deadline<=prefix_end: candidates.append(dict(time=max(decision,deadline),reason='timeout'))
        if scene_end_time is not None and scene_end_time<=prefix_end:
            candidates.append(dict(time=float(scene_end_time),reason='scene_end' if commit else 'log_end'))
        if is_tail and not (commit and scene_end_time is not None and scene_end_time<=prefix_end):
            candidates.append(dict(time=prefix_end,reason='log_end',**({'detail':'final_plan_support_exhausted_before_scene_end'} if commit else {})))
        # Event chronology is primary; collision wins exact simultaneous ties.
        priority={'collision':0,'offroad':1,'goal_reached':2,'timeout':3,'log_end':4,'scene_end':4,'invalid_state':5}
        earliest=min((c['time'] for c in candidates),default=None)
        simultaneous=[c for c in candidates if c['time']<=earliest+cfg['time_refinement_tolerance_s']] if earliest is not None else []
        chosen=min(simultaneous,key=lambda x:(priority[x['reason']],x['time'])) if simultaneous else None
        stop=float(chosen['time']) if chosen else prefix_end
        if chosen and chosen['reason']=='invalid_state':
            row.update(window_execution_status='invalid',episode_execution_status='invalid')
        else:
            query=_grid(decision,stop,cfg['collision_check_dt_s'],window['timestamps'])
            value,valid=_query(window,query,cfg)
            if not valid.all(): raise ValueError('Terminal interpolation lost plan support')
            if prior_endpoint is not None: resets.append(dict(decision_time=decision,position_reset_m=float(np.linalg.norm(value[0,:2]-prior_endpoint))))
            prior_endpoint=value[-1,:2].copy()
            distance=float(np.linalg.norm(np.diff(value[:,:2],axis=0),axis=1).sum())
            executed_distance+=distance; executed_time+=max(0.,stop-decision)
            segment=dict(decision_frame_id=frame,timestamps=query.tolist(),values=value.tolist())
            executed.append(segment)
            row.update(executed_prefix_duration=max(0.,stop-decision),executed_distance=distance,
                       interpolation_used=not bool(np.any(np.isclose(window['timestamps'],stop,atol=1e-8,rtol=0))))
            if route_ok:
                row['route_s_before']=_route_s(route,value[0,:2])
                for stamp,state in zip(query,value):
                    current=_route_s(route,state[:2]); max_progress=max(max_progress,current-s_start)
                    route_rows.append(dict(episode_id=ident,scene_id=scene_id,agent_name=agent_name,decision_frame_id=frame,
                        timestamp=float(stamp),route_s=current,route_progress_ratio=float(np.clip(max_progress/total,0,1)),
                        x=float(state[0]),y=float(state[1]),segment_kind='executed_prefix'))
                last_s=_route_s(route,value[-1,:2])
            compliance=_compliance(scene_id,query,value,map_context,cfg)
            if compliance is not None:
                traffic.append(compliance)
                offroad|=bool(compliance['rules'].get('offroad',{}).get('violated'))
            row.update(route_s_after=last_s,route_progress_ratio_after=float(np.clip(max_progress/total,0,1)) if route_ok else None)
        if is_tail: tail_duration=max(0.,stop-decision) if row['window_execution_status']=='executed' else 0.
        end=stop; last_frame=frame
        if chosen:
            done=chosen; row['episode_done_after_window']=True
            if chosen['reason']=='collision':
                row.update(executed_prefix_collision=True,executed_prefix_first_collision_time=stop)
                event=dict(chosen,episode_id=ident,scene_id=scene_id,agent=agent_name,agent_name=agent_name,
                    decision_frame_id=frame,first_collision_time=stop,
                    collision_event_id=f'{ident}/terminal/{frame}',first_collision_frame=frame,
                    first_collision_frame_domain='terminal_decision_frame; native nearest frame supplied by dataset integration')
                if terminal_callback is not None:
                    result=terminal_callback(event,window,executed)
                    if result is not None: terminal.append(result)
                else: terminal.append(dict(event,status='unavailable',reason='terminal_responsibility_callback_not_provided'))
        row['collision_beyond_execution_prefix']=bool(row['open_loop_collision'] and open_time is not None and open_time>stop and not row['executed_prefix_collision'])
        row['future_horizon_collision_not_executed']=row['collision_beyond_execution_prefix']
    reason=done['reason'] if done else 'log_end'; collided=reason=='collision'; goal=reason=='goal_reached'
    rule_names=sorted({name for result in traffic for name in result['rules']})
    aggregated_rules={}
    for name in rule_names:
        observed=[r['rules'][name] for r in traffic if r['rules'][name].get('observable')]
        exposure=sum(r['exposure_duration'] for r in observed); duration=sum(r['duration'] or 0. for r in observed)
        aggregated_rules[name]=dict(exposure_duration=exposure,violation_duration=duration,
            severity=(max(1-r['rule_score'] for r in observed) if name in ('red_light','stop_sign','illegal_uturn','illegal_overtaking') else duration/exposure) if exposure else (max(1-r['rule_score'] for r in observed) if observed else None))
    valid_rules=[r for r in aggregated_rules.values() if r['severity'] is not None]
    rule_weights=cfg.get('traffic_config',{}).get('rule_weights',{})
    weighted=[(float(rule_weights.get(name,1.)),r) for name,r in aggregated_rules.items() if r['severity'] is not None]
    if any(w<0 or not np.isfinite(w) for w,_ in weighted): raise ValueError('Invalid shared traffic rule weight')
    weight_sum=sum(w for w,_ in weighted)
    compliance_score=sum(w*(1-r['severity']) for w,r in weighted)/weight_sum if weight_sum else None
    coverage=sum(min(1.,r['exposure_duration']/executed_time) for r in valid_rules)/len(rule_names) if executed_time and rule_names else 0.
    responsibility=terminal[0] if terminal else {}
    episode=dict(episode_id=ident,scene_id=scene_id,agent_name=agent_name,start_frame=start_frame,start_time=start,
        end_frame=last_frame,end_frame_domain='last_executed_decision_frame',end_time=end,duration=end-start,episode_duration=end-start,episode_start_route_s=s_start,
        termination_reason=reason,status='invalid' if reason=='invalid_state' else 'valid',termination_detail=done.get('detail') if done else None,
        collision=collided,collision_time=end if collided else None,collision_pair=done.get('collision_pair') if collided else None,
        time_to_collision=end-start if collided else None,distance_before_collision=executed_distance if collided else None,
        goal_reached=goal,route_available=route_ok,route_status='proxy' if route_ok and cfg['route']['source']=='logged_ego_path_proxy' else 'valid' if route_ok else 'unavailable',
        route_kind=cfg['route']['source'] if route_ok else 'unavailable',route_provenance=route_provenance or {},
        goal_kind='log_endpoint_proxy' if cfg['route']['source']=='logged_ego_path_proxy' else 'reference_route_endpoint',
        route_progress_distance=max_progress if route_ok else None,net_route_progress=last_s-s_start if route_ok else None,
        max_route_progress=max_progress if route_ok else None,route_total_distance=total if route_ok else None,
        route_progress_ratio=float(np.clip(max_progress/total,0,1)) if route_ok else None,
        s_start=s_start,s_current=last_s,s_goal=s_goal,
        route_completed=int(goal),executed_decision_count=len(executed),available_decision_count=len(ordered),
        executed_time=executed_time,executed_distance=executed_distance,offroad_before_termination=offroad,
        terminal_ego_responsibility_ratio=responsibility.get('ego_responsibility_ratio'),
        terminal_other_responsibility_ratio=responsibility.get('other_responsibility_ratio'),
        terminal_dominant_actor=responsibility.get('dominant_actor'),
        terminal_ego_responsibility_class=responsibility.get('ego_responsibility_class'),
        terminal_responsibility_dominant=responsibility.get('ego_responsibility_class',responsibility.get('dominant_actor')),
        terminal_dominance_strength=responsibility.get('responsibility_dominance_strength'),
        terminal_responsibility_confidence=responsibility.get('responsibility_confidence'),
        terminal_collision_type=responsibility.get('collision_type'),
        terminal_collision_time=end if collided else None,terminal_collision_pair=done.get('collision_pair') if collided else None,
        terminal_collision_frame=responsibility.get('collision_frame',responsibility.get('first_collision_frame',last_frame)) if collided else None,
        terminal_collision_frame_domain='nearest_native_scene_frame' if responsibility.get('collision_frame') is not None else 'terminal_decision_frame',
        traffic_compliance_score=compliance_score,traffic_compliance_coverage=coverage,
        episode_ego_traffic_compliance=dict(score=compliance_score,coverage=coverage,rules=aggregated_rules),
        pose_resets=resets,execution_mode='logged_state_prefix',is_true_closed_loop=False)
    final_window=ordered[-1]
    available_duration=scene_end_time-start if scene_end_time is not None else None
    episode.update(terminal_execution_mode=cfg['terminal_execution_mode'],terminal_commit_enabled=commit,
        terminal_commit_executed=bool(commit and tail_started),
        terminal_decision_frame=final_window['decision_frame_id'],terminal_decision_time=float(final_window['decision_time']),
        native_scene_end_time=scene_end_time,terminal_plan_horizon_s=float(final_window['timestamps'][-1]-final_window['decision_time']),
        terminal_scene_remaining_s=max(0.,scene_end_time-final_window['decision_time']) if scene_end_time is not None else None,
        terminal_executed_duration_s=tail_duration,terminal_collision=bool(collided and tail_started),
        terminal_collision_offset_s=end-final_window['decision_time'] if collided and tail_started else None,
        terminal_goal_reached=bool(goal and tail_started),route_progress_before_terminal_commit=progress_before,
        route_progress_after_terminal_commit=episode['route_progress_ratio'] if tail_started else None,
        progress_gained_in_terminal_commit=episode['route_progress_ratio']-progress_before if tail_started and route_ok else None,
        temporal_episode_coverage=executed_time/available_duration if available_duration is not None and available_duration>0 else None,
        proxy_route_progress_ratio=episode['route_progress_ratio'] if cfg['route']['source']=='logged_ego_path_proxy' else None,
        proxy_route_completed=int(goal) if cfg['route']['source']=='logged_ego_path_proxy' else None,
        route_completed_proxy=int(goal) if cfg['route']['source']=='logged_ego_path_proxy' else None)
    if commit:
        episode['termination_reason']=('goal_reached_proxy' if goal and cfg['route']['source']=='logged_ego_path_proxy'
            else 'invalid' if reason=='invalid_state' else reason)
    result=dict(episode=episode,windows=rows,terminal_responsibility=terminal,route_timeseries=route_rows)
    if cfg.get('route_projection',{}).get('enabled',False):
        from driveaudit_eval.route_projection import correct_executed_route
        correct_executed_route(result,route,cfg['route_projection'])
    return _jsonable(result)


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()



def evaluate_terminal_responsibility(event, window, executed_history, scene, times, values, tracks, raw_tracks, map_context, config, context, timestamps):
    """Reuse frozen V1 -> V2 -> V3 on only the terminal executed prefix."""
    from driveaudit_eval.responsibility.evidence import evaluate_event
    from driveaudit_eval.responsibility.attribution import classify
    from driveaudit_eval.responsibility.cars_adapter import evaluate_collision_responsibility
    cfg=_config(config.get('episode_rollout',{}))
    tcfg=config['evaluation']['trajectory']
    end=event['first_collision_time']; decision=window['decision_time']
    prefix=np.asarray(executed_history[-1]['timestamps']); state=np.asarray(executed_history[-1]['values'])
    past=times<decision
    combined_times=np.r_[times[past],prefix];combined_values=np.vstack([values[past],state])
    prediction=ego_track(scene,combined_times,combined_values,int(past.sum()),'generated')
    prediction.metadata.update(replay_max_gap_s=cfg['max_prediction_gap_s'],
        history_policy='logged_pre_decision_history_plus_only_executed_terminal_prefix; no_post_terminal_ego')
    nearest=min(timestamps,key=lambda f:abs(timestamps[f]-end))
    source=dict(event,first_collision_frame=nearest,first_collision_frame_domain='nearest_native_scene_frame',
        first_collision_frame_time_error_s=abs(timestamps[nearest]-end))
    canonical_tracks={t.agent_id:t for t in tracks};canonical_tracks['ego']=prediction
    cars=evaluate_collision_responsibility([source],canonical_tracks,map_context,
        dict(config.get('collision_responsibility',{}),traffic_config=config.get('traffic_compliance',{})),context)['events'][0]
    evidence=evaluate_event(cars,dict(raw_tracks,ego=prediction),map_context,config.get('responsibility_v2',{}),
        cars_event=cars,traffic_config=config.get('traffic_compliance',{}),kinematic_thresholds=tcfg.get('kinematics',{}))
    final=classify(evidence,config.get('collision_responsibility_v3',{}))
    final.update(episode_id=event['episode_id'],responsibility_scope='episode_terminal_collision_only',
        history_source='logged_pre_decision_history_plus_terminal_executed_prefix; not_stitched_reset_trajectory')
    return final

def evaluate(config, context, summary):
    """Load verified saved predictions, keep legacy scoring untouched, add episodes."""
    cfg=_config(config.get('episode_rollout',{})); root=Path(context['root']); output=Path(context['output_dir'])
    acfg=config['evaluation']['agent']; tcfg=config['evaluation']['trajectory']
    paths=acfg.get('scene_paths',tcfg.get('scene_paths',[])); agents=config.get('agents',{}).get('include',acfg.get('agents',[]))
    datasets={}; prediction_hashes={}; metric_hashes={}
    manifest_path=output/'real_agents'/'input_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    from driveaudit_eval.agents.input_gt import prediction_windows
    expected_windows=prediction_windows(acfg,context)
    if manifest['common_windows']!=expected_windows: raise ValueError('Episode manifest/current decision alignment mismatch')
    for image_path,digest in manifest['image_sha256'].items():
        if _digest(root/image_path)!=digest: raise ValueError('Episode current RGB hash mismatch: '+image_path)
    frame_inputs={(scene['scene_id'],frame['frame_id']):frame for scene in manifest['scenes'] for frame in scene['frames']}
    for agent in agents:
        prediction_path=output/'real_agents'/agent/'predictions.jsonl'
        metric_path=output/'real_agents'/agent/'agent_details'/agent/'per_frame_metrics.csv'
        if not prediction_path.is_file() or not metric_path.is_file():
            statuses=[s for s in manifest.get('scene_status',[]) if s['agent']==agent]
            if statuses and all(s['input_status']=='unavailable' for s in statuses): continue
            raise ValueError('Episode requires verified prediction and metric artifacts for '+agent)
        inference=json.loads((prediction_path.parent/'inference_status.json').read_text())
        if inference.get('status')!='COMPLETED' or inference.get('predictions_sha256')!=_digest(prediction_path):
            raise ValueError('Episode inference status/hash mismatch: '+agent)
        reuse_path=prediction_path.parent/'prediction_reuse.json'
        source_manifest=manifest_path
        if reuse_path.is_file():
            reuse=json.loads(reuse_path.read_text());source_manifest=Path(reuse['source']).parent.parent/'input_manifest.json'
            source=json.loads(source_manifest.read_text())
            if source['image_sha256']!=manifest['image_sha256'] or source['common_windows']!=manifest['common_windows']:
                raise ValueError('Episode reused prediction source manifest mismatch')
        # Inference manifest digest binds original inference; reused manifests may move output paths.
        if _digest(source_manifest)!=inference.get('manifest_sha256'):
            original=source_manifest.parent/agent/'prediction_reuse.json'
            seen=set()
            while original.is_file() and str(original) not in seen:
                seen.add(str(original));previous=json.loads(original.read_text())
                source_manifest=Path(previous['source']).parent.parent/'input_manifest.json'
                if _digest(source_manifest)==inference.get('manifest_sha256'): break
                original=source_manifest.parent/agent/'prediction_reuse.json'
            if _digest(source_manifest)!=inference.get('manifest_sha256'): raise ValueError('Original inference manifest digest unresolved')
        prediction_inputs=frame_inputs
        if reuse_path.is_file():
            source=json.loads(source_manifest.read_text())
            prediction_inputs={(s['scene_id'],f['frame_id']):f for s in source['scenes'] for f in s['frames']}
        predictions={}
        for line in prediction_path.read_text().splitlines():
            if not line.strip(): continue
            record=json.loads(line); key=(record['scene_id'],int(record['frame_id']))
            if key in predictions: raise ValueError('Duplicate prediction key')
            if key not in prediction_inputs or record.get('agent')!=agent or record.get('inference_success') is not True or record.get('coordinate_system')!='ego_x_forward_y_left':
                raise ValueError('Episode prediction identity/coordinate/inference mismatch')
            native_timestamp=prediction_inputs[key]['model_input']['ego_state']['timestamp']
            if record.get('timestamp')!=native_timestamp: raise ValueError('Episode prediction native timestamp mismatch')
            predictions[key]=record
        with metric_path.open() as handle: metrics=list(csv.DictReader(handle))
        datasets[agent]=(predictions,metrics); prediction_hashes[str(prediction_path)]=_digest(prediction_path);metric_hashes[str(metric_path)]=_digest(metric_path)
    result=dict(status='completed',windows=[],episodes=[],terminal_responsibility=[],route_timeseries=[],config=cfg)
    if cfg['terminal_execution_mode']=='horizon_commit':
        result['strict_prefix']=dict(status='completed',config=dict(cfg,terminal_execution_mode='nominal_prefix'),episodes=[],windows=[],terminal_responsibility=[],route_timeseries=[])
    audit=dict(evaluation_mode='logged_receding_horizon_with_terminal_commit' if cfg['terminal_execution_mode']=='horizon_commit' else 'logged_receding_horizon_execution',is_true_closed_loop=False,
        prediction_reinference_executed=False,other_actor_response=False,stateful_physical_rollout=False,
        prediction_representation='saved ego-relative positions transformed at each logged decision pose',
        collision_geometry='same original agent SAT fixed ego proxy footprint and native sampled other dimensions',
        collision_search_limit='sampled time search and bisection; sub-check-step contacts not analytically certified',
        route_definition='full logged ego path cumulative arc length; log endpoint proxy, not an observed mission goal',
        EP_definition='open_loop_progress_proxy over independent three-second logged reference path',
        prediction_hashes_before=prediction_hashes,metric_hashes_before=metric_hashes,scene_audit=[],
        current_rgb_hashes_verified=True,current_common_windows_verified=True,inference_status_hashes_verified=True,
        input_manifest_sha256=_digest(manifest_path),rgb_image_count=len(manifest['image_sha256']))
    from driveaudit_eval.responsibility.ratios import raw_evidence_tracks
    from driveaudit_eval.responsibility.evidence import evaluate_event
    from driveaudit_eval.responsibility.attribution import classify
    from driveaudit_eval.responsibility.cars_adapter import evaluate_collision_responsibility
    for relative in paths:
        path=root/relative; canonical=path.parent.parent.name+'__'+path.name
        timestamps,times,values,raw_actors,timeline=load_logged(path)
        tracks,provenance=load_tracks(path); scene=provenance['scene_id']
        raw_tracks=raw_evidence_tracks(scene,raw_actors,provenance);raw_tracks.update({t.agent_id:t for t in tracks})
        from driveaudit_eval.traffic_compliance import load_map_context
        map_context=load_map_context(path,acfg.get('native_map',tcfg.get('native_map')))
        active={agent:data for agent,data in datasets.items() if any(r['scene_id']==canonical for r in data[1])}
        scene_keys=[]
        for agent,(predictions,metrics) in active.items():
            scene_keys.append({int(r['frame_id']) for r in metrics if r['scene_id']==canonical and (canonical,int(r['frame_id'])) in predictions})
        common=set.intersection(*scene_keys) if scene_keys else set()
        if not common:
            if manifest.get('scene_status'):
                audit['scene_audit'].append(dict(scene_id=scene,status='unavailable',reason='no_supported_agent_predictions'))
                continue
            raise ValueError('No common valid decision windows in '+scene)
        frames=sorted(common,key=lambda frame:timestamps[frame])
        route=values[:,:2] if cfg['route']['source']=='logged_ego_path_proxy' else None
        if cfg['route']['source'] not in ('logged_ego_path_proxy','unavailable'): raise ValueError('No audited external reference route loader for configured source')
        scene_audit=dict(scene_id=scene,canonical_scene_id=canonical,common_decision_frames=frames,
            decision_timestamps=[timestamps[f] for f in frames],scene_end_time=float(times[-1]),
            replan_intervals_s=np.diff([timestamps[f] for f in frames]).tolist(),agents_share_decision_timestamps=True,
            route_kind=cfg['route']['source'],route_point_count=len(route) if route is not None else 0,
            native_timeline_sha256=_digest(path/'timeline.json'),native_ego_history_sha256=_digest(path/'history/ego.csv'),
            native_ego_future_sha256=_digest(path/'future/ego_trajectory.csv'))
        audit['scene_audit'].append(scene_audit)
        for agent,(predictions,metrics) in active.items():
            context['logger'].info('[EPISODE] %s / %s',scene,agent)
            metric_by_frame={int(r['frame_id']):r for r in metrics if r['scene_id']==canonical}
            windows=[]
            for frame in frames:
                record=predictions[canonical,frame]; decision=float(timestamps[frame]); origin,ok=sample(times,values,[decision])
                if not ok.all(): raise ValueError('Episode decision logged pose unavailable')
                offsets=np.asarray(record['time_offsets'],float)
                if len(offsets)<1 or offsets[0]<=0 or np.any(np.diff(offsets)<=0): raise ValueError('Invalid saved prediction offsets')
                world=ego_to_world(record['planning_traj'],[*origin[0,:2],0.],origin[0,2])
                positions=np.vstack([origin[0,:2],world]); headings=np.r_[origin[0,2],trajectory_headings(positions,origin[0,2])[1:]]
                original=metric_by_frame[frame]; relative_collision=original.get('first_collision_time')
                first=decision+float(relative_collision) if relative_collision not in (None,'','nan','None') else None
                windows.append(dict(decision_frame_id=frame,decision_time=decision,timestamps=np.r_[decision,decision+offsets],
                    positions=positions,headings=headings,open_loop_collision=float(original.get('collision_rate') or 0)>0,
                    open_loop_first_collision_time=first,prediction_record=record))
            def terminal_callback(event, window, executed_history):
                return evaluate_terminal_responsibility(event,window,executed_history,scene,times,values,
                    tracks,raw_tracks,map_context,config,context,timestamps)
            one=run_episode(scene,agent,windows,raw_actors,route,dict(cfg,traffic_config=config.get('traffic_compliance',{})),map_context,terminal_callback,float(times[-1]),
                dict(source=cfg['route']['source'],history_path=str(path/'history/ego.csv'),future_path=str(path/'future/ego_trajectory.csv'),
                     mission_route_available=False,goal_is_logged_endpoint_proxy=True))
            if 'strict_prefix' in result:
                strict=run_episode(scene,agent,windows,raw_actors,route,dict(cfg,terminal_execution_mode='nominal_prefix',traffic_config=config.get('traffic_compliance',{})),map_context,terminal_callback,float(times[-1]),one['episode']['route_provenance'])
                result['strict_prefix']['episodes'].append(strict['episode'])
                for key in ('windows','terminal_responsibility','route_timeseries'):result['strict_prefix'][key].extend(strict[key])
            result['episodes'].append(one['episode'])
            for key in ('windows','terminal_responsibility','route_timeseries'):result[key].extend(one[key])
    audit['prediction_hashes_after']={p:_digest(p) for p in prediction_hashes}
    audit['metric_hashes_after']={p:_digest(p) for p in metric_hashes}
    audit['prediction_hashes_unchanged']=audit['prediction_hashes_after']==prediction_hashes
    audit['metric_hashes_unchanged']=audit['metric_hashes_after']==metric_hashes
    if not audit['prediction_hashes_unchanged'] or not audit['metric_hashes_unchanged']: raise ValueError('Original artifacts changed during episode evaluation')
    result['audit']=audit
    return _jsonable(result)

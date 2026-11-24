"""Model-based collision responsibility evidence attribution; not legal responsibility.

Ratios normalize diagnostic evidence, not calibrated probabilities. All motion
features use observations at or before collision; projected corridors are explicit
constant-velocity proxies. No CARS availability gate controls basic analysis.
"""
from copy import copy, deepcopy
import math
import numpy as np
from driveaudit_eval.traffic_compliance import MapContext, evaluate_actor_compliance
from driveaudit_eval.trajectory.kinematic_metrics import kinematic_series

DEFAULTS = {
    'enabled': True, 'pre_collision_window_s': 3., 'minimum_basic_window_s': .5,
    'max_gap_s': .5, 'epsilon': 1e-9,
    'weights': {'rule': .30, 'causal': .30, 'avoidability': .25, 'maneuver': .15},
    'responsibility_class': {'primary_threshold': .65, 'shared_lower': .35, 'minimum_confidence': .35},
    'avoidability': {'ttc_warning_threshold_s': 3., 'reaction_time_s': .75, 'comfortable_deceleration_mps2': 4.},
    'geometry': {'corridor_horizon_s': 1., 'center_corridor_tolerance_m': 1.,
                 'parallel_heading_rad': math.pi/4, 'head_on_heading_rad': 3*math.pi/4,
                 'lateral_motion_threshold_mps': .5, 'cut_in_displacement_m': 1.,
                 'intrusion_reference_speed_mps': 2., 'rear_end_leader_score': 0.,
                 'cut_in_other_score': .1, 'head_on_conflict_boost': .4, 'symmetric_conflict_score': .5},
    'maneuver': {'reverse_speed_mps': .5, 'sudden_heading_change_rad': .5},
    'confidence': {'separation_floor': .5, 'history_floor': .5, 'unknown_type_factor': .8,
                   'missing_size_factor': .8},
    'cars': {'enabled': True, 'alpha': .5},
    'responsibility_adjusted_nc': {'enabled': False},
    'experiments': {'ablation': True, 'weight_sensitivity': True, 'window_sensitivity': True},
}


def _config(config):
    out=deepcopy(DEFAULTS)
    def merge(dst, src):
        for key,value in src.items():
            if isinstance(value,dict) and isinstance(dst.get(key),dict): merge(dst[key],value)
            else: dst[key]=deepcopy(value)
    merge(out,config or {})
    if out.get('basic_history_protocol','duration_v1') not in ('duration_v1','observed_pair_v1'):
        raise ValueError('unknown basic responsibility history protocol')
    if set(out['weights']) != {'rule','causal','avoidability','maneuver'}:
        raise ValueError('unknown responsibility weight key')
    if any(not np.isfinite(v) or v<0 for v in out['weights'].values()) or sum(out['weights'].values())<=0:
        raise ValueError('responsibility weights must be finite and nonnegative')
    if not 0<=out['cars']['alpha']<=1: raise ValueError('CARS alpha must be in [0,1]')
    if any(not np.isfinite(out[k]) or out[k]<=0 for k in ('pre_collision_window_s','minimum_basic_window_s','max_gap_s','epsilon')): raise ValueError('history and gap must be finite and positive')
    classes=out['responsibility_class']
    if any(not np.isfinite(v) or not 0<=v<=1 for v in classes.values()) or classes['shared_lower']>=classes['primary_threshold']:
        raise ValueError('invalid responsibility class thresholds')
    if any(not np.isfinite(v) or v<=0 for v in out['avoidability'].values()):
        raise ValueError('avoidability thresholds must be finite and positive')
    return out


def _clean(value):
    if isinstance(value,dict): return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [_clean(v) for v in value]
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def _slice(track, start, end):
    out=copy(track)
    ix=np.flatnonzero((track.timestamps>=start)&(track.timestamps<=end))
    for name in ('timestamps','positions','headings','velocities','accelerations','valid_mask'):
        setattr(out,name,np.asarray(getattr(track,name))[ix].copy())
    out.metadata=dict(track.metadata,history_frames=0)
    return out


def _states(track):
    return [dict(timestamp=t,x=p[0],y=p[1],yaw=h,speed=v,acceleration=a,valid=valid)
        for t,p,h,v,a,valid in zip(track.timestamps,track.positions,track.headings,
                                 track.velocities,track.accelerations,track.valid_mask)]


def _motion(track, times, max_gap):
    ix=np.flatnonzero(track.valid_mask & np.isfinite(track.headings) & np.isfinite(track.positions).all(1))
    if len(ix)<2: return None
    t=track.timestamps[ix]
    if np.any(np.diff(t)<=0): return None
    right=np.searchsorted(t,times,side='left').clip(0,len(t)-1)
    left=np.maximum(right-1,0)
    exact=np.isclose(t[right],times,rtol=0,atol=1e-9)
    support=(times>=t[0])&(times<=t[-1])&(exact|((t[right]-t[left])<=max_gap))
    p=np.column_stack([np.interp(times,t,track.positions[ix,k]) for k in (0,1)])
    h=np.interp(times,t,np.unwrap(track.headings[ix]))
    return p,h,support


def _rule_evidence(track, tracks, context, config):
    if track.agent_type not in ('ego','vehicle'):
        return dict(score=None,rule_evidence_coverage=0.,reason='vehicle_rules_not_applicable',rules={})
    if track.validate():
        return dict(score=None,rule_evidence_coverage=0.,reason='canonical_vehicle_or_geometry_unavailable',rules={})
    # Only validated vehicle peers enter vehicle-specific following rules.
    peers=[t for t in tracks if t.agent_type in ('ego','vehicle') and not t.validate()]
    try: result=evaluate_actor_compliance(track.agent_id,track,peers,context or MapContext(),config)
    except (ValueError,KeyError,IndexError) as exc:
        return dict(score=None,rule_evidence_coverage=0.,reason='traffic_evaluator:'+str(exc),rules={})
    rules=result['rules']; profile=result['profiles']['normalized']
    eligible={k:r for k,r in rules.items() if r['observable'] and r['severity'] is not None}
    return dict(score=1-profile['traffic_compliance_score'] if profile['traffic_compliance_score'] is not None else None,
        rule_evidence_coverage=profile['traffic_compliance_coverage'],
        active_rule_violations=[k for k,r in eligible.items() if r['violated']],
        rule_violation_duration={k:r['duration'] for k,r in eligible.items()},
        rule_violation_severity={k:r['severity'] for k,r in eligible.items()},rules=rules,
        reason=result.get('reason'))


def _cars(cars_event, actor):
    direction=(cars_event or {}).get('directions',{}).get(str(actor),{})
    controllers=direction.get('controllers',{})
    valid=[r for r in controllers.values() if r.get('status')=='valid' and isinstance(r.get('raw',{}).get('preventable'),(bool,int))]
    votes=sum(bool(r['raw']['preventable']) for r in valid) if len(valid)==3 else None
    return dict(controllers=controllers,votes=votes,vote_ratio=votes/3 if votes is not None else None)


def rescore(evidence_result, config=None, use_cars=True):
    """Return a JSONable copy scored with new weights, without replaying motion.

    config is the collision_responsibility_v2 subsection. Set unwanted dimensions
    to weight zero for ablation. Missing dimensions are absent, never zero-filled.
    """
    cfg=_config(config); out=deepcopy(evidence_result); weights=cfg['weights']
    total=sum(weights.values()); active=[]; unavailable=[]; weighted_quality=[]; coverages=[]
    scores=[]; vote_agreement=[]
    for key in ('actor_a','actor_b'):
        e=out['evidence']['actors'][key]; component=dict(e['scores'])
        vote=e.get('cars_vote_ratio') if use_cars and cfg['cars']['enabled'] and out.get('cars_status')=='valid' else None
        if vote is not None and cfg['cars']['alpha']>0:
            basic=component['avoidability']; alpha=cfg['cars']['alpha']
            component['avoidability']=vote if basic is None else (1-alpha)*basic+alpha*vote
            vote_agreement.append(max(vote,1-vote))
        available={k:v for k,v in component.items() if v is not None and weights.get(k,0)>0}
        denominator=sum(weights[k] for k in available)
        score=sum(weights[k]*v for k,v in available.items())/denominator if denominator else None
        scores.append(score); coverages.append(denominator/total if total else 0.)
        quality=sum(weights[k]*(e.get('rule_coverage',0.) if k=='rule' else 1.) for k in available)
        weighted_quality.append(quality/total if total else 0.)
        for k,v in component.items():
            field='avoidability' if k=='avoidability' else k
            out[key+'_'+field+'_score']=v
            out[key+'_'+('avoid' if k=='avoidability' else k)+'_available']=v is not None
            (active if k in available else unavailable).append(key+'.'+k)
        out[key+'_raw_responsibility_score']=score
        for name,rule in e.get('traffic_rules',{}).get('rules',{}).items():
            if not rule.get('observable'): unavailable.append(key+'.rule.'+name)
    coverage=float(np.mean(coverages)); score_total=sum(s for s in scores if s is not None)
    ratios=[s/score_total for s in scores] if all(s is not None for s in scores) and score_total>cfg['epsilon'] else [None,None]
    if out.get('basic_status')=='unscorable': ratios=[None,None]
    sep=abs(ratios[0]-ratios[1]) if ratios[0] is not None else 0.
    cc=cfg['confidence']; terms=dict(coverage_term=float(np.mean(weighted_quality)),
        separation_term=cc['separation_floor']+(1-cc['separation_floor'])*sep,
        model_agreement_term=float(np.mean(vote_agreement)) if vote_agreement else 1.,
        history_term=cc['history_floor']+(1-cc['history_floor'])*out.get('history_coverage',0.),
        actor_type_term=cc['unknown_type_factor'] if 'other' in (out.get('actor_a_type'),out.get('actor_b_type')) else 1.,
        geometry_term=cc['missing_size_factor'] if out['evidence'].get('geometry',{}).get('geometry_size_status') in ('unavailable','proxy') else 1.)
    confidence=float(np.clip(np.prod(list(terms.values())),0,1)) if ratios[0] is not None else 0.
    cls='uncertain'; reason=None; thresholds=cfg['responsibility_class']
    if out.get('basic_status')=='unscorable': cls='unscorable'; reason=out.get('reason')
    elif ratios[0] is None: reason='insufficient_discriminative_evidence'
    elif confidence<thresholds['minimum_confidence']: reason='low_evidence_confidence'
    else:
        cls='actor_a_primary' if ratios[0]>=thresholds['primary_threshold'] else 'actor_b_primary' if ratios[0]<=thresholds['shared_lower'] else 'shared'
    out.update(actor_a_responsibility_ratio=ratios[0],actor_b_responsibility_ratio=ratios[1],
        responsibility_evidence_ratio=dict(actor_a=ratios[0],actor_b=ratios[1]),
        responsibility_class=cls,responsibility_confidence=confidence,confidence_terms=terms,
        evidence_coverage=coverage,responsibility_evidence_coverage=coverage,
        status='unscorable' if cls=='unscorable' else 'uncertain' if cls=='uncertain' else 'valid',
        active_evidence=active,unavailable_evidence=unavailable,reason=reason,
        scoring_weights=dict(weights),cars_fusion_enabled=bool(use_cars and cfg['cars']['enabled']))
    out['cars_available']=out.get('cars_status')=='valid'
    out['cars_status_definition']='raw_three_controller_evidence_availability; fusion controlled separately by cars_fusion_enabled'
    for name in ('rule','causal','avoid','maneuver'):
        out[name+'_available']=any(out[k+'_'+name+'_available'] for k in ('actor_a','actor_b'))
    if out.get('ego_involved'):
        idx=0 if out.get('actor_a_type')=='ego' or out.get('actor_a_id')=='ego' else 1
        own='actor_a_primary' if idx==0 else 'actor_b_primary'
        out.update(ego_responsibility_ratio=ratios[idx],other_responsibility_ratio=ratios[1-idx],
            ego_responsibility_class='ego_primary' if cls==own else 'other_primary' if cls in ('actor_a_primary','actor_b_primary') else cls,
            ego_responsibility_confidence=confidence)
    else:
        out.update(ego_responsibility_ratio=None,other_responsibility_ratio=None,
            ego_responsibility_class=None,ego_responsibility_confidence=None)
    return _clean(out)


def evaluate_event(source, tracks_by_id, map_context=None, config=None, cars_event=None,
                   traffic_config=None, kinematic_thresholds=None):
    cfg=_config(config); geo=cfg['geometry']; eps=cfg['epsilon']
    pair=source.get('collision_pair') or [source.get('actor_a'),source.get('actor_b')]
    ids=[str(x) if x is not None else None for x in pair]
    collision=source.get('first_collision_time',source.get('collision_time'))
    out=dict(source=deepcopy(source),scene_id=source.get('scene_id'),agent_name=source.get('agent_name',source.get('agent')),
        decision_frame_id=source.get('decision_frame_id'),event_id=source.get('collision_event_id',source.get('event_id')),
        actor_a_id=ids[0],actor_b_id=ids[1],collision_time=collision,collision_frame=source.get('collision_frame',source.get('first_collision_frame')),
        basic_status='unscorable',cars_status='unavailable',cars_output=deepcopy(cars_event),
        collision_type='unknown',available_history_s=0.,history_coverage=0.,
        actor_a_states=[],actor_b_states=[],evidence={'actors':{},'geometry':{},'history':{}},
        attribution_definition='model-based responsibility evidence attribution; not legal responsibility')
    tracks=[tracks_by_id.get(x) for x in ids]
    for key,track,actor in zip(('actor_a','actor_b'),tracks,ids):
        out[key+'_type']=track.agent_type if track is not None else None
        out[key+'_type_confidence']=track.metadata.get('actor_type_confidence',1. if track.agent_type!='other' else 0.) if track is not None else None
        cars=_cars(cars_event,actor)
        out['evidence']['actors'][key]=dict(scores=dict.fromkeys(cfg['weights']),rule_coverage=0.,cars_vote_ratio=cars['vote_ratio'])
        out['cars_'+key+'_votes']=cars['votes']
        for controller in ('FSM','CCJP','RSS'): out['cars_'+key+'_'+controller.lower()]=cars['controllers'].get(controller)
    out['ego_involved']=any(x=='ego' for x in ids) or any(t is not None and t.agent_type=='ego' for t in tracks)
    out['cars_status']='not_applicable' if any(t is not None and t.agent_type in ('pedestrian','bicycle') for t in tracks) else 'valid' if all(out['evidence']['actors'][k]['cars_vote_ratio'] is not None for k in ('actor_a','actor_b')) else 'unavailable'
    def fail(reason):
        out['reason']=reason
        return rescore(out,cfg)
    if any(t is None for t in tracks) or ids[0]==ids[1]: return fail('collision_pair_or_position_trajectory_unresolved')
    if collision is None or not np.isfinite(collision): return fail('collision_time_unresolved')
    if tracks[0].coordinate_system!=tracks[1].coordinate_system or tracks[0].coordinate_system not in ('world','history_last_ego'):
        return fail('incompatible_coordinate_frames')
    if source.get('geometry_valid') is False: return fail('collision_geometry_untrusted')
    tracks=[_slice(t,collision-cfg['pre_collision_window_s'],collision) for t in tracks]
    for key,t in zip(('actor_a','actor_b'),tracks): out[key+'_states']=_states(t)
    if any(len(t.timestamps)<2 for t in tracks): return fail('insufficient_position_history')
    times=np.unique(np.concatenate([t.timestamps for t in tracks]))
    gap_limits=[min(cfg['max_gap_s'],float(t.metadata.get('replay_max_gap_s',cfg['max_gap_s']))) for t in tracks]
    motions=[_motion(t,times,gap) for t,gap in zip(tracks,gap_limits)]
    if any(x is None for x in motions): return fail('insufficient_valid_position_history')
    mask=motions[0][2]&motions[1][2]
    times=times[mask]
    if len(times)<2: return fail('unsynchronized_history')
    # Use the most recent contiguous supported run; never bridge absent history.
    gaps=np.flatnonzero(np.diff(times)>min(gap_limits)+eps)
    start=int(gaps[-1]+1) if len(gaps) else 0
    times=times[start:]; motions=[(p[mask][start:],h[mask][start:]) for p,h,_ in motions]
    duration=float(times[-1]-times[0]) if len(times)>1 else 0.
    out.update(available_history_s=duration,pre_collision_duration=duration,history_coverage=min(1.,duration/cfg['pre_collision_window_s']))
    out['evidence']['history']=dict(synchronized_timestamps=times,policy='last_contiguous_common_pre_collision_run',max_gap_s=min(gap_limits),actor_max_gap_s=dict(zip(ids,gap_limits)))
    if cfg.get('basic_history_protocol')=='observed_pair_v1':
        # Two synchronized observations identify relative motion. Duration remains
        # confidence evidence; CARS and higher-order maneuvers keep their own gates.
        out.update(basic_history_protocol='observed_pair_v1',basic_history_sample_count=len(times),
                   minimum_history_duration_met=duration+eps>=cfg['minimum_basic_window_s'])
        if len(times)<2 or duration<=eps: return fail('insufficient_contiguous_position_history')
    elif duration+eps<cfg['minimum_basic_window_s']: return fail('less_than_minimum_basic_history')
    collision_gap=min(gap_limits) if cfg.get('basic_history_protocol')=='observed_pair_v1' else cfg['max_gap_s']
    if collision-times[-1]>collision_gap: return fail('history_not_connected_to_collision')
    tracks=[_slice(t,times[0],times[-1]) for t in tracks]
    for key,t in zip(('actor_a','actor_b'),tracks): out[key+'_states']=_states(t)
    out['basic_status']='valid'; p=[x[0] for x in motions]; h=[x[1] for x in motions]
    v=[np.vstack([np.diff(x,axis=0)[0]/np.diff(times)[0],np.diff(x,axis=0)/np.diff(times)[:,None]]) for x in p]
    speed=[np.linalg.norm(x,axis=1) for x in v]
    delta=p[1]-p[0]; distance=np.linalg.norm(delta,axis=1); dv=v[1]-v[0]
    closing=-np.sum(delta*dv,axis=1)/np.maximum(distance,eps)
    rel_angle=np.abs((h[1]-h[0]+np.pi)%(2*np.pi)-np.pi)
    sizes=[all(np.isfinite(t.shape.get(k,np.nan)) and t.shape.get(k,0)>0 for k in ('length','width')) for t in tracks]
    shape_sources={ids[i]:tracks[i].metadata.get('shape_source','supplied_shape_provenance_unspecified') for i in (0,1)}
    proxy_size=any('proxy' in str(value).lower() for value in shape_sources.values())
    width=sum(t.shape['width']/2 for t in tracks) if all(sizes) else geo['center_corridor_tolerance_m']
    half_length=sum(t.shape['length']/2 for t in tracks) if all(sizes) else 0.
    relative=[]; intrusions=[]
    for i in (0,1):
        j=1-i; unit=np.column_stack([np.cos(h[i]),np.sin(h[i])]); lateral=np.column_stack([-np.sin(h[i]),np.cos(h[i])])
        offset=p[j]-p[i]; long=np.sum(offset*unit,axis=1); lat=np.sum(offset*lateral,axis=1)
        relv=v[j]-v[i]; latv=np.sum(relv*lateral,axis=1)
        # ponytail: straight constant-velocity corridor, curved lane polygons if map fidelity is required.
        projected=lat+latv*geo['corridor_horizon_s']; depth=np.maximum(0.,width-np.abs(projected))
        inward=np.maximum(0.,-np.sign(lat)*latv)
        projected_long=long+np.sum(relv*unit,axis=1)*geo['corridor_horizon_s']
        corridor=(np.maximum(long,projected_long)>=-half_length)&(np.minimum(long,projected_long)<=speed[i]*geo['corridor_horizon_s']+half_length)
        score=float(np.max(np.clip(inward/geo['intrusion_reference_speed_mps'],0,1)*np.clip(depth/max(width,eps),0,1)*corridor))
        intrusion_times=np.flatnonzero((depth>0)&corridor&(inward>geo['lateral_motion_threshold_mps']))
        relative.append(dict(longitudinal_offset=long,lateral_offset=lat,relative_heading=rel_angle,
            relative_velocity=relv,lateral_relative_velocity=latv,projected_longitudinal_offset=projected_long))
        intrusions.append(dict(intruding_actor_id=ids[j],intrusion_score=score,
            time_to_corridor_intrusion=float(max(0.,(abs(lat[intrusion_times[0]])-width)/max(inward[intrusion_times[0]],eps))) if len(intrusion_times) else None,
            intrusion_depth=float(np.max(depth*corridor)),intrusion_duration=float(np.sum(np.diff(times)*((depth>0)&corridor)[:-1])),
            lateral_displacement_toward_corridor=float(abs(lat[0])-abs(lat[-1]))))
    # Entry is actor motion toward the other's corridor, not relative lateral velocity alone.
    entering=[]
    for i in (0,1):
        j=1-i; axis=np.column_stack([-np.sin(h[j]),np.cos(h[j])]); own_lat=np.abs(np.sum(v[i]*axis,axis=1))
        entering.append(intrusions[j]['lateral_displacement_toward_corridor']>=geo['cut_in_displacement_m'] and np.max(own_lat)>=geo['lateral_motion_threshold_mps'])
    angle=float(np.median(rel_angle)); parallel=angle<geo['parallel_heading_rad']; causal=[intrusions[1]['intrusion_score'],intrusions[0]['intrusion_score']]
    if parallel and any(entering):
        ctype='merge/cut-in' if all(t.agent_type in ('ego','vehicle') for t in tracks) and all(np.max(z)>geo['lateral_motion_threshold_mps'] for z in speed) else 'side_collision'
        causal=[1. if entering[i] else geo['cut_in_other_score'] for i in (0,1)]
    elif angle>geo['head_on_heading_rad']: ctype='head_on'; causal=[geo['symmetric_conflict_score']]*2
    elif parallel and np.median(np.abs(relative[0]['lateral_offset']))<=width:
        ctype='rear_end'; follower=0 if np.median(relative[0]['longitudinal_offset'])>0 else 1
        causal=[geo['rear_end_leader_score']]*2
        if np.max(closing)>eps: causal[follower]=1.
        leader=1-follower
        leader_forward=np.sum(v[leader]*np.column_stack([np.cos(h[leader]),np.sin(h[leader])]),axis=1)
        if np.min(leader_forward)<-cfg['maneuver']['reverse_speed_mps']:
            causal[leader]=1.; causal[follower]=geo['cut_in_other_score']; ctype='unknown'
    elif not parallel: ctype='crossing'; causal=[geo['symmetric_conflict_score']]*2
    elif any(causal): ctype='side_collision'
    else: ctype='unknown'
    # Static orientation does not create crossing/head-on causal participation.
    participation=[np.maximum(0.,np.sum(v[i]*(p[1-i]-p[i]),axis=1)/np.maximum(distance,eps)) for i in (0,1)]
    if ctype in ('crossing','head_on'):
        strengths=[float(np.max(np.where(closing>eps,z,0.))) for z in participation]
        strongest=max(strengths)
        causal=[geo['symmetric_conflict_score']*z/strongest if strongest>eps else 0. for z in strengths]
    out['collision_type']=ctype
    ttc=np.divide(np.maximum(distance-half_length,0),closing,out=np.full(len(times),np.nan),where=closing>eps)
    warning=np.flatnonzero(np.isfinite(ttc)&(ttc<=cfg['avoidability']['ttc_warning_threshold_s']))
    detectable=float(times[warning[0]]) if len(warning) else None
    geometry=dict(geometry_size_status=('proxy' if proxy_size else 'available') if all(sizes) else 'unavailable',
        actor_shape_sources=shape_sources,causal_closing_participation=participation,
        actor_dimensions={ids[i]:dict(tracks[i].shape) if sizes[i] else None for i in (0,1)},
        native_dimensions={ids[i]:tracks[i].metadata.get('native_dimensions') for i in (0,1)},
        size_sampling_policy='fixed_supplied_shape_for_corridor; native_dimensions_retained_for_audit',
        corridor_definition=('proxy_size_corridor_proxy' if proxy_size else 'measured_size_corridor_proxy') if all(sizes) else 'center_trajectory_proxy',
        corridor_half_width_m=width,corridor_horizon_s=geo['corridor_horizon_s'],
        priority_status='unavailable',relative_coordinates={'actor_a':relative[0],'actor_b':relative[1]},
        actor_a_intrusion=intrusions[1],actor_b_intrusion=intrusions[0],lane_change_actor=[ids[i] for i in (0,1) if entering[i]],
        relative_distance=distance,closing_speed=closing,ttc=ttc,
        first_conflict_detectable_time=detectable,conflict_detection_definition='radial_TTC_warning_proxy')
    if ctype=='crossing':
        arrivals=[[],[]]
        for k in range(len(times)):
            matrix=np.column_stack([v[0][k],-v[1][k]])
            solution=np.linalg.solve(matrix,delta[k]) if abs(np.linalg.det(matrix))>eps else [np.nan,np.nan]
            for i in (0,1): arrivals[i].append(float(solution[i]) if solution[i]>=0 else np.nan)
        geometry.update(time_to_conflict_point=dict(actor_a=arrivals[0],actor_b=arrivals[1]),
            arrival_time_difference=np.asarray(arrivals[0])-np.asarray(arrivals[1]),
            first_entry_to_conflict_area={ids[i]:float(times[np.flatnonzero(np.linalg.norm(p[i]-np.mean([p[0][-1],p[1][-1]],axis=0),axis=1)<=width)[0]]) if np.any(np.linalg.norm(p[i]-np.mean([p[0][-1],p[1][-1]],axis=0),axis=1)<=width) else None for i in (0,1)},
            conflict_area_definition='terminal_collision_center_proxy_no_priority_inference')
    out['evidence']['geometry']=geometry
    rule_rows=[_rule_evidence(t,tracks,map_context,traffic_config) for t in tracks]
    if ctype=='head_on':
        wrong=[bool(r.get('rules',{}).get('wrong_way',{}).get('violated')) for r in rule_rows]
        if sum(wrong)==1 and closing.max()>eps:
            # Rule records time in violation; causal records direction's participation in closing conflict.
            causal=[float(np.clip(geo['symmetric_conflict_score']+geo['head_on_conflict_boost']*(1 if w else -1),0,1)) if np.max(participation[i])>eps else 0. for i,w in enumerate(wrong)]
    for i,key in enumerate(('actor_a','actor_b')):
        t=tracks[i]; vehicle=t.agent_type in ('ego','vehicle'); kin=kinematic_series(t)
        acc=np.asarray(kin['longitudinal_acceleration'])
        finite=np.isfinite(acc)&(t.timestamps>=detectable)&(t.timestamps<collision) if detectable is not None else np.zeros(len(acc),bool)
        interval=np.maximum(0.,np.minimum(np.r_[t.timestamps[1:],collision],collision)-t.timestamps)
        interval*=finite
        actual=float(np.sum(np.maximum(0.,-np.nan_to_num(acc))*interval)/interval.sum()) if interval.sum()>eps else None
        peak=float(max(0.,-np.min(acc[finite]))) if finite.any() else None
        opportunity=max(0.,collision-detectable-cfg['avoidability']['reaction_time_s']) if detectable is not None else 0.
        required=None; deficit=None; avoid=None; avoid_reason='not_applicable_vehicle_braking_model'
        reaction_gap=None
        if vehicle:
            avoid_reason='no_detectable_closing_conflict'
            if ctype=='rear_end' and i!=follower:
                avoid=0.; avoid_reason='not_follower'
            elif len(warning):
                k=int(warning[0]); reaction=cfg['avoidability']['reaction_time_s']
                if ctype=='rear_end':
                    gap=float(relative[i]['longitudinal_offset'][k]-half_length)
                    reaction_gap=gap-float(closing[k])*reaction
                    if closing[k]<=eps: avoid_reason='no_closing_speed'
                    elif gap<=eps: avoid_reason='already_overlapping'
                    elif reaction_gap<=eps or opportunity<=eps: avoid=0.; avoid_reason='insufficient_space_after_reaction'
                    else: required=float(closing[k]**2/(2*reaction_gap))
                else:
                    avoid_reason='lateral_conflict_no_longitudinal_DRAC'
                    if opportunity<=eps: avoid=0.; avoid_reason='insufficient_reaction_time'
                    else: required=float(speed[i][k]/opportunity)
                if required is not None and actual is not None:
                    deficit=max(0.,required-actual)
                    feasible=required<=cfg['avoidability']['comfortable_deceleration_mps2']
                    avoid=float(min(1.,opportunity/cfg['avoidability']['ttc_warning_threshold_s'])*(deficit/max(required,eps))) if feasible else 0.
                    avoid_reason='braking_opportunity_observed' if feasible else 'required_deceleration_exceeds_available_model'
                elif required is not None: avoid_reason='post_warning_acceleration_unavailable'
        limits=(kinematic_thresholds or {}).get('vehicle' if t.agent_type=='ego' else t.agent_type)
        maneuver=None; flags={}; dangerous={}; mitigation=[]
        if limits is not None and vehicle and len(t.timestamps)>=4 and np.isfinite(kin['longitudinal_jerk']).any():
            checks={'hard_braking':acc<limits['longitudinal_acc_min'],
                'hard_acceleration':acc>limits['longitudinal_acc_max'],
                'high_lateral_acceleration':np.abs(kin['lateral_acceleration'])>limits['lateral_acc_max'],
                'high_jerk':np.abs(kin['longitudinal_jerk'])>limits['longitudinal_jerk_max'],
                'high_yaw_rate':np.abs(kin['yaw_rate'])>limits['yaw_rate_max'],
                'reverse_motion':kin['forward_speed'] < -cfg['maneuver']['reverse_speed_mps'],
                'sudden_heading_change':np.r_[False,np.abs(np.diff(np.unwrap(t.headings)))>cfg['maneuver']['sudden_heading_change_rad']]}
            before=t.timestamps<detectable if detectable is not None else np.ones(len(t.timestamps),bool)
            for name,flag in checks.items():
                flag=flag&t.valid_mask&(t.timestamps<collision);flags[name]=t.timestamps[flag].tolist()
                dangerous[name]=t.timestamps[flag&before].tolist() if name!='hard_braking' else []
            mitigation=t.timestamps[checks['hard_braking']&~before&t.valid_mask&(t.timestamps<collision)].tolist()
            maneuver=float(max((len(x)/len(t.timestamps) for x in dangerous.values()),default=0.))
        e=out['evidence']['actors'][key]
        e.update(scores=dict(rule=rule_rows[i]['score'],causal=float(causal[i]),avoidability=avoid,maneuver=maneuver),
            rule_coverage=rule_rows[i]['rule_evidence_coverage'],traffic_rules=rule_rows[i],
            actor_type_confidence=t.metadata.get('actor_type_confidence',1. if t.agent_type!='other' else 0.),
            avoidability=dict(first_conflict_detectable_time=detectable,available_reaction_time=opportunity,
                required_deceleration=required,actual_deceleration=actual,braking_deficit=deficit,
                space_after_reaction_m=reaction_gap,reason=avoid_reason,
                peak_pre_collision_deceleration=peak,
                actual_braking_definition='time_weighted_post_warning_deceleration_strictly_before_collision; kinematic_proxy_not_counterfactual_proof',
                model='longitudinal_DRAC_or_conflict_time_stopping_proxy' if vehicle else 'not_applicable_vehicle_braking_model'),
            maneuver=dict(events=flags,dangerous_pre_conflict_events=dangerous,evasive_braking_timestamps=mitigation,
                kinematic_series=kin,thresholds=limits,reason=None if maneuver is not None else 'insufficient_kinematic_series_or_no_applicable_vehicle_thresholds'))
    return rescore(out,cfg)

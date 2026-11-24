"""V2 integration and descriptive experiments; original scores remain untouched."""
from collections import Counter, defaultdict
import copy

import numpy as np

from driveaudit_eval.trajectory.io import derive_scalar_motion
from driveaudit_eval.trajectory.schema import Trajectory

DISCLAIMER = ('Responsibility ratio 表示基于运动学、交通规则、碰撞因果和可避免性证据得到的模型化责任证据比例，并非法律责任比例。')


def raw_evidence_tracks(scene_id, raw_actors, provenance):
    """Retain excluded types for V2 only; never change canonical scoring tracks."""
    categories = {str(ident): category for category, ids in
                  provenance.get('excluded_unsupported_categories', {}).items() for ident in ids}
    result = {}
    for ident, data in raw_actors.items():
        data = np.asarray(data, float)
        if data.ndim != 2 or data.shape[1] < 4 or not len(data):
            continue
        t, p, h = data[:, 0], data[:, 1:3], data[:, 3]
        valid = np.isfinite(data[:, :4]).all(axis=1)
        speed, acceleration = derive_scalar_motion(p, valid, t)
        dimensions = data[:, 4:6] if data.shape[1] >= 6 else np.empty((len(t), 0))
        measured = dimensions.shape[1] == 2 and np.isfinite(dimensions).all() and (dimensions > 0).all()
        shape = dict(length=float(dimensions[0, 0]), width=float(dimensions[0, 1])) if measured else {}
        result[str(ident)] = Trajectory(scene_id, str(ident), 'other', 'ground_truth', 'history_last_ego',
            t.copy(), p.copy(), h.copy(), speed, acceleration, valid, shape,
            dict(heading_unit='rad', dt=float(np.median(np.diff(t))) if len(t)>1 else None,
                 history_frames=0, source_category=categories.get(str(ident), 'unknown'),
                 actor_type_confidence=0., shape_source='native_annotation_dimensions' if measured else 'unavailable',
                 shape_policy='first native annotation for scalar schema; full native dimensions retained',
                 native_dimensions=dimensions.tolist() if measured else None,
                 shape_timestamps=t.tolist(), replay_max_gap_s=.25,
                 adapter_scope='V2 evidence only; excluded from original metrics and vehicle rules'))
    return result


def _finite(value):
    return isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(value)


def summarize(events, total_windows=None, adjusted_nc=False):
    """Event counts and window exposure are different denominators.

    Unknown ratios yield burden bounds, never silent zero penalty. Multiple
    collisions in a window count separately for burden; NC_RA uses their max.
    """
    n = len(events)
    ego = [e for e in events if e.get('ego_involved', e.get('is_ego_involved', False))]
    classes = Counter(e.get('ego_responsibility_class', 'unscorable') for e in ego)
    scorable = sum(e.get('basic_status') in ('valid', 'uncertain') for e in events)
    ratios = [float(e['ego_responsibility_ratio']) for e in ego if _finite(e.get('ego_responsibility_ratio'))]
    missing = len(ego)-len(ratios)
    by_window = defaultdict(list)
    for e in ego:
        by_window[(e.get('scene_id'), e.get('agent_name', e.get('agent')), e.get('decision_frame_id'))].append(e.get('ego_responsibility_ratio'))
    def avg(key):
        values = [e[key] for e in events if _finite(e.get(key))]
        return float(np.mean(values)) if values else None
    cars_count = sum(e.get('cars_status') in ('valid', 'cars_valid') for e in events)
    denominator = total_windows if total_windows is not None and total_windows > 0 else None
    lower = sum(ratios)/denominator if denominator else None
    upper = (sum(ratios)+missing)/denominator if denominator else None
    result = dict(collision_event_count=n, ego_collision_event_count=len(ego),
        collision_window_count=len(by_window), total_windows=total_windows,
        responsibility_scorable_count=scorable, responsibility_scorable_rate=scorable/n if n else None,
        unscorable_count=n-scorable, responsibility_ratio_count=len(ratios), missing_ego_ratio_count=missing,
        ego_primary_count=classes['ego_primary'], other_primary_count=classes['other_primary'],
        shared_count=sum(e.get('responsibility_class')=='shared' for e in events),
        uncertain_count=sum(e.get('responsibility_class')=='uncertain' for e in events),
        ego_primary_rate=classes['ego_primary']/len(ego) if ego else None,
        other_primary_rate=classes['other_primary']/len(ego) if ego else None,
        shared_rate=classes['shared']/len(ego) if ego else None,
        ego_responsibility_mean=float(np.mean(ratios)) if ratios else None,
        other_responsibility_mean=avg('other_responsibility_ratio'),
        ego_responsibility_median=float(np.median(ratios)) if ratios else None,
        ego_responsibility_p90=float(np.percentile(ratios,90)) if ratios else None,
        mean_confidence=avg('responsibility_confidence'), mean_evidence_coverage=avg('evidence_coverage'),
        cars_available_count=cars_count, cars_available_rate=cars_count/n if n else None,
        raw_collision_rate=len(by_window)/denominator if denominator else None,
        ego_attributed_collision_burden=lower if missing==0 else None,
        ego_attributed_collision_burden_lower_bound=lower,
        ego_attributed_collision_burden_upper_bound=upper,
        burden_status='unavailable_window_denominator' if not denominator else 'partial_unknown_ratios' if missing else 'valid',
        class_distribution=dict(Counter(e.get('responsibility_class') for e in events)),
        rate_denominator='all original ego collision events including uncertain/unscorable; raw rate uses unique collision windows',
        burden_definition='sum of finite ego event evidence ratios / all scored windows; unknown ratios produce bounds',
        NC_RA=None, NC_RA_status='disabled')
    if adjusted_nc:
        penalty=sum(max(values) for values in by_window.values()) if not missing else None
        result.update(NC_RA=1-penalty/denominator if penalty is not None and denominator else None,
                      NC_RA_status='valid' if penalty is not None and denominator else 'unavailable',
                      NC_RA_definition='1 - sum(max ego event ratio per collision window)/all windows; experimental')
    return result


def _compact(row):
    return {key:value for key,value in row.items() if key not in ('evidence', 'actor_a_states', 'actor_b_states', 'directions', 'traffic_before_collision') and not isinstance(value, (dict,list))}


def evaluate_batch(v1_events, tracks, map_context, config, traffic_config, kinematic_thresholds):
    from driveaudit_eval.responsibility.evidence import evaluate_event, rescore
    output = dict(events=[], ablation=[], weight_sensitivity=[], window_sensitivity=[], size_sensitivity=[], cars_comparison=[])
    experiments = config.get('experiments', {})
    for old in v1_events:
        row = evaluate_event(old, tracks, map_context, config, cars_event=old,
                             traffic_config=traffic_config, kinematic_thresholds=kinematic_thresholds)
        row['v1_responsibility_class'] = old.get('responsibility_class')
        row['v1_responsibility_status'] = old.get('responsibility_status')
        output['events'].append(row)
        basic = rescore(row, config, use_cars=False)
        output['cars_comparison'].append(dict(event_id=row['event_id'], scene_id=row['scene_id'],
            agent_name=row.get('agent_name'), cars_status=row['cars_status'],
            ratio_before=basic.get('actor_a_responsibility_ratio'), ratio_after=row.get('actor_a_responsibility_ratio'),
            ego_ratio_before=basic.get('ego_responsibility_ratio'), ego_ratio_after=row.get('ego_responsibility_ratio'),
            class_before=basic['responsibility_class'], class_after=row['responsibility_class'],
            confidence_before=basic['responsibility_confidence'], confidence_after=row['responsibility_confidence'],
            changed_class=basic['responsibility_class']!=row['responsibility_class']))
        if experiments.get('ablation', True):
            for name, dimensions, cars in [('Rule only',['rule'],False),('Causal only',['causal'],False),
                ('Avoidability only',['avoidability'],False),('Maneuver only',['maneuver'],False),
                ('Rule + Causal',['rule','causal'],False),
                ('Rule + Causal + Avoidability',['rule','causal','avoidability'],False),
                ('Full Basic',['rule','causal','avoidability','maneuver'],False),
                ('Full Basic + CARS',['rule','causal','avoidability','maneuver'],True)]:
                cfg=copy.deepcopy(config)
                weights=dict(rule=.30,causal=.30,avoidability=.25,maneuver=.15)
                weights.update(config.get('weights',{}))
                cfg['weights']={k:v if k in dimensions else 0. for k,v in weights.items()}
                output['ablation'].append(dict(_compact(rescore(row,cfg,use_cars=cars)), experiment=name))
        if experiments.get('weight_sensitivity', True):
            settings=experiments.get('weight_sets',dict(Balanced=[.25,.25,.25,.25],Default=[.30,.30,.25,.15],
                        Safety_heavy=[.20,.25,.40,.15],Causation_heavy=[.20,.45,.25,.10],Rule_heavy=[.45,.20,.25,.10]))
            for name, values in settings.items():
                cfg=copy.deepcopy(config); cfg['weights']=dict(zip(('rule','causal','avoidability','maneuver'),values))
                output['weight_sensitivity'].append(dict(_compact(rescore(row,cfg)), experiment=name,weights=cfg['weights']))
        if experiments.get('window_sensitivity', True):
            for seconds in experiments.get('windows_s',[.5,1.,2.,3.]):
                cfg=dict(config,pre_collision_window_s=float(seconds))
                varied=evaluate_event(old,tracks,map_context,cfg,cars_event=old,
                                     traffic_config=traffic_config,kinematic_thresholds=kinematic_thresholds)
                output['window_sensitivity'].append(dict(_compact(varied),experiment=f'{seconds:g}s',
                    requested_window_s=float(seconds),cars_window_policy='fixed V1 CARS history; vary basic evidence window only'))
        if experiments.get('size_sensitivity', True):
            proxy_ids=[ident for ident,track in tracks.items() if 'proxy' in track.metadata.get('shape_source','')]
            if proxy_ids:
                for scale in experiments.get('proxy_size_scales',[.9,1.,1.1]):
                    varied_tracks=dict(tracks)
                    for ident in proxy_ids:
                        track=copy.copy(tracks[ident]); track.shape={k:v*scale for k,v in track.shape.items()}
                        track.metadata=dict(track.metadata,proxy_size_scale=scale)
                        varied_tracks[ident]=track
                    varied=evaluate_event(old,varied_tracks,map_context,config,cars_event=old,
                        traffic_config=traffic_config,kinematic_thresholds=kinematic_thresholds)
                    output['size_sensitivity'].append(dict(_compact(varied),experiment=f'{scale:g}x',
                        proxy_size_scale=scale,policy='vary proxy footprint in V2 evidence only; original event trigger and V1 CARS frozen'))
    return output


def assemble(batches, agent_scenes, scene_ids, config):
    result={key:[row for batch in batches for row in batch[key]] for key in
            ('events','ablation','weight_sensitivity','window_sensitivity','size_sensitivity','cars_comparison')}
    events=result['events']
    adjusted=config.get('responsibility_adjusted_nc',{}).get('enabled',False)
    result['agent_summary']=[]
    for agent in sorted({r['agent'] for r in agent_scenes}):
        models=[r for r in agent_scenes if r['agent']==agent]
        subset=[r for r in events if r.get('agent_name')==agent]
        result['agent_summary'].append(dict(agent_name=agent,**summarize(subset,sum(r['total_windows'] for r in models),adjusted)))
    result['scene_summary']=[]
    for scene in scene_ids:
        for agent in [None,*sorted({r['agent'] for r in agent_scenes if r['scene_id']==scene})]:
            subset=[r for r in events if r['scene_id']==scene and r.get('agent_name')==agent]
            windows=sum(r['total_windows'] for r in agent_scenes if r['scene_id']==scene and r['agent']==agent) if agent else None
            result['scene_summary'].append(dict(scene_id=scene,agent_name=agent,
                scope='scenario_ground_truth' if agent is None else 'agent_open_loop',**summarize(subset,windows,adjusted)))
    result['overall']=summarize(events,sum(r['total_windows'] for r in agent_scenes),adjusted)
    result['overall']['v1_scorable_count']=sum(r.get('v1_responsibility_status')=='valid' for r in events)
    result['overall']['cars_changed_final_class_count']=sum(r['changed_class'] for r in result['cars_comparison'] if r['cars_status']=='valid')
    result['collision_type_summary']=[dict(collision_type=kind,**summarize([r for r in events if r['collision_type']==kind]))
                                      for kind in sorted({r['collision_type'] for r in events})]
    result['experiment_summary']=[]
    for kind in ('ablation','weight_sensitivity','window_sensitivity','size_sensitivity'):
        for name in sorted({r['experiment'] for r in result[kind]}):
            rows=[r for r in result[kind] if r['experiment']==name]
            result['experiment_summary'].append(dict(experiment_family=kind,experiment=name,**summarize(rows)))
    result['stability']=[]
    for family in ('weight_sensitivity','window_sensitivity','size_sensitivity'):
        for event in events:
            rows=[r for r in result[family] if r['event_id']==event['event_id']]
            ratios=[r['actor_a_responsibility_ratio'] for r in rows if _finite(r.get('actor_a_responsibility_ratio'))]
            primary=[r['responsibility_class'] for r in rows if r['responsibility_class'] in ('actor_a_primary','actor_b_primary')]
            pairs=[(a,b) for i,a in enumerate(primary) for b in primary[i+1:]]
            result['stability'].append(dict(event_id=event['event_id'],experiment_family=family,
                ratio_std=float(np.std(ratios)) if ratios else None,finite_ratio_count=len(ratios),
                primary_actor_flip_rate=sum(a!=b for a,b in pairs)/len(pairs) if pairs else None,
                class_consistency=max(Counter(r['responsibility_class'] for r in rows).values())/len(rows) if rows else None,
                class_flip_rate_vs_default=sum(r['responsibility_class']!=event['responsibility_class'] for r in rows)/len(rows) if rows else None))
    result.update(status='completed',method='model-based responsibility attribution',disclaimer=DISCLAIMER,
        scope='descriptive evaluation on two scenes; not a statistically representative benchmark',config=config)
    return result

"""Mandatory direction attribution of unchanged V2 evidence ratios.

Confidence measures reliability, never permission to classify. Equal unresolved
ratios retain a valid shared_equal result; missing ratios are unscorable.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np

from driveaudit_eval.responsibility.ratios import summarize as summarize_v2

DEFAULTS = dict(enabled=True, mandatory_attribution=True, ratio_tie_epsilon=1e-6,
    dominant_boundary=.5, tie_break_order=['causal','avoidability','rule','maneuver','cars'],
    confidence=dict(high=.70,medium=.40), dominance=dict(strong=.40,moderate=.20,weak=0.),
    strength=dict(very_strong=.80,strong=.65,moderate=.55))


def _finite(x):
    return isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(x)


def _config(config):
    cfg=deepcopy(DEFAULTS)
    for key,value in (config or {}).items():
        if isinstance(value,dict) and isinstance(cfg.get(key),dict):cfg[key].update(value)
        else:cfg[key]=deepcopy(value)
    if cfg['mandatory_attribution'] is not True or cfg['dominant_boundary']!=.5:
        raise ValueError('V3 mandatory attribution requires the fixed 0.5 direction boundary')
    if not _finite(cfg['ratio_tie_epsilon']) or not 0<=cfg['ratio_tie_epsilon']<1:
        raise ValueError('invalid ratio tie epsilon')
    if len(cfg['tie_break_order'])!=5 or set(cfg['tie_break_order'])!=set(DEFAULTS['tie_break_order']):
        raise ValueError('tie_break_order must contain each of the five evidence dimensions once')
    for group,order in [('confidence',['high','medium']),('dominance',['strong','moderate','weak']),('strength',['very_strong','strong','moderate'])]:
        values=[cfg[group][k] for k in order]
        if not all(_finite(x) and 0<=x<=1 for x in values) or any(a<=b for a,b in zip(values,values[1:])):
            raise ValueError('invalid '+group+' thresholds')
    return cfg


def classify(event, config=None):
    cfg=_config(config);row=deepcopy(event)
    a,b=[row.get(key+'_responsibility_ratio') for key in ('actor_a','actor_b')]
    finite=_finite(a) and _finite(b)
    if finite and (not 0<=a<=1 or not 0<=b<=1 or not math.isclose(a+b,1.,rel_tol=0.,abs_tol=1e-12)):
        raise ValueError('finite responsibility ratios must lie in [0,1] and sum to one; V3 never repairs ratios')
    winner=None;decision=None;comparisons=[]
    if finite:
        if abs(a-b)>cfg['ratio_tie_epsilon']:
            winner='actor_a' if a>b else 'actor_b';decision='ratio'
        else:
            for name in cfg['tie_break_order']:
                values=[row.get('cars_'+k+'_votes') if name=='cars' else row.get(k+'_'+name+'_score') for k in ('actor_a','actor_b')]
                selected=row.get('scoring_weights',{}).get('avoidability' if name=='cars' else name,1.)>0
                available=selected and all(_finite(x) for x in values) and (name!='cars' or (row.get('cars_status')=='valid' and row.get('cars_fusion_enabled',True)))
                comparisons.append(dict(evidence=name,actor_a=values[0],actor_b=values[1],comparable=available))
                if available and abs(values[0]-values[1])>cfg['ratio_tie_epsilon']:
                    winner='actor_a' if values[0]>values[1] else 'actor_b';decision=name;break
            if winner is None:
                if a==b:decision='shared_equal_no_discriminative_tie_evidence'
                else:
                    winner='actor_a' if a>b else 'actor_b';decision='original_ratio_after_non_discriminative_near_tie'
    cls=winner+'_dominant' if winner else 'shared_equal' if finite else 'unscorable'
    confidence=row.get('responsibility_confidence')
    level=('high' if confidence>=cfg['confidence']['high'] else 'medium' if confidence>=cfg['confidence']['medium'] else 'low') if _finite(confidence) else 'unavailable'
    margin=abs(a-b) if finite else None
    strength=('strong' if margin>=cfg['dominance']['strong'] else 'moderate' if margin>=cfg['dominance']['moderate'] else 'weak') if finite else None
    maximum=max(a,b) if finite else None
    attribution_strength=next((key for key in ('very_strong','strong','moderate') if maximum>=cfg['strength'][key]),'weak') if finite else None
    row.update(responsibility_class=cls,responsibility_status='valid' if finite else 'unscorable',
        status='valid' if finite else 'unscorable',dominant_responsible_actor=winner,
        dominant_actor_id=row.get(winner+'_id') if winner else None,
        responsibility_confidence_level=level,responsibility_dominance_margin=margin,
        responsibility_dominance_strength=strength,attribution_strength=attribution_strength,
        tie_break_applied=finite and abs(a-b)<=cfg['ratio_tie_epsilon'],tie_break_decision=decision,
        tie_break_evidence=comparisons,reason=None if finite else 'responsibility_ratio_not_available',
        overall_evidence_coverage=row.get('evidence_coverage'),
        confidence_definition='reliability of the evidence ratio; never used for direction or valid-status gating')
    ego=row.get('ego_involved',False)
    own='actor_a' if row.get('actor_a_id')=='ego' or row.get('actor_a_type')=='ego' else 'actor_b'
    if ego:
        own_ratio,other_ratio=(a,b) if own=='actor_a' else (b,a)
        ego_class=('ego_dominant' if winner==own else 'other_dominant') if winner else cls
        row.update(ego_responsibility_ratio=own_ratio,other_responsibility_ratio=other_ratio,
                   ego_responsibility_class=ego_class,ego_responsibility_confidence=confidence,
                   ego_dominant=winner==own if finite else None,
                   other_dominant=winner is not None and winner!=own if finite else None)
    else:
        row.update(ego_responsibility_ratio=None,other_responsibility_ratio=None,ego_responsibility_class=None,
                   ego_dominant=None,other_dominant=None)
    row['dominant_actor']=('ego' if winner==own else 'other') if ego and winner else winner
    for name in ('rule','causal','avoidability','maneuver'):
        values=[]
        for actor in ('actor_a','actor_b'):
            if name=='rule':value=row.get('evidence',{}).get('actors',{}).get(actor,{}).get('rule_coverage')
            else:
                flag=actor+'_'+('avoid' if name=='avoidability' else name)+'_available'
                value=float(bool(row[flag])) if flag in row else None
            if value is not None:values.append(value)
        row[name+'_coverage']=float(np.mean(values)) if values else None
    return row


def summarize(events,total_windows=None):
    base=summarize_v2(events,total_windows)
    fields=['collision_event_count','ego_collision_event_count','collision_window_count','total_windows','raw_collision_rate',
        'ego_responsibility_mean','ego_responsibility_median','ego_responsibility_p90','mean_confidence','mean_evidence_coverage',
        'ego_attributed_collision_burden','ego_attributed_collision_burden_lower_bound','ego_attributed_collision_burden_upper_bound',
        'missing_ego_ratio_count','burden_status','cars_available_count','cars_available_rate']
    result={k:base[k] for k in fields}
    valid=[r for r in events if r['responsibility_status']=='valid'];ego=[r for r in events if r.get('ego_involved')]
    counts=Counter(r.get('ego_responsibility_class') for r in ego)
    weighted=sum(r['ego_responsibility_ratio'] for r in ego if _finite(r.get('ego_responsibility_ratio')))
    margin=[r['responsibility_dominance_margin'] for r in valid]
    result.update(responsibility_scorable_count=len(valid),responsibility_scorable_rate=len(valid)/len(events) if events else None,
        finite_ratio_count=len(valid),dominant_attribution_count=sum(r['dominant_responsible_actor'] is not None for r in valid),
        shared_equal_count=sum(r['responsibility_class']=='shared_equal' for r in valid),
        unscorable_count=len(events)-len(valid),uncertain_count=0,
        ego_dominant_collision_count=counts['ego_dominant'],other_dominant_collision_count=counts['other_dominant'],
        responsibility_weighted_collision_count=weighted if not base['missing_ego_ratio_count'] else None,
        responsibility_weighted_collision_count_lower_bound=weighted,
        responsibility_weighted_collision_count_upper_bound=weighted+base['missing_ego_ratio_count'],
        ego_responsibility_burden=base['ego_attributed_collision_burden'],
        mean_dominance_margin=float(np.mean(margin)) if margin else None,
        class_distribution=dict(Counter(r['responsibility_class'] for r in events)),
        confidence_level_distribution=dict(Counter(r['responsibility_confidence_level'] for r in events)),
        attribution_strength_distribution=dict(Counter(r['attribution_strength'] for r in valid)))
    for side in ('ego','other'):
        windows={(r.get('scene_id'),r.get('agent_name'),r.get('decision_frame_id')) for r in ego if r['ego_responsibility_class']==side+'_dominant'}
        result[side+'_dominant_collision_rate']=len(windows)/total_windows if total_windows else None
    result['dominant_rate_denominator']='unique scored windows containing at least one event of that direction / all agent windows'
    return result


def _compact(row):
    return {key:value for key,value in row.items() if not isinstance(value,(dict,list))}


def _stability(rows, event_id, family):
    valid=[r for r in rows if r['responsibility_status']=='valid']
    directional=[r['dominant_responsible_actor'] for r in valid if r['dominant_responsible_actor'] is not None]
    pairs=[(a,b) for i,a in enumerate(directional) for b in directional[i+1:]]
    counts=Counter(r['dominant_responsible_actor'] or 'shared_equal' for r in valid)
    flips=sum(a!=b for a,b in pairs)
    return dict(event_id=event_id,experiment_family=family,config_count=len(rows),finite_config_count=len(valid),
        direction_available_count=len(directional),shared_equal_configs=len(valid)-len(directional),
        dominant_pair_count=len(pairs),dominant_flip_pair_count=flips,
        dominant_actor_flip_rate=flips/len(pairs) if pairs else None,
        responsibility_ratio_std=float(np.std([r['actor_a_responsibility_ratio'] for r in valid])) if valid else None,
        responsibility_direction_stability=max(counts.values())/len(valid) if valid else None,
        stability_definition='modal attribution frequency / finite configs; flip rate compares A vs B pairs only; exact ties counted separately')


def assemble(v2,config=None):
    from driveaudit_eval.responsibility.evidence import rescore
    cfg=_config(config)
    events=[classify(row,cfg) for row in v2['events']]
    result=dict(status='completed',method='Responsibility V3 mandatory model-based evidence attribution',config=cfg,events=events)
    result['comparison']=[dict(event_id=a['event_id'],ratio_a_v2=a.get('actor_a_responsibility_ratio'),ratio_a_v3=b['actor_a_responsibility_ratio'],
        ratio_b_v2=a.get('actor_b_responsibility_ratio'),ratio_b_v3=b['actor_b_responsibility_ratio'],
        class_v2=a['responsibility_class'],class_v3=b['responsibility_class'],confidence=b['responsibility_confidence'],
        dominance_margin=b['responsibility_dominance_margin']) for a,b in zip(v2['events'],events)]
    result['agent_summary']=[dict(agent_name=m['agent_name'],**summarize([r for r in events if r.get('agent_name')==m['agent_name']],m['total_windows'])) for m in v2['agent_summary']]
    result['scene_summary']=[dict(scene_id=s['scene_id'],agent_name=s.get('agent_name'),scope=s['scope'],
        **summarize([r for r in events if r['scene_id']==s['scene_id'] and r.get('agent_name')==s.get('agent_name')],s.get('total_windows'))) for s in v2['scene_summary']]
    result['overall']=summarize(events,sum(r['total_windows'] for r in result['agent_summary']))
    result['overall']['background_class_distribution']=dict(Counter(r['responsibility_class'] for r in events if not r.get('ego_involved')))
    result['direction_stability']=[]
    original_by_id={r['event_id']:r for r in v2['events']}
    for family in ('weight_sensitivity','window_sensitivity','ablation','size_sensitivity'):
        result[family]=[]
        for row in v2.get(family,[]):
            # Compact V2 sweeps retain flat scores but omit raw vote/coverage structures.
            # Carry immutable votes, never the original ratio or window-specific coverage.
            original=original_by_id[row['event_id']]
            enriched=dict(row)
            enriched['scoring_weights']=row.get('weights',v2['config'].get('weights',{}))
            if family=='ablation':
                name=row['experiment']
                dimensions=set(enriched['scoring_weights']) if name.startswith('Full Basic') else set(name.removesuffix(' only').lower().split(' + '))
                enriched['scoring_weights']={k:w if k in dimensions else 0. for k,w in enriched['scoring_weights'].items()}
                enriched['cars_fusion_enabled']=name=='Full Basic + CARS'
            for key in ('cars_actor_a_votes','cars_actor_b_votes'):
                if key not in enriched:enriched[key]=original.get(key)
            result[family].append(dict(_compact(classify(enriched,cfg)),experiment=row['experiment']))
        if family in ('weight_sensitivity','window_sensitivity'):
            result['direction_stability'].extend(_stability([r for r in result[family] if r['event_id']==e['event_id']],e['event_id'],family) for e in events)
    for event in events:
        rows=[r for family in ('weight_sensitivity','window_sensitivity') for r in result[family] if r['event_id']==event['event_id']]
        combined=_stability(rows,event['event_id'],'combined_weight_and_window')
        event['responsibility_direction_stability']=combined['responsibility_direction_stability']
        result['direction_stability'].append(combined)
    result['cars_comparison']=[]
    for old,after in zip(v2['events'],events):
        before=classify(rescore(old,v2['config'],use_cars=False),cfg)
        result['cars_comparison'].append(dict(event_id=after['event_id'],cars_status=after['cars_status'],
            ratio_before=before['actor_a_responsibility_ratio'],ratio_after=after['actor_a_responsibility_ratio'],
            dominant_before=before['dominant_responsible_actor'],dominant_after=after['dominant_responsible_actor'],
            dominant_actor_flip=before['dominant_responsible_actor'] in ('actor_a','actor_b') and after['dominant_responsible_actor'] in ('actor_a','actor_b') and before['dominant_responsible_actor']!=after['dominant_responsible_actor']))
    values=[r['ego_responsibility_ratio'] for r in events if _finite(r.get('ego_responsibility_ratio'))]
    result['ratio_distribution']=dict(count=len(values),mean=float(np.mean(values)) if values else None,std=float(np.std(values)) if values else None)
    result['ratio_distribution'].update({f'p{q}':float(np.percentile(values,q)) if values else None for q in (10,25,50,75,90)})
    result['ratio_distribution'].update(min=min(values) if values else None,max=max(values) if values else None)
    result['confidence_summary']=[dict(confidence_level=level,**summarize([r for r in events if r['responsibility_confidence_level']==level])) for level in ('high','medium','low','unavailable')]
    result['strength_summary']=[dict(attribution_strength=level,**summarize([r for r in events if r['attribution_strength']==level])) for level in ('very_strong','strong','moderate','weak')]
    result['overall']['cars_dominant_direction_flips']=sum(r['dominant_actor_flip'] for r in result['cars_comparison'] if r['cars_status']=='valid')
    for family in ('weight_sensitivity','window_sensitivity'):
        rows=[r for r in result['direction_stability'] if r['experiment_family']==family]
        pairs=sum(r['dominant_pair_count'] for r in rows)
        result['overall'][family+'_dominant_flip_rate']=sum(r['dominant_flip_pair_count'] for r in rows)/pairs if pairs else None
    return result

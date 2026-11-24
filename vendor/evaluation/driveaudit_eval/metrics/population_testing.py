"""Paper Eq. QTES: four local retentions and one matched population TRS.

This explicit version leaves the historical gt_relative_v1 API unchanged.
Numeric invalid-scene contributions require completed measurement records;
missing execution or an unsupported population TRS never becomes a score.
"""
from copy import deepcopy
import math
from pathlib import Path
from statistics import mean

from .quality_retention import QualityBaselineCache, canonicalize, reliable_coverage, EXPORT_NAMES
from .testing_effectiveness import content_hash, unit, testing_effectiveness
from driveaudit_eval.research.qtes_population import _confirmed_invalid, _hash

VERSION = 'paper_population_v1'
FIELDS = ('clip_iqa', 'dino_t', 'epi_3px', 'tvc')


class LocalQualityCache(QualityBaselineCache):
    SCHEMA = 'qtes_paper_local_quality_v1'
    FIELDS = FIELDS


def local_retention(generated, reference, epsilons=None):
    """Exact paper retention; valid zeros use the declared positive epsilon."""
    identity = ('dataset','scene_id')
    if any(generated[k] != reference[k] for k in identity):
        raise ValueError('Local quality pairing mismatch')
    if any(row.get('computation_status') != 'completed' for row in (generated, reference)):
        raise ValueError('Local quality execution is missing or incomplete')
    result = {k:reference[k] for k in identity}
    result.update(invalid_reason=[], dimension_status={}, computation_status='completed')
    for key in FIELDS:
        epsilon = (epsilons or {}).get(key,1e-6)
        if isinstance(epsilon,bool) or not isinstance(epsilon,(int,float)) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError('Invalid quality epsilon: '+key)
        values=[]
        for side,row in (('generated',generated),('gt',reference)):
            try:
                space=row.get('score_space','raw')
                value=unit(row['scores'].get(key),key,-1. if key=='dino_t' and space=='raw' else 0.)
                value=canonicalize(value,key,space)
                if row.get('dimension_status',{}).get(key) != 'valid' or not reliable_coverage(row,key):
                    raise ValueError('insufficient measured support')
                values.append(value)
            except (ValueError,KeyError) as exc:
                values.append(None)
                result['invalid_reason'].append(key+':'+side+':'+str(exc))
        ratio = min(1.,values[0]/max(values[1],epsilon)) if all(v is not None for v in values) else None
        if key == 'tvc':
            masks=[r.get('tvc_component_mask') for r in (generated,reference)]
            from .core import TVC_FIELDS
            if (not all(isinstance(mask,list) and mask and len(mask)==len(set(mask))
                        and set(mask)<=set(TVC_FIELDS) for mask in masks)
                    or set(masks[0]) != set(masks[1])):
                result['invalid_reason'].append('tvc:component_mask_mismatch_or_missing')
                ratio=None
            result['tvc_component_mask']=sorted(masks[0]) if masks[0] else []
        if key == 'dino_t':
            pairs=[_temporal_support(r) for r in (generated,reference)]
            if not pairs[0] or pairs[0] != pairs[1]:
                result['invalid_reason'].append('dino_t:paired_native_intervals_mismatch_or_missing')
                ratio=None
        name=EXPORT_NAMES[key]
        result.update({name+'_generated':values[0],name+'_gt':values[1],name+'_retention':ratio})
        result['dimension_status'][key]='valid' if ratio is not None else 'invalid'
    ratios=[result[EXPORT_NAMES[k]+'_retention'] for k in FIELDS]
    result['B'] = math.prod(ratios[:3])**(1/6)*ratios[3]**(1/4) if all(v is not None for v in ratios) else None
    result['quality_retention_score']=result['B']
    result['quality_retention_coverage']=sum(v is not None for v in ratios)/4
    for name,record in [('generated',generated),('reference',reference)]:
        if all(k in record for k in LocalQualityCache.IDENTITY_FIELDS):
            result.setdefault('evidence_identity',{})[name]={k:record[k] for k in LocalQualityCache.IDENTITY_FIELDS}
    return result


def _temporal_support(record):
    """Preserve exact camera/endpoint/native interval support, including duplicates."""
    rows=record.get('temporal_pairs')
    if not isinstance(rows,list):
        raise ValueError('Missing executed DINO temporal pair evidence')
    result={}
    for row in rows:
        from driveaudit_eval.testing import scene_key
        if 'scene_id' in row and scene_key(row['scene_id'])!=scene_key(record['scene_id']):
            raise ValueError('Foreign scene in DINO temporal evidence')
        key=tuple(row[k] for k in ('camera','frame_id','next_frame_id','interval'))
        if key in result:
            raise ValueError('Duplicate DINO temporal pair')
        actual,requested=(row.get(k) for k in ('actual_delta_s','requested_delta_s'))
        interval={'short':.3,'medium':1.,'long':2.5}.get(key[-1])
        if (not isinstance(key[0],str) or not key[0] or any(type(v) is not int for v in key[1:3])
                or key[1]>=key[2] or interval is None
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (actual,requested))
                or not math.isclose(requested,interval,rel_tol=0.,abs_tol=1e-12)
                or abs(actual-requested)>.08+1e-12):
            raise ValueError('Invalid DINO native interval evidence')
        result[key]=(actual,requested)
    return result


def aggregate(population, comparison, quality, expected_scenes):
    """Mean(V*TE*B), never mean(TE)*mean(B); no scene is silently dropped."""
    from driveaudit_eval.testing import scene_key
    expected=sorted(scene_key(s) for s in expected_scenes)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty or duplicate QTES population')
    actual=sorted(scene_key(s) for s in population.get('scene_ids',[]))
    if actual != expected or population.get('status','').lower() not in ('available','valid','completed','complete'):
        raise ValueError('TRS population unavailable or does not match QTES selection')
    trs=unit(population.get('TRS'),'population_TRS')
    if population.get('population_hash') != content_hash(sorted(population['scene_ids'])):
        raise ValueError('TRS population hash mismatch')
    snapshot=_hash(population.get('trajectory_snapshot_hash'),'trajectory_snapshot_hash')
    source=population.get('source_manifest')
    if not isinstance(source,dict) or sorted(source)!=expected or snapshot!=content_hash({sid:source[sid]['candidate'] for sid in expected}):
        raise ValueError('TRS trajectory source manifest hash mismatch')
    if _hash(population.get('reference_snapshot_hash'),'reference_snapshot_hash')!=content_hash({sid:source[sid]['reference'] for sid in expected}):
        raise ValueError('TRS reference source manifest hash mismatch')
    for scene in source.values():
        for side in ('candidate','reference'):
            if not {'timeline.json','generation_input.json','history/agents.csv','future/future_trajectories.csv'}<=set(scene[side]):
                raise ValueError('TRS input identity incomplete')
            for digest in scene[side].values():
                _hash(digest,'TRS source input hash')
    def index(rows,key):
        indexed={scene_key(row[key]):row for row in rows}
        if len(indexed)!=len(rows) or sorted(indexed)!=expected:
            raise ValueError('Missing/duplicate local scene measurements')
        return indexed
    qualities=index(quality['scenes'],'scene_id')
    effects=index(comparison['scenes'],'scene')
    known_quality_errors={r['scene_id']+':'+reason for r in quality['scenes'] for reason in r.get('invalid_reason',[])}
    if set(quality.get('invalid_reason',[]))-known_quality_errors:
        raise ValueError('Quality contains structural or unaccounted invalid reasons')
    if comparison.get('input_identity') is not None and sorted(comparison['input_identity'])!=expected:
        raise ValueError('Policy input identity population mismatch')
    pair_rows=comparison['rows']
    from driveaudit_eval.agents.registry import FORMAL_AGENTS
    pair_keys=[(scene_key(row['scene']),row['agent']) for row in pair_rows]
    if len(set(pair_keys))!=len(pair_keys) or set(pair_keys)!={(sid,a) for sid in expected for a in FORMAL_AGENTS}:
        raise ValueError('Missing, duplicate or foreign six-policy pair rows')
    pair_errors={str((p['dataset'],p['scene'],p['agent']))+':'+reason
                 for p in pair_rows for reason in p.get('invalid_reason',[])}
    if set(comparison.get('invalid_reason',[]))-pair_errors:
        raise ValueError('Comparison contains structural or unaccounted invalid reasons')
    if comparison.get('validity')!=1 and not pair_errors:
        raise ValueError('Comparison is invalid without explained measured failures')
    terms=[];rows=[]
    for sid in expected:
        q,t=qualities[sid],effects[sid]
        if 'V' in q and (type(q['V']) is not int or q['V'] not in (0,1)):
            raise ValueError('Scene validity must be an explicit zero or one')
        pairs=[p for p in pair_rows if scene_key(p['scene'])==sid]
        if q.get('dataset') != t.get('dataset') or any(p.get('dataset') != q.get('dataset') for p in pairs):
            raise ValueError('Dataset mismatch between local quality and six-policy evidence')
        if q.get('V') == 0:
            path,digest=_confirmed_invalid(dict(q,scene_id=sid),population)
            rows.append(dict(scene_id=sid,V=0,TE=None,B=None,contribution=0.,
                invalid_reason=q.get('invalid_reason',['confirmed_invalid_submission']),
                invalid_submission_evidence_path=path,invalid_submission_evidence_sha256=digest))
            terms.append(0.)
            continue
        if (len(pairs)!=6 or {p['agent'] for p in pairs}!=set(FORMAL_AGENTS)
                or q.get('computation_status')!='completed'):
            raise ValueError('Missing executed six-policy/local-quality evidence')
        # Pairing/audit failures are not a measured degradation and cannot be imputed.
        if any(type(p.get('validity')) is not int or p['validity'] != 1 or p.get('invalid_reason') for p in pairs):
            raise ValueError('Policy pair not completely measured: '+sid)
        for p in pairs:
            calculated=testing_effectiveness(p.get('rpdms_gt'),p.get('rpdms_generated'),comparison.get('eps',1e-6))
            if not math.isclose(calculated,unit(p.get('testing_effectiveness'),'pair_TE'),rel_tol=0.,abs_tol=1e-12):
                raise ValueError('Pair TE differs from measured RPDMS')
        te=unit(t.get('testing_effectiveness'),'scene_TE')
        if not math.isclose(te,mean(unit(p['testing_effectiveness'],'pair_TE') for p in pairs),rel_tol=0.,abs_tol=1e-12):
            raise ValueError('TE differs from six-policy mean')
        if source is not None and (q.get('trajectory_inputs')!=source[sid]['candidate']
                                  or q.get('reference_trajectory_inputs')!=source[sid]['reference']):
            raise ValueError('Quality and TRS input identities disagree')
        if comparison.get('input_identity') is not None:
            for side in ('generated','reference'):
                actual=q.get('evidence_identity',{}).get(side,{})
                measured=comparison['input_identity'].get(sid,{}).get(side,{})
                if any(_hash(actual.get(k),'quality '+k)!=_hash(measured.get(k),'policy '+k)
                       for k in ('scene_hash','rgb_hash')):
                    raise ValueError('Quality and policy input identities disagree')
        reasons=list(q['invalid_reason'])
        if reasons and (q.get('B') is not None or all(q.get('dimension_status',{}).get(k)=='valid' for k in FIELDS)):
            raise ValueError('Local invalidity disagrees with component evidence')
        b=unit(q['B'],'B') if not reasons else None
        if not reasons:
            retentions=[unit(q.get(EXPORT_NAMES[k]+'_retention'),k+'_retention') for k in FIELDS]
            calculated=math.prod(retentions[:3])**(1/6)*retentions[3]**(1/4)
            if (any(q.get('dimension_status',{}).get(k)!='valid' for k in FIELDS)
                    or not math.isclose(b,calculated,rel_tol=0.,abs_tol=1e-12)):
                raise ValueError('B differs from valid measured four-component retention')
        valid=int(not reasons)
        contribution=te*b if valid else 0.
        rows.append(dict(scene_id=sid,V=valid,TE=te,B=b,contribution=contribution,invalid_reason=reasons))
        terms.append(contribution)
    score=100*trs**.25*mean(terms)
    return dict(qtes=score,testing_effectiveness=mean(r['TE'] for r in rows) if all(r['TE'] is not None for r in rows) else None,
        quality_retention_score=trs**.25*mean(r['B'] if r['V'] else 0. for r in rows),
        qtes_definition_version=VERSION,validity=1,status='valid',invalid_reason=[],
        invalid_scene_count=sum(1-r['V'] for r in rows),scene_count=len(rows),scenes=rows,
        population_TRS=trs,population_hash=population['population_hash'],
        trajectory_snapshot_hash=population['trajectory_snapshot_hash'],
        aggregation='100 * TRS_population**0.25 * equal_scene_mean(V*TE*B)',
        uncertainty_status='not_computed; joint source-recording bootstrap required for confidence intervals')


def evaluate_local_quality(summary,output,config,context,gt_scenes,generated_scenes,gt_evidence,generated_evidence):
    from driveaudit_eval.quality import scene_quality,quality_signature,quality_identity,evaluate_quality_only
    from driveaudit_eval.utils.io import write_json
    options=config['evaluation']['testing_effectiveness'].get('quality_retention',{})
    cache=LocalQualityCache(Path(context['root'])/options.get('baseline_cache','cache/paper_gt_local_quality.json')
                            if options.get('retain_intermediates',True) else None)
    if not gt_scenes or any(set(values)!=set(gt_scenes) for values in (generated_scenes,gt_evidence,generated_evidence)):
        raise ValueError('Local quality input population mismatch')
    _require_execution(summary, generated_scenes)
    generated=scene_quality(summary,output,generated_scenes,fields=FIELDS)
    signature=quality_signature(config)
    reference={};pending={};hits=0
    for sid,scene in gt_scenes.items():
        generated[sid].update(quality_identity(config,generated_scenes[sid],generated_evidence[sid],signature),
                              computation_status='completed')
        identity=quality_identity(config,scene,gt_evidence[sid],signature)
        record=cache.get(identity)
        if record is not None:
            hits+=1
        elif gt_evidence[sid]['scene_hash']==generated_evidence[sid]['scene_hash']:
            record=deepcopy(generated[sid])
            record.update(identity,source='identical_measured_GT_input',computation_status='completed')
            cache.put(record)
        else:
            pending[sid]=identity
            continue
        reference[sid]=record
    if pending:
        # Keep scene kernels and evidence unchanged; expose all unresolved scenes
        # to the existing bounded scene scheduler in one evaluator invocation.
        work=Path(output)/'gt_quality'/'batch'
        cfg=deepcopy(config)
        cfg['evaluation']['consistency']['candidate_role']='ground_truth'
        selected={sid:gt_scenes[sid] for sid in pending}
        if context.get('logger'):
            context['logger'].info('[QTES] Computing GT local quality batch on %d scene(s)',len(selected))
        measured=evaluate_quality_only(cfg,[scene['scene_path'] for scene in selected.values()],work,context)
        _require_execution(measured, selected)
        records=scene_quality(measured,work,selected,fields=FIELDS)
        for sid,identity in pending.items():
            record=records[sid]
            record.update(identity,source=str(work),computation_status='completed')
            cache.put(record)
            reference[sid]=record
    rows=[local_retention(generated[sid],reference[sid],options.get('epsilons')) for sid in sorted(generated)]
    from driveaudit_eval.agents.provenance import digest
    population=summary['core_metrics']['trs_population']
    for row in rows:
        sid=row['scene_id']
        for side,scenes,field in [('candidate',generated_scenes,'trajectory_inputs'),
                                  ('reference',gt_scenes,'reference_trajectory_inputs')]:
            inputs=population['source_manifest'][sid][side]
            if any(digest(Path(scenes[sid]['scene_path'])/name)!=value for name,value in inputs.items()):
                raise ValueError('Quality/policy input changed since TRS extraction: '+sid)
            row[field]=dict(inputs)
    result=dict(scenes=rows,invalid_reason=[r['scene_id']+':'+v for r in rows for v in r['invalid_reason']],
        qtes_definition_version=VERSION,cache_hits=hits,cache_misses=len(reference)-hits,
        quality_retention_score=mean(r['B'] for r in rows) if all(r['B'] is not None for r in rows) else None,
        quality_retention_coverage=mean(r['quality_retention_coverage'] for r in rows))
    write_json(Path(output)/'gt_quality_baseline.json',dict(schema=cache.SCHEMA,qtes_definition_version=VERSION,
        entries={cache.key(r):dict(record=r,sha256=content_hash(r)) for r in reference.values()}))
    return result


def _require_execution(summary, expected_scenes=None):
    """Scientific low support is allowed; disabled/missing/failed execution is not."""
    video=summary.get('video',{})
    metrics=video.get('visual_quality',{}).get('metrics',{})
    completed={'valid','valid_low_sample','valid_low_coverage','completed','available','insufficient_samples'}
    for value in [metrics.get('clipiqa',{}),metrics.get('dino_temporal',{}),video.get('geometry_consistency',{})]:
        if value.get('skipped') or str(value.get('status','')).lower() not in completed:
            raise ValueError('Required local video metric execution incomplete')
    scenes=summary.get('consistency',{}).get('scenes',[])
    if not scenes or any(s.get('backends',{}).get('mtpc',{}).get('execution_status')!='completed' for s in scenes):
        raise ValueError('Required local TVC execution incomplete')
    if expected_scenes is not None:
        from driveaudit_eval.testing import scene_key
        actual=[scene_key(scene.get('scene_id',scene.get('scene'))) for scene in scenes]
        expected=[scene_key(sid) for sid in expected_scenes]
        if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
            raise ValueError('Required local TVC scene execution support mismatch')

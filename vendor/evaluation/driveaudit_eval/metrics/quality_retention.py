"""GT-relative quality retention. Ratios are formed before scene aggregation."""
import math
from statistics import mean
from driveaudit_eval.utils.io import jsonable

from driveaudit_eval.metrics.testing_effectiveness import BaselineCache, QUALITY_FIELDS, content_hash

QUALITY_VERSION = 'gt_relative_v1'
EPSILONS = dict.fromkeys(QUALITY_FIELDS, 1e-6)
GROUPS = {'visual_quality': QUALITY_FIELDS[:3], 'scene_fidelity': QUALITY_FIELDS[3:]}
from driveaudit_eval.metrics.core_metric_gate import REQUIREMENTS
MIN_COVERAGE = {key: REQUIREMENTS[key][1] for key in QUALITY_FIELDS}

EXPORT_NAMES = dict(zip(QUALITY_FIELDS, ('clip_iqa','dino_t','epi3','tvc','trs')))


def canonicalize(value, metric, score_space='raw'):
    if metric not in QUALITY_FIELDS or score_space not in ('raw', 'canonical'):
        raise ValueError('Unknown metric or score space')
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(metric + ': missing or nonfinite')
    if metric == 'dino_t' and score_space == 'raw':
        value = (value+1)/2
    return max(0., min(1., float(value)))


def reliable_coverage(row, metric):
    coverage = row.get('metric_coverage',{}).get(metric)
    return (not isinstance(coverage,bool) and isinstance(coverage,(int,float))
            and math.isfinite(coverage) and MIN_COVERAGE[metric] <= coverage <= 1)


def reference_audit(records, epsilons=None):
    """Empirical minima are audit triggers, never score offsets or normalizers.

    Zero/invalid/insufficiently observed references are excluded. A low but positive,
    coverage-qualified score remains usable after the same reliability checks.
    """
    eps = dict(EPSILONS, **(epsilons or {}))
    dimensions = {}
    for metric in QUALITY_FIELDS:
        values = []
        excluded = []
        for row in records:
            try:
                value = canonicalize(row['scores'].get(metric),metric,row.get('score_space','raw'))
                reliable = (row.get('dimension_status',{}).get(metric) == 'valid'
                            and reliable_coverage(row,metric) and value > eps[metric])
                if not reliable:
                    raise ValueError('invalid, incomplete or numerically zero reference')
                values.append(value)
            except ValueError as exc:
                excluded.append(dict(dataset=row['dataset'],scene_id=row['scene_id'],reason=str(exc)))
        dimensions[metric] = dict(reference_floor=min(values) if values else None,
            n_valid=len(values), minimum=min(values) if values else None,
            maximum=max(values) if values else None, excluded=excluded)
    return dict(method='minimum_positive_coverage_qualified_GT_validation_score',
                scope='Frozen GT validation records only; no generated model enters floor selection',
                dimensions=dimensions, epsilons=eps,
                validation_records_hash=content_hash(jsonable(records)))


def motion_guard(gt_hashes, generated_hashes, pairs):
    """Exact decoded-RGB repeats on native-time pairs where GT actually changes.

    Native repeated exposures/static GT do not count as collapses. No optical
    flow threshold is needed. This detects exact freezes, not every slow motion.
    """
    observable = collapsed = 0
    seen = set()
    for pair in pairs:
        key = (pair['camera'],pair['frame_id'],pair['next_frame_id'])
        if key in seen:
            continue
        seen.add(key)
        left,right = (str(pair['camera'])+'/'+str(pair[k]) for k in ('frame_id','next_frame_id'))
        if any(k not in hashes for k in (left,right) for hashes in (gt_hashes,generated_hashes)):
            return dict(status='invalid',reason='missing_decoded_RGB_pair',motion_collapse_ratio=None)
        if gt_hashes[left] != gt_hashes[right]:
            observable += 1
            collapsed += int(generated_hashes[left] == generated_hashes[right])
    return dict(status='valid' if observable else 'reference_unobservable',
                motion_collapse_ratio=collapsed/observable if observable else None,
                observable_pairs=observable, collapsed_pairs=collapsed, evaluated_pairs=len(seen),
                method='paired_exact_decoded_repeat_v1')


def quality_retention(generated, gt, audit, epsilons=None, guard=None):
    if (generated['dataset'],generated['scene_id']) != (gt['dataset'],gt['scene_id']):
        raise ValueError('Quality scene pairing mismatch')
    eps = dict(EPSILONS, **(epsilons or {}))
    result = dict(dataset=gt['dataset'],scene_id=gt['scene_id'],qtes_definition_version=QUALITY_VERSION,
                  dimension_status={},reference_checks={},invalid_reason=[],motion_guard=guard)
    ratios = {}
    for metric in QUALITY_FIELDS:
        if not math.isfinite(eps[metric]) or not 0 < eps[metric] < 1:
            raise ValueError('Invalid epsilon: '+metric)
        floor = audit['dimensions'][metric]['reference_floor']
        if floor is not None and (not isinstance(floor,(int,float)) or not math.isfinite(floor) or not 0 < floor <= 1):
            raise ValueError('Invalid GT validation reference floor: '+metric)
        values = []
        for row in (generated,gt):
            try:
                values.append(canonicalize(row['scores'].get(metric),metric,row.get('score_space','raw')))
            except ValueError:
                values.append(None)
        gen, ref = values
        reference_valid = (ref is not None and ref > eps[metric]
            and gt.get('dimension_status',{}).get(metric) == 'valid'
            and reliable_coverage(gt,metric))
        temporal_unobservable = metric == 'dino_t' and guard and guard.get('status') == 'reference_unobservable'
        if temporal_unobservable:
            reference_valid = False
        result['reference_checks'][metric] = dict(reference_floor=floor,eps=eps[metric],
            below_reference_floor=ref is not None and floor is not None and ref < floor,
            reference_status=gt.get('dimension_status',{}).get(metric),
            reference_coverage=gt.get('metric_coverage',{}).get(metric), reliable=reference_valid)
        if temporal_unobservable:
            result['reference_checks'][metric]['reason'] = 'GT_temporal_motion_unobservable'
        status = 'valid'
        ratio = None
        if not reference_valid:
            status = 'reference_invalid'
        elif (gen is None or generated.get('dimension_status',{}).get(metric) != 'valid'
              or not reliable_coverage(generated,metric)):
            status = 'generated_invalid'
            result['invalid_reason'].append(metric+':generated_invalid')
        else:
            ratio = min(1.,max(0.,gen/max(ref,eps[metric])))
        if metric == 'dino_t' and reference_valid:
            if not guard or guard.get('status') != 'valid':
                ratio = None
                status = 'generated_invalid'
                result['invalid_reason'].append('dino_t:missing_or_invalid_motion_guard')
            elif ratio is not None:
                collapse = guard['motion_collapse_ratio']
                if not isinstance(collapse,(int,float)) or not math.isfinite(collapse) or not 0 <= collapse <= 1:
                    raise ValueError('Invalid motion collapse ratio')
                result['dino_t_retention_before_guard'] = ratio
                ratio *= 1-collapse
        ratios[metric] = ratio
        name = EXPORT_NAMES[metric]
        result.update({name+'_generated':gen,name+'_gt':ref,name+'_retention':ratio})
        result['dimension_status'][metric] = status
    for group, metrics in GROUPS.items():
        valid = [ratios[m] for m in metrics if ratios[m] is not None]
        result[group+'_retention_coverage'] = len(valid)/len(metrics)
        result[group+'_retention_score'] = math.prod(valid)**(1/len(valid)) if valid else None
        if not valid:
            result['invalid_reason'].append(group+':no_valid_dimensions')
    vq,sf = (result[g+'_retention_score'] for g in GROUPS)
    result['quality_retention_coverage'] = mean(result[g+'_retention_coverage'] for g in GROUPS)
    result['quality_retention_score'] = math.sqrt(vq*sf) if not result['invalid_reason'] else None
    return result


def aggregate_retention(rows):
    if not rows or len({(r['dataset'],r['scene_id']) for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate quality scene pairing')
    reasons = [r['scene_id']+':'+reason for r in rows for reason in r['invalid_reason']]
    fields = [name+'_'+suffix for name in EXPORT_NAMES.values() for suffix in ('generated','gt','retention')]
    fields += [g+'_retention_'+suffix for g in (*GROUPS,'quality') for suffix in ('score','coverage')]
    result = {}
    for key in fields:
        values = [row[key] for row in rows if row.get(key) is not None]
        # Dimension diagnostics average available scene values and expose scene counts.
        # Final Q never discards an invalid scene.
        result[key] = mean(values) if values else None
    if reasons:
        result['quality_retention_score'] = None
    result.update(scenes=rows, invalid_reason=reasons,qtes_definition_version=QUALITY_VERSION,
                  aggregation='strict_scene_pairs_then_scene_equal_mean_of_Q_rel',
                  dimension_scene_counts={name:sum(r[name+'_retention'] is not None for r in rows)
                                          for name in EXPORT_NAMES.values()})
    return result


class QualityBaselineCache(BaselineCache):
    SCHEMA = 'qtes_gt_quality_baseline_v1'
    FIELDS = QUALITY_FIELDS
    IDENTITY_FIELDS = ('dataset','scene_id','scene_hash','rgb_hash','manifest_hash',
                       'metric_implementation_hash','metric_config_hash')

    @classmethod
    def record_errors(cls, row):
        reasons = ['missing_'+k for k in QualityBaselineCache.IDENTITY_FIELDS if not row.get(k)]
        reasons += ['invalid_'+k for k in QualityBaselineCache.IDENTITY_FIELDS if k.endswith('_hash') and
                    (not isinstance(row.get(k),str) or len(row[k]) != 64 or any(c not in '0123456789abcdef' for c in row[k]))]
        if not all(k in row.get('scores',{}) for k in cls.FIELDS):
            reasons.append('missing_quality_dimensions')
        return reasons

    @staticmethod
    def key(record):
        return content_hash({k:record[k] for k in QualityBaselineCache.IDENTITY_FIELDS})

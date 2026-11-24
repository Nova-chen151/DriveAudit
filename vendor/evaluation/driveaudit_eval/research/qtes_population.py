"""CPU-only population QTES candidate and read-only local evidence audit.

This module never dispatches evaluators, inference, cache construction or exports
to the formal leaderboard. The arithmetic interface requires the complete frozen
population; the availability audit does not turn missing measurements into scores.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from driveaudit_eval.agents.registry import FORMAL_AGENTS
from driveaudit_eval.metrics.testing_effectiveness import (
    content_hash, paired_comparison, unit,
)


VERSION = 'population_qtes_fixed_local_exponents_candidate_v1'
PROFILE = 'common6_geometric_excess_candidate_v1'
LOCAL_FIELDS = ('q_iqa', 'q_dino', 'q_epi', 'q_tvc')
TRAJECTORY_FIELDS = ('video_generation_trajectory_hash',
                     'quality_trajectory_hash', 'rpdms_environment_trajectory_hash')
VIDEO_FIELDS = ('quality_rgb_hash', 'agent_input_rgb_hash', 'rpdms_input_rgb_hash')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _hash(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Missing or invalid ' + name)
    return value


def local_factor(row):
    """Only four valid, supported local retentions admit these fixed exponents."""
    if any(row.get('component_status', {}).get(k) != 'valid' for k in LOCAL_FIELDS):
        raise ValueError('MISSING_OR_UNSUPPORTED_LOCAL_MEASUREMENT')
    iqa, dino, epi, tvc = [unit(row.get(k), k) for k in LOCAL_FIELDS]
    return (iqa * dino * epi) ** (1 / 6) * tvc ** (1 / 4)


def validate_alignment(row, population):
    """Bind every generated-world stage to one trajectory and one RGB bundle.

    Hashes are verified against files by the caller/audit, not inferred from a
    method label. GT RGB/state may differ and are paired through frozen TE fields.
    """
    expected = _hash(population['trajectory_snapshot_hash'], 'trajectory_snapshot_hash')
    for field in TRAJECTORY_FIELDS:
        if _hash(row.get(field), field) != expected:
            raise ValueError('TRAJECTORY_HASH_MISMATCH:' + field)
    rgb = _hash(row.get('rgb_hash'), 'rgb_hash')
    for field in VIDEO_FIELDS:
        if _hash(row.get(field), field) != rgb:
            raise ValueError('VIDEO_HASH_MISMATCH:' + field)


def _confirmed_invalid(row, population):
    """A saved, hash-checked validation failure is different from no measurement."""
    path = row.get('invalid_submission_evidence_path')
    expected = _hash(row.get('invalid_submission_evidence_sha256'), 'invalid_submission_evidence_sha256')
    if not path or not Path(path).is_file() or sha256(path) != expected:
        raise ValueError('CONFIRMED_INVALID_EVIDENCE_MISSING_OR_HASH_MISMATCH')
    receipt = read(path)
    if (receipt.get('status') != 'CONFIRMED_INVALID_SUBMISSION'
            or receipt.get('observed_output_present') is not True
            or receipt.get('failure_type') not in ('generated_rgb_validation_failed', 'submission_protocol_violation')
            or receipt.get('scene_id') != row['scene_id']
            or receipt.get('trajectory_snapshot_hash') != population['trajectory_snapshot_hash']):
        raise ValueError('CONFIRMED_INVALID_EVIDENCE_DOES_NOT_IDENTIFY_OBSERVED_SUBMISSION_FAILURE')
    return str(Path(path).resolve()), expected


def aggregate_population(population, scene_rows, expected_scenes):
    """Aggregate already measured rows, retaining exactly the declared support.

    expected_scenes contains frozen TRS scene IDs; population_hash is their sorted
    content_hash. TE must be accompanied by six valid strict-pair result rows.
    One T object is stored; scene rows only reference population_metric_id.
    """
    expected = sorted(expected_scenes)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('EMPTY_OR_DUPLICATE_POPULATION')
    if population.get('population_hash') != content_hash(expected):
        raise ValueError('POPULATION_HASH_MISMATCH')
    actual = [r['scene_id'] for r in scene_rows]
    if len(set(actual)) != len(actual) or sorted(actual) != expected:
        raise ValueError('SCENE_SUPPORT_MISMATCH')
    t = unit(population.get('TRS_candidate'), 'TRS_candidate')
    obj = dict(population, qtes_aggregation_candidate_version=VERSION,
               trs_profile_version=population.get('trs_profile_version', PROFILE),
               geometric_policy_id=population['policy_id'], scene_count=len(expected))
    obj.setdefault('population_metric_id', content_hash(obj))
    terms, scenes = [], []
    for row in sorted(scene_rows, key=lambda r: r['scene_id']):
        validity = row.get('V')
        if type(validity) is not int or validity not in (0, 1):
            raise ValueError('MISSING_MEASUREMENT_IS_NOT_CONFIRMED_INVALID_SUBMISSION')
        if validity == 0:
            path, receipt_hash = _confirmed_invalid(row, obj)
            terms.append(0.)
            scenes.append(dict(scene_id=row['scene_id'], population_metric_id=obj['population_metric_id'],
                               V=0, TE=None, B=None, contribution=0.,
                               invalid_submission_evidence_path=path,
                               invalid_submission_evidence_sha256=receipt_hash))
            continue
        validate_alignment(row, obj)
        b = local_factor(row)
        pairs = row.get('agent_pairs', [])
        if (len(pairs) != len(FORMAL_AGENTS) or {r.get('agent') for r in pairs} != set(FORMAL_AGENTS)
                or any(r.get('validity') != 1 or r.get('invalid_reason') for r in pairs)
                or any(r.get('scene') != row['scene_id'].replace('/scenes/', '/')
                       or r.get('dataset') != row['dataset'] for r in pairs)):
            raise ValueError('MISSING_OR_INVALID_STRICT_SIX_AGENT_PAIRS')
        te = mean(unit(r.get('testing_effectiveness'), 'paired_TE') for r in pairs)
        if not math.isclose(te, unit(row.get('TE'), 'TE'), rel_tol=0., abs_tol=1e-12):
            raise ValueError('TE_DOES_NOT_MATCH_FROZEN_SIX_AGENT_AGGREGATION')
        terms.append(validity * te * b)
        scenes.append(dict(scene_id=row['scene_id'], population_metric_id=obj['population_metric_id'],
                           V=validity, TE=te, B=b))
    return dict(status='COMPUTED', QTES_candidate=100 * t ** (1 / 4) * mean(terms),
                population_metric=obj, scenes=scenes,
                aggregation='fixed_population_equal_scene_mean_of_V_TE_B',
                independent_scene_TRS_observations=False)


def analytic_self_check():
    """An explicit analytic fixture, never generated evidence or a cache fallback."""
    sid = 'analytic/self'
    population = dict(method_id='analytic_gt_self', trajectory_snapshot_hash='a' * 64,
                      population_hash=content_hash([sid]), policy_id='no_penalty', TRS_candidate=1.)
    row = dict(scene_id=sid, dataset='analytic', V=1, TE=.5, rgb_hash='b' * 64,
               component_status=dict.fromkeys(LOCAL_FIELDS, 'valid'), **dict.fromkeys(LOCAL_FIELDS, 1.),
               **dict.fromkeys(TRAJECTORY_FIELDS, 'a' * 64), **dict.fromkeys(VIDEO_FIELDS, 'b' * 64),
               agent_pairs=[dict(agent=a, dataset='analytic', scene=sid, validity=1,
                                 testing_effectiveness=.5, invalid_reason=[]) for a in FORMAL_AGENTS])
    result = aggregate_population(population, [row], [sid])
    assert result['QTES_candidate'] == 50.
    return dict(status='PASS', result=50., evidence_kind='ANALYTIC_TEST', scene_count=1,
                actual_generated_evidence=False)


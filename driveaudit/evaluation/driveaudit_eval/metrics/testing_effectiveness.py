"""Frozen TE arithmetic, absolute diagnostics, GT-relative final score and caches."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from driveaudit_eval.agents.registry import FORMAL_AGENTS

PAIR_FIELDS = ('dataset', 'scene', 'agent', 'checkpoint_hash', 'agent_config_hash',
               'camera_only_protocol_hash', 'scene_hash', 'scoring_hash',
               'conditioning_hash', 'decision_timestamps', 'decision_frames', 'request_seeds')
QUALITY_FIELDS = ('clip_iqa', 'dino_t', 'epi_3px', 'tvc', 'trs')


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def unit(value, name, lower=0., upper=1.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(name + ': missing, nonfinite or outside declared range')
    return float(value)


def testing_effectiveness(r_gt, r_generated, eps=1e-6):
    eps = unit(eps, 'eps')
    if eps == 0:
        raise ValueError('eps must be positive')
    gt = unit(r_gt, 'rpdms_gt')
    generated = unit(r_generated, 'rpdms_generated')
    if gt < eps:
        return .5 if generated < eps else 0.
    return .5 + .5 * max(-1., min(1., (gt-generated)/max(gt, eps)))


def quality_validity(values):
    reasons = []
    normalized = {}
    for key in QUALITY_FIELDS:
        try:
            value = unit(values.get(key), key, -1. if key == 'dino_t' else 0.)
            normalized[key] = max(0., min(1., (value+1)/2)) if key == 'dino_t' else value
        except ValueError as exc:
            normalized[key] = None
            reasons.append(str(exc))
    vq = sf = quality = None
    if all(normalized[key] is not None for key in QUALITY_FIELDS[:3]):
        vq = math.prod(normalized[key] for key in QUALITY_FIELDS[:3]) ** (1/3)
    if all(normalized[key] is not None for key in QUALITY_FIELDS[3:]):
        sf = math.sqrt(normalized['tvc']*normalized['trs'])
    if vq is not None and sf is not None:
        quality = math.sqrt(vq*sf)
    return dict(visual_quality_absolute_score=vq, scene_fidelity_absolute_score=sf,
                quality_absolute_score=quality,
                normalized_components=normalized, invalid_reason=reasons)


def final_score(effectiveness, quality, invalid_reason=()):
    reasons = list(invalid_reason)
    values = {}
    for key, value in [('testing_effectiveness', effectiveness), ('quality_retention_score', quality)]:
        try:
            values[key] = unit(value, key)
        except ValueError as exc:
            values[key] = None
            reasons.append(str(exc))
    valid = int(not reasons)
    # Gate explicitly: missing inputs remain null; 0 is a validity penalty, not imputation.
    score = 100*values['testing_effectiveness']*values['quality_retention_score'] if valid else 0.
    return dict(values, qtes=score, qtes_definition_version='gt_relative_v1', validity=valid, invalid_reason=list(dict.fromkeys(reasons)),
                status='valid' if valid else 'invalid')


def record_errors(row):
    reasons = []
    for key in PAIR_FIELDS:
        if key not in row or row[key] in (None, '', []):
            reasons.append('missing_' + key)
        elif key.endswith('_hash') and (not isinstance(row[key], str) or len(row[key]) != 64
                                       or any(c not in '0123456789abcdef' for c in row[key])):
            reasons.append('invalid_' + key)
    frames, times, seeds = (row.get(key) for key in ('decision_frames', 'decision_timestamps', 'request_seeds'))
    if not all(isinstance(v, list) and v for v in (frames, times, seeds)):
        reasons.append('missing_decision_support')
    elif (len(frames) != len(times) or len(times) != len(seeds)
          or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in frames+times+seeds)
          or frames != sorted(set(frames)) or times != sorted(set(times))):
        reasons.append('invalid_decision_support')
    for field in ('required_rgb_complete', 'six_camera_valid', 'timestamps_valid', 'inference_valid'):
        if row.get(field) is not True:
            reasons.append(field + '_not_verified')
    if row.get('real_lidar_used') is not False or row.get('sensor_leakage') is not False:
        reasons.append('sensor_leakage_or_missing_audit')
    if row.get('sensor_protocol') != 'camera_only':
        reasons.append('camera_only_required')
    expected_mode = 'zero' if row.get('agent') in ('goalflow', 'diffusiondrivev2') else 'not_requested'
    if row.get('lidar_mode') != expected_mode:
        reasons.append('zero_lidar_protocol_mismatch')
    try:
        unit(row.get('rpdms'), 'rpdms')
    except ValueError as exc:
        reasons.append(str(exc))
    return reasons


def paired_comparison(gt_rows, generated_rows, expected_scenes, eps=1e-6):
    expected_scenes = list(expected_scenes)
    expected = {(dataset, scene, agent) for dataset, scene in expected_scenes for agent in FORMAL_AGENTS}
    reasons = []
    if not expected or len(set(expected_scenes)) != len(expected_scenes):
        reasons.append('empty_or_duplicate_scene_selection')
    def index(rows, side):
        result = {}
        for row in rows:
            key = tuple(row.get(k) for k in ('dataset', 'scene', 'agent'))
            if key in result:
                reasons.append(side + ':duplicate_pair:' + str(key))
            result[key] = row
        if set(result) != expected:
            reasons.append(side + ':pair_set_mismatch')
        return result
    gt, generated = index(gt_rows, 'gt'), index(generated_rows, 'generated')
    rows = []
    for key in sorted(expected):
        left, right = gt.get(key, {}), generated.get(key, {})
        errors = ['gt:' + r for r in record_errors(left)] + ['generated:' + r for r in record_errors(right)]
        errors += ['pair_mismatch:' + field for field in PAIR_FIELDS if left.get(field) != right.get(field)]
        value = None
        if not errors:
            value = testing_effectiveness(left['rpdms'], right['rpdms'], eps)
        rows.append(dict(zip(('dataset', 'scene', 'agent'), key), rpdms_gt=left.get('rpdms'),
                         rpdms_generated=right.get('rpdms'), testing_effectiveness=value,
                         validity=int(not errors), invalid_reason=errors))
        reasons.extend(str(key) + ':' + r for r in errors)
    scenes = []
    for dataset, scene in expected_scenes:
        values = [r['testing_effectiveness'] for r in rows if (r['dataset'], r['scene']) == (dataset, scene)]
        value = float(np.mean(values)) if len(values) == 6 and all(v is not None for v in values) else None
        scenes.append(dict(dataset=dataset, scene=scene, testing_effectiveness=value))
    stats = dict.fromkeys(('mean', 'std', 'median', 'p10', 'p90'))
    if scenes and not reasons:
        values = [r['testing_effectiveness'] for r in scenes]
        stats = dict(mean=float(np.mean(values)), std=float(np.std(values)), median=float(np.median(values)),
                     p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)))
    return dict(rows=rows, scenes=scenes, statistics=stats, validity=int(not reasons),
                invalid_reason=list(dict.fromkeys(reasons)), aggregation='agent_equal_then_scene_equal',
                statistics_population='TE_scene; population std, linear percentiles', eps=eps)


class BaselineCache:
    """Content-addressed records; changed identity misses, changed contents fail closed."""
    SCHEMA = 'qtes_gt_baseline_v1'
    record_errors = staticmethod(record_errors)

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self.entries = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('schema') != self.SCHEMA:
                raise ValueError('GT baseline cache schema mismatch')
            self.entries = data['entries']
            for key, value in self.entries.items():
                if (value['sha256'] != content_hash(value['record']) or key != self.key(value['record'])
                        or self.record_errors(value['record'])):
                    raise ValueError('GT baseline cache integrity mismatch')

    @staticmethod
    def key(record):
        return content_hash({key: record[key] for key in PAIR_FIELDS})

    def get(self, identity):
        entry = self.entries.get(self.key(identity))
        return json.loads(json.dumps(entry['record'])) if entry else None

    def put(self, record):
        import fcntl
        import os
        import tempfile
        if self.record_errors(record):
            raise ValueError('Cannot cache invalid GT baseline: ' + ';'.join(self.record_errors(record)))
        if self.path is None:
            key = self.key(record)
            entry = dict(record=record, sha256=content_hash(record))
            if key in self.entries and self.entries[key]['sha256'] != entry['sha256']:
                raise ValueError('GT baseline identity already has different evidence')
            self.entries[key] = json.loads(json.dumps(entry))
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.entries = type(self)(self.path).entries
            key = self.key(record)
            entry = dict(record=record, sha256=content_hash(record))
            if key in self.entries and self.entries[key]['sha256'] != entry['sha256']:
                raise ValueError('GT baseline identity already has different evidence')
            self.entries[key] = entry
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as handle:
                json.dump(dict(schema=self.SCHEMA, entries=self.entries), handle, indent=2, allow_nan=False)
                handle.write('\n')
            os.replace(handle.name, self.path)

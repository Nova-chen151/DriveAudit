"""CAT-K population TRS. One fixed definition and one current result directory.

Run ``python -m driveaudit_eval.trajectory.trs --reproduce`` to verify and recompute
from sealed histograms, scene exposure and joint bootstrap draws, without models.
"""
from __future__ import annotations
import argparse
import fcntl
import tempfile
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from .trs_distribution import FEATURES, score_population
from driveaudit_eval.diagnostics.rounded_footprint import signed_distance

from driveaudit_eval.metrics.registry import TRS_POLICY
from driveaudit_eval.utils.io import file_sha256 as sha256, write_csv

VERSION = TRS_POLICY['version']
Q = TRS_POLICY['q']
ANCHOR = TRS_POLICY['anchor_excess_fraction']
ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'outputs/trs'
INPUT = ROOT / 'challenge/input-catk'


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def geometry_factor(excess):
    excess = np.asarray(excess, dtype=np.float64)
    if not np.isfinite(excess).all() or np.any((excess < 0) | (excess > 1)):
        raise ValueError('E_plus must be a finite fraction in [0,1]')
    return Q ** (excess / ANCHOR)


def sample_weights(times):
    dt = float(np.median(np.diff(times)))
    delta = np.r_[np.diff(times), dt]
    return np.where(delta <= dt*1.5, delta, dt), dt


def episode_indices(hit, valid, times):
    """Invalid samples are absent evidence and always delimit episodes."""
    indices = np.flatnonzero(hit & valid)
    if not len(indices):
        return []
    native_dt = float(np.median(np.diff(times)))
    common = times[valid]
    dt = float(np.median(np.diff(common))) if len(common)>1 else native_dt
    gaps = np.diff(times[indices])
    cuts = np.flatnonzero((np.diff(indices) != 1) | (gaps > 1.5*dt) | (gaps > 1.5*native_dt)) + 1
    return np.split(indices, cuts)


def actor_exposure(boxes, valid, sample_weight_s):
    """G2 union of overlapping neighbors, divided by reference actor-seconds."""
    boxes, valid = np.asarray(boxes, np.float64), np.asarray(valid, bool)
    weights = np.asarray(sample_weight_s, np.float64)
    if (boxes.shape != valid.shape + (5,) or valid.ndim != 2 or
            weights.shape != (valid.shape[1],) or not np.isfinite(weights).all() or
            np.any(weights <= 0) or not np.isfinite(boxes[valid]).all() or
            np.any(boxes[valid, 3:] <= 0)):
        raise ValueError('G2 requires finite valid boxes, positive dimensions and native sample durations')
    denominator = float(np.sum(valid * weights))
    if not denominator:
        raise ValueError('No reference actor support')
    exposed = np.zeros_like(valid)
    left, right = np.triu_indices(len(boxes), 1)
    # ponytail: exact quadratic pairs in blocks; spatial indexing only if actor count grows.
    for start in range(0, len(left), 256):
        a, b = left[start:start + 256], right[start:start + 256]
        mask = valid[a] & valid[b]
        hit = np.zeros_like(mask)
        hit[mask] = signed_distance(boxes[a][mask], boxes[b][mask], .7) < 0
        np.logical_or.at(exposed, a, hit)
        np.logical_or.at(exposed, b, hit)
    numerator = float(np.sum(exposed * weights))
    return dict(actor_exposure_rate=numerator / denominator,
                actor_overlap_seconds=numerator, valid_actor_seconds=denominator)


def compute_trs(reference_rows, generated_rows, exposure_by_scene, transition_bins):
    """Compute one population score from exactly paired frozen scene evidence.

    Exposure rows contain ``gt`` and ``catk`` fractions. Histograms already encode
    the frozen native windows and reference support; no missing value is imputed.
    """
    refs, gens = list(reference_rows), list(generated_rows)
    ids = sorted(r['scene_id'] for r in refs)
    if (not ids or len(ids) != len(set(ids)) or
            sorted(r['scene_id'] for r in gens) != ids or set(exposure_by_scene) != set(ids)):
        raise ValueError('Exact, unique reference/generated/exposure scene pairing required')
    if set(transition_bins) != {'0.5', '1.0', '2.0'}:
        raise ValueError('Frozen transition deltas .5, 1 and 2 seconds required')
    rates = np.asarray([[exposure_by_scene[s][m] for m in ('gt', 'catk')] for s in ids], np.float64)
    if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError('Exposure rates must be finite fractions in [0,1]')
    support = {f: [r['scene_id'] for r in refs if r['state'].get(f) is not None] for f in FEATURES}
    distribution = score_population(refs, gens, support, transition_bins)
    delta = rates[:, 1] - rates[:, 0]
    # Same scene-equal reduction as the frozen E_plus audit, without rounding.
    weights = np.ones(len(ids)) / len(ids)
    plus, minus, net = [float(np.sum(weights * v)) for v in
                        (np.maximum(delta, 0.), np.maximum(-delta, 0.), delta)]
    G = float(geometry_factor(plus))
    D = distribution['score']
    return {'metric_version': VERSION, 'scene_count': len(ids), 'D': D,
            'E_plus': plus, 'E_minus': minus, 'net_delta': net, 'G': G,
            'TRS': None if D is None else D * G, 'q': Q,
            'status': 'available' if D is not None else 'NOT_COMPUTED_MISSING_SUPPORT',
            'reference_support': distribution['support'], 'population_hash': content_hash(ids)}


def joint_intervals(distribution_draws, excess_draws):
    d, e = np.asarray(distribution_draws, np.float64), np.asarray(excess_draws, np.float64)
    if (d.ndim != 1 or d.shape != e.shape or len(d) != 10000 or
            not np.isfinite(d).all() or np.any((d < 0) | (d > 1))):
        raise ValueError('10,000 complete, paired, finite D/E_plus draws required')
    g = geometry_factor(e)
    return {name: np.percentile(values, [2.5, 97.5]).tolist()
            for name, values in dict(D=d, E_plus=e, G=g, TRS=d * g).items()}


def verify_inputs(input_root=INPUT):
    input_root = Path(input_root)
    release = json.loads((RESULT / 'evidence/release_manifest.json').read_text())
    if sorted(str(p.parent.relative_to(input_root)) for p in input_root.glob('*/scenes/*/catk_manifest.json')) != release['scene_ids']:
        raise ValueError('Release must contain the exact frozen 500 scenes')
    for relative, expected in release['required_file_hashes'].items():
        if sha256(input_root / relative) != expected:
            raise ValueError('RELEASE_INPUT_HASH_MISMATCH:' + relative)
    for relative, expected in release['reference_file_hashes'].items():
        if sha256(ROOT / 'challenge/input-gt' / relative) != expected:
            raise ValueError('REFERENCE_INPUT_HASH_MISMATCH:' + relative)
    for relative, target in release['static_links'].items():
        link = input_root / relative
        if not link.is_symlink() or not link.exists() or str(link.resolve()) != target:
            raise ValueError('RELEASE_LINK_MISMATCH:' + relative)
    return release


def recompute(input_root=INPUT):
    freeze = json.loads((RESULT / 'metric_freeze.json').read_text())
    if freeze['metric_version'] != VERSION or freeze['q'] != Q or freeze.get('anchor_excess_fraction') != ANCHOR or freeze['features'] != list(FEATURES):
        raise ValueError('Wrong TRS protocol/cache version')
    for relative, expected in freeze['evidence_sha256'].items():
        if sha256(RESULT / relative) != expected:
            raise ValueError('FROZEN_EVIDENCE_MISMATCH:' + relative)
    for relative, expected in freeze['code_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('FROZEN_IMPLEMENTATION_MISMATCH:' + relative)
    release = verify_inputs(input_root)
    with gzip.open(RESULT / 'evidence/histograms.json.gz', 'rt') as stream:
        rows = json.load(stream)
    with (RESULT / 'evidence/scene_statistics.csv').open() as stream:
        stats = list(csv.DictReader(stream))
    exposures = {r['scene_id']: {'gt': float(r['gt_actor_exposure_rate']),
                                'catk': float(r['catk_actor_exposure_rate'])} for r in stats}
    refs, gens = [r['gt'] for r in rows], [r['catk'] for r in rows]
    if sorted(exposures) != release['scene_ids'] or len(stats) != len(exposures):
        raise ValueError('Scene statistics do not match the exact release population')
    result = compute_trs(refs, gens, exposures, freeze['transition_bins'])
    self_exposures = {s: {'gt': r['gt'], 'catk': r['gt']} for s, r in exposures.items()}
    self_test = compute_trs(refs, refs, self_exposures, freeze['transition_bins'])
    if self_test['TRS'] != 1.:
        raise ValueError('GT self regression failed')
    with np.load(RESULT / 'evidence/joint_bootstrap.npz', allow_pickle=False) as draws:
        intervals = joint_intervals(draws['D'], draws['E_plus'])
        for key in ('E_minus', 'net_delta'):
            values = draws[key]
            if values.shape != (10000,) or not np.isfinite(values).all():
                raise ValueError('Invalid paired '+key+' replicates')
            intervals[key] = np.percentile(values, [2.5, 97.5]).tolist()
    return result, intervals, release


def score_row(result, intervals, release):
    row = {'model': 'CAT-K', 'scene_count': result['scene_count'], 'D': result['D'],
           'E_plus_pp': result['E_plus'] * 100, 'E_minus_pp': result['E_minus'] * 100,
           'net_delta_pp': result['net_delta'] * 100, 'G': result['G'], 'TRS': result['TRS'], 'q': Q}
    for key, values in intervals.items():
        for side, value in zip(('low', 'high'), values):
            is_exposure = key in ('E_plus', 'E_minus', 'net_delta')
            row[key + ('_pp' if is_exposure else '') + '_ci95_' + side] = value * (100 if is_exposure else 1)
    row.update(trajectory_content_hash=release['trajectory_content_hash'],
               release_manifest_hash=sha256(RESULT / 'evidence/release_manifest.json'),
               metric_freeze_hash=sha256(RESULT / 'metric_freeze.json'),
               reference_support_hash=content_hash(result['reference_support']),
               population_hash=result['population_hash'], scoring_policy='OWNER_APPROVED')
    return row


def run(config=None, *, reproduce=False, dry_run=False):
    """Score/reproduce the fixed release, with one lock and atomic CSV replacement."""
    if config is not None:
        from driveaudit_eval.config import load_config
        settings = load_config(config)['evaluation']['core_metrics']['trs']
        for key, expected in (('input_root', INPUT), ('reference_root', ROOT/'challenge/input-gt')):
            if (ROOT / settings[key]).resolve() != expected.resolve():
                raise ValueError('TRS policy requires the frozen ' + key)
    if dry_run:
        freeze_path = RESULT/'metric_freeze.json'
        if not freeze_path.is_file():
            freeze_path = ROOT/'tools/configs/trs_protocol.json'
        freeze = json.loads(freeze_path.read_text())
        if freeze['metric_version'] != VERSION or freeze['q'] != Q or freeze.get('anchor_excess_fraction') != ANCHOR:
            raise ValueError('TRS policy/config differs from release manifest; re-score the approved policy first')
        print(f'TRS configuration PASS | q={Q:.2f} | dry_run=YES | inference=NO | configuration={freeze_path} | sealed_result_available={(RESULT / "run_manifest.json").is_file()}')
        return None
    if not RESULT.is_dir():
        raise ValueError('Historical sealed TRS outputs were cleared; use tools/configs/formal50.yaml for the current measured population')
    # ponytail: one release lock; separate locks only if independent releases are added.
    with (RESULT.parent/'.trs.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another TRS release operation holds the output lock') from exc
        result, intervals, release = recompute()
        row = score_row(result, intervals, release)
        path = RESULT/'scores.csv'
        descriptor, temporary = tempfile.mkstemp(prefix='.scores-', suffix='.csv.tmp', dir=RESULT)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            write_csv(temporary, [row])
            if reproduce:
                manifest = json.loads((RESULT/'run_manifest.json').read_text())
                if (manifest['metric_version'] != VERSION or manifest['q'] != Q or
                        manifest.get('anchor_excess_fraction') != ANCHOR or manifest.get('status') != 'COMPLETE'):
                    raise ValueError('REPRODUCTION_POLICY_MISMATCH')
                for relative, expected in manifest['artifact_sha256'].items():
                    if sha256(RESULT/relative) != expected:
                        raise ValueError('REPRODUCTION_ARTIFACT_MISMATCH:' + relative)
                if not path.exists() or path.read_bytes() != temporary.read_bytes():
                    raise ValueError('REPRODUCTION_MISMATCH: existing result retained')
            elif not path.exists() or path.read_bytes() != temporary.read_bytes():
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"CAT-K | TRS={result['TRS']:.12f} | 95% CI [{intervals['TRS'][0]:.12f}, {intervals['TRS'][1]:.12f}] | q={Q:.2f} | scenes={result['scene_count']} | inference=NO")
    return result


def main():
    parser = argparse.ArgumentParser(description='Compatibility entry for the same CAT-K TRS implementation; no inference.')
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    run(reproduce=args.reproduce)


if __name__ == '__main__':
    main()

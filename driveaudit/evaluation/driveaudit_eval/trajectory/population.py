"""Fresh paired population evidence using the frozen common-six and G2 kernels.

``prepare_gt`` runs before driving inference; ``run`` consumes its verified
same-run evidence after inference and measures only the selected CAT-K traffic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from driveaudit_eval.diagnostics.rounded_footprint import signed_distance
from driveaudit_eval.metrics.capabilities import map_capabilities
from driveaudit_eval.utils.io import file_sha256, jsonable, write_csv, write_json
from .input_gt import load_scene
from .trs import (ANCHOR, Q, VERSION, actor_exposure, compute_trs, content_hash,
                  episode_indices, geometry_factor, sample_weights)
from .trs_distribution import (FEATURES, eligible_tracks, extract_features,
                               make_histograms, score_population)

ROOT = Path(__file__).resolve().parents[2]


def _protocol(value):
    protocol = json.loads(Path(value).read_text()) if not isinstance(value, dict) else value
    if (protocol['metric_version'] != VERSION or protocol['q'] != Q or
            protocol['anchor_excess_fraction'] != ANCHOR or protocol['features'] != list(FEATURES) or
            protocol['geometry']['c'] != .7 or protocol['geometry']['dtype'] != 'float64' or
            protocol['geometry']['overlap'] != 'signed_distance < 0' or
            protocol['D']['horizon_seconds'] != 11.15 or
            protocol['D']['endpoint_tolerance_seconds'] != 0.043350750000000104 or
            set(protocol['transition_bins']) != {'0.5', '1.0', '2.0'}):
        raise ValueError('Frozen TRS protocol mismatch')
    return protocol


def _validate_rows(rows):
    rows = sorted((dict(row) for row in rows), key=lambda row: row['scene_id'])
    if not rows or len({row['scene_id'] for row in rows}) != len(rows):
        raise ValueError('Unique nonempty selected scene manifest required')
    for row in rows:
        relative = Path(row['relative'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3 or
                relative.parts[1] != 'scenes' or not row['dataset'] or not row['cluster_id'] or
                row['scene_id'] != relative.parts[0] + '__' + relative.name):
            raise ValueError('Invalid selected scene identity: ' + str(row))
    return rows


def _identity(scene, row, protocol, reference_hash=None):
    names = ['timeline.json', 'generation_input.json', 'history/agents.csv',
             'future/future_trajectories.csv']
    names += [str(p.relative_to(scene)) for p in (scene / 'map').glob('*') if p.is_file()]
    if (scene / 'catk_manifest.json').is_file():
        names.append('catk_manifest.json')
    inputs = {name: file_sha256(scene / name) for name in sorted(names)}
    code = {name: file_sha256(Path(__file__).parent / name)
            for name in ('population.py', 'input_gt.py', 'trs.py', 'trs_distribution.py')}
    code['rounded_footprint.py'] = file_sha256(ROOT / 'driveaudit_eval/diagnostics/rounded_footprint.py')
    return content_hash([row, protocol, inputs, code, reference_hash]), inputs


def _geometry(boxes, valid, times):
    weights, _ = sample_weights(times)
    result = actor_exposure(boxes, valid, weights)
    pair_count = pair_samples = episodes = 0
    exposed = np.zeros_like(valid)
    left, right = np.triu_indices(len(boxes), 1)
    # ponytail: a second exact pair pass retains counts; merge with the frozen
    # exposure kernel only if measured geometry runtime warrants a protocol edit.
    for start in range(0, len(left), 256):
        a, b = left[start:start + 256], right[start:start + 256]
        mask = valid[a] & valid[b]
        hit = np.zeros_like(mask)
        hit[mask] = signed_distance(boxes[a][mask], boxes[b][mask], .7) < 0
        pair_count += int(hit.any(axis=1).sum())
        pair_samples += int(hit.sum())
        for hits, support in zip(hit, mask):
            if hits.any():
                episodes += len(episode_indices(hits, support, times))
        np.logical_or.at(exposed, a, hit)
        np.logical_or.at(exposed, b, hit)
    return dict(result, status='available', valid_actor_samples=int(valid.sum()),
                overlap_actor_samples=int(exposed.sum()), overlap_pair_samples=pair_samples,
                unique_overlapping_pairs=pair_count, overlap_episodes=episodes,
                actors_ever_overlapping=int(exposed.any(axis=1).sum()),
                physical_collision='NOT_ASSESSED')


def _features(boxes, valid, times, history, ids, scene, dataset, protocol, reference_masks=None):
    provider, map_error = None, None
    if dataset != 'pandaset' and map_capabilities(scene)['drivable_area']:
        try:
            from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
            provider = InputGTMapProvider(scene)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            map_error = str(exc)
    values = extract_features(boxes, valid, times, history, ids, provider)
    hist = make_histograms(values, times[history:] - times[history - 1],
                           protocol['D']['horizon_seconds'], protocol['transition_bins'],
                           protocol['D']['endpoint_tolerance_seconds'], reference_masks)
    masks = hist.pop('reference_masks')
    road = values['distance_to_road_edge']
    diagnostics = {}
    for name, feature in values.items():
        finite = feature[np.isfinite(feature)]
        diagnostics[name] = dict(observations=len(finite), mean=float(finite.mean()) if len(finite) else None,
                                 minimum=float(finite.min()) if len(finite) else None,
                                 maximum=float(finite.max()) if len(finite) else None)
    return hist, masks, values, dict(features=diagnostics,
        map_status='available' if np.isfinite(road).any() else 'N/A',
        map_reason=map_error or ('PandaSet map diagnostic unavailable' if dataset == 'pandaset'
                               else None if provider else 'insufficient_drivable_area_map_support'),
        map_observations=int(np.isfinite(road).sum()))


def _validate_future_timestamps(scene, times, history, *, require_timestamp=False):
    seen = set()
    with (scene / 'future/future_trajectories.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        has_timestamp = 'timestamp' in (reader.fieldnames or ())
        if require_timestamp and not has_timestamp:
            raise ValueError('CAT-K future CSV requires its declared timestamp column')
        for row in reader:
            frame = int(row['frame_index'])
            key = (row['agent_id'], frame)
            if key in seen or not history <= frame < len(times):
                raise ValueError('Duplicate or out-of-bound future actor sample')
            seen.add(key)
            if has_timestamp:
                timestamp_us = float(row['timestamp']) * 1e6
                if not np.isfinite(timestamp_us) or abs(timestamp_us - times[frame] * 1e6) > 1e-5:
                    raise ValueError('Future CSV timestamp differs from native timeline')


def _extract_gt(row, scene, protocol):
    tracks, provenance = load_scene(scene)
    tracks = eligible_tracks(tracks)
    if not tracks:
        raise ValueError('No eligible GT vehicle actors')
    times = tracks[0].timestamps.astype(np.float64)
    history = int(tracks[0].metadata['history_frames'])
    _validate_future_timestamps(scene, times, history)
    ids = [track.agent_id for track in tracks]
    boxes = np.stack([np.column_stack((track.positions, track.headings,
        np.full(len(times), track.shape['length']), np.full(len(times), track.shape['width']))) for track in tracks])
    valid = np.stack([track.valid_mask for track in tracks])
    hist, masks, values, diagnostics = _features(boxes, valid, times, history, ids, scene,
                                                row['dataset'], protocol)
    geometry = _geometry(boxes[:, history:], valid[:, history:], times[history:])
    metrics = dict(row, branch='gt', eligible_actors=len(ids),
        history_frames=history, future_frames=len(times) - history,
        future_horizon_seconds=float(times[-1] - times[history - 1]),
        future_first_timestamp_s=float(times[history]), future_last_timestamp_s=float(times[-1]),
        native_dt_median_s=float(np.median(np.diff(times))),
        native_gap_count=int((np.diff(times) > 1.5 * np.median(np.diff(times))).sum()),
        future_reference_valid_fraction=float(valid[:, history:].mean()),
        missing_reference_actor_samples=0, **geometry, **diagnostics)
    data = dict(boxes=boxes, valid=valid, times=times, actor_ids=np.asarray(ids),
                history_frames=np.asarray(history), **{'feature_' + k: v for k, v in values.items()},
                **{'mask_' + k: v for k, v in masks.items()})
    return dict(metrics=metrics, histogram=dict(hist, scene_id=row['scene_id']), provenance=provenance), data


def _extract_catk(row, scene, protocol, reference, data, *, branch='catk'):
    tracks, provenance = load_scene(scene)
    indexed = {track.agent_id: track for track in tracks}
    times, history = data['times'], int(data['history_frames'])
    _validate_future_timestamps(scene, times, history, require_timestamp=branch == 'catk')
    boxes, reference_valid = data['boxes'].copy(), data['valid']
    valid = reference_valid.copy()
    for i, actor in enumerate(data['actor_ids']):
        track = indexed.get(str(actor))
        if track is None:
            valid[i, history:] = False
            continue
        if (not np.array_equal(track.timestamps, times) or track.metadata['history_frames'] != history or
                not np.array_equal(track.valid_mask[:history], reference_valid[i, :history]) or
                not np.array_equal(track.positions[:history], boxes[i, :history, :2]) or
                not np.array_equal(track.headings[:history], boxes[i, :history, 2])):
            raise ValueError('Candidate timeline/history identity mismatch: ' + str(actor))
        # The frozen scalar shape comes from GT's first available annotation.
        boxes[i, history:, :2] = track.positions[history:]
        boxes[i, history:, 2] = track.headings[history:]
        valid[i, history:] &= track.valid_mask[history:]
    missing = int((reference_valid[:, history:] & ~valid[:, history:]).sum())
    hist, masks, values, diagnostics = _features(boxes, valid, times, history,
        list(data['actor_ids']), scene, row['dataset'], protocol,
        {feature: data['mask_' + feature] for feature in FEATURES})
    geometry = (_geometry(boxes[:, history:], reference_valid[:, history:], times[history:])
                if not missing else dict(status='NOT_COMPUTED_MISSING_SUPPORT', actor_exposure_rate=None,
                                        actor_overlap_seconds=None, valid_actor_seconds=reference['metrics']['valid_actor_seconds']))
    metrics = dict(reference['metrics'], branch=branch, **diagnostics)
    # Erase GT geometry before overlaying CAT-K, including unavailable quantities.
    for key in ('actor_exposure_rate', 'actor_overlap_seconds', 'valid_actor_seconds', 'valid_actor_samples',
                'overlap_actor_samples', 'overlap_pair_samples', 'unique_overlapping_pairs',
                'overlap_episodes', 'actors_ever_overlapping'):
        metrics[key] = geometry.get(key)
    metrics.update(status=geometry['status'], missing_reference_actor_samples=missing,
                   future_generated_valid_fraction=float(valid[:, history:].mean()))
    arrays = dict(boxes=boxes, valid=valid, times=times, actor_ids=data['actor_ids'],
                  history_frames=data['history_frames'], **{'feature_' + k: v for k, v in values.items()})
    return dict(metrics=metrics, histogram=dict(hist, scene_id=row['scene_id']), provenance=provenance), arrays


def _load_cache(path, expected_hash):
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError('Same-run trajectory evidence hash mismatch: ' + str(path))
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    return json.loads(str(data.pop('metadata_json'))), data


def _measure(rows, output, protocol, root, branch, resume, *, input_root=None, reference_root=None):
    folder = output / 'evidence/trajectory' / branch
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / 'index.json'
    old = json.loads(index_path.read_text()) if resume and index_path.exists() else {}
    index, results = {}, []
    gt_index = (json.loads((output / 'evidence/trajectory/gt/index.json').read_text())
                if branch != 'gt' else {})
    input_root = Path(input_root) if input_root is not None else root / ('challenge/input-gt' if branch == 'gt' else 'challenge/input-catk')
    reference_root = Path(reference_root) if reference_root is not None else root / 'challenge/input-gt'
    for row in rows:
        sid = row['scene_id']
        path = folder / (sid + '.npz')
        scene = input_root / row['relative']
        try:
            reference_hash = gt_index.get(sid, {}).get('sha256') if branch != 'gt' else None
            identity, inputs = _identity(scene, row, protocol, reference_hash)
            if old.get(sid, {}).get('identity') == identity:
                metadata, _ = _load_cache(path, old[sid]['sha256'])
                metadata['metrics']['computation'] = 'same_run_resumed'
                index[sid] = old[sid]
            else:
                if branch == 'gt':
                    metadata, arrays = _extract_gt(row, scene, protocol)
                else:
                    gt_path = output / 'evidence/trajectory/gt' / (sid + '.npz')
                    reference, arrays = _load_cache(gt_path, reference_hash)
                    current_gt_identity, _ = _identity(reference_root / row['relative'], row, protocol)
                    if reference['identity'] != current_gt_identity:
                        raise ValueError('GT inputs/protocol/code changed since same-run preparation')
                    metadata, arrays = _extract_catk(row, scene, protocol, reference, arrays, branch=branch)
                metadata.update(identity=identity, inputs=inputs)
                metadata['metrics']['computation'] = 'freshly_computed'
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, metadata_json=np.asarray(json.dumps(jsonable(metadata), allow_nan=False)), **arrays)
                temporary.replace(path)
                index[sid] = dict(identity=identity, sha256=file_sha256(path))
            results.append(metadata)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append(dict(metrics=dict(row, branch=branch, status='ERROR', error=f'{type(exc).__name__}: {exc}')))
        write_json(index_path, index)
        print(f'TRAJECTORY {branch} {len(results)}/{len(rows)} {sid} {results[-1]["metrics"]["status"]}', flush=True)
    (output / branch).mkdir(exist_ok=True)
    write_csv(output / branch / 'trajectory_metrics.csv', [item['metrics'] for item in results])
    return results


def _point(rows, reference, generated, bins):
    by_ref = {item['metrics']['scene_id']: item for item in reference}
    by_gen = {item['metrics']['scene_id']: item for item in generated}
    ids = [row['scene_id'] for row in rows]
    supported = [sid for sid in ids if by_ref[sid]['metrics'].get('actor_exposure_rate') is not None and
                 by_gen[sid]['metrics'].get('actor_exposure_rate') is not None]
    common = dict(requested_scenes=len(ids), valid_scenes=len(supported),
                  independent_clusters=len({(row['dataset'], row['cluster_id']) for row in rows}))
    if len(supported) != len(ids):
        return dict(common, status='NOT_COMPUTED_MISSING_SUPPORT',
                    missing_scene_ids=sorted(set(ids) - set(supported)),
                    **dict.fromkeys(('D', 'E_plus', 'E_minus', 'net_delta', 'G', 'TRS', 'actor_exposure_rate')))
    result = compute_trs([by_ref[sid]['histogram'] for sid in ids], [by_gen[sid]['histogram'] for sid in ids],
        {sid: dict(gt=by_ref[sid]['metrics']['actor_exposure_rate'], catk=by_gen[sid]['metrics']['actor_exposure_rate']) for sid in ids}, bins)
    return dict(result, **common, actor_exposure_rate=float(np.mean([by_gen[sid]['metrics']['actor_exposure_rate'] for sid in ids])),
                actor_overlap_seconds=sum(by_gen[sid]['metrics']['actor_overlap_seconds'] for sid in ids),
                valid_actor_seconds=sum(by_gen[sid]['metrics']['valid_actor_seconds'] for sid in ids))


def _points(rows, refs, gens, protocol):
    return {scope: _point(subset, refs, gens, protocol['transition_bins']) for scope, subset in
            [('overall', rows)] + [(dataset, [row for row in rows if row['dataset'] == dataset])
                                  for dataset in sorted({row['dataset'] for row in rows})]}


def evaluate_paired(scene_paths, candidate_root, reference_root, output, protocol=None, *,
                    resume=True, bootstrap=False, scene_manifest=None):
    """Score the actual selected input-gt-format submission against its paired GT.

    A recording manifest is needed only for clustered intervals. No historical
    score directory, CAT-K naming, inference, or per-scene TRS is required.
    """
    from driveaudit_eval.metrics.testing_effectiveness import content_hash as manifest_hash
    candidate_root, reference_root = Path(candidate_root).resolve(), Path(reference_root).resolve()
    output = Path(output)
    protocol = _protocol(protocol if protocol is not None else ROOT / 'tools/configs/trs_protocol.json')
    declared = {}
    if scene_manifest is not None:
        with Path(scene_manifest).open(newline='') as stream:
            manifest_rows = list(csv.DictReader(stream))
        declared = {row['relative']: row for row in manifest_rows}
        if len(declared) != len(manifest_rows):
            raise ValueError('Duplicate scene in recording manifest')
    if bootstrap and not declared:
        raise ValueError('Clustered TRS intervals require an explicit recording manifest')
    rows = []
    for path in scene_paths:
        relative = Path(path).resolve().relative_to(candidate_root)
        if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[1] != 'scenes':
            raise ValueError('Selected TRS scene must be candidate_root/dataset/scenes/scene')
        if declared:
            row = dict(declared[str(relative)])
        else:
            context = json.loads((reference_root / relative / 'scene_context.json').read_text())
            row = dict(relative=str(relative), dataset=context['dataset'], cluster_id='NOT_AVAILABLE')
        row['scene_id'] = relative.parts[0] + '__' + relative.name
        rows.append(row)
    rows = _validate_rows(rows)
    refs = _measure(rows, output, protocol, ROOT, 'gt', resume, input_root=reference_root)
    gens = _measure(rows, output, protocol, ROOT, 'candidate', resume,
                    input_root=candidate_root, reference_root=reference_root)
    points = _points(rows, refs, gens, protocol)
    gt_points = _points(rows, refs, refs, protocol)
    intervals = _bootstrap(rows, refs, gens, protocol, output) if bootstrap else {}
    source_manifest = {}
    for row, ref, gen in zip(rows, refs, gens):
        relative = Path(row['relative'])
        sid = relative.parts[0] + '/' + relative.name
        source_manifest[sid] = dict(relative=row['relative'], candidate={}, reference={})
        for side, base, item in [('reference', reference_root, ref), ('candidate', candidate_root, gen)]:
            inputs = item.get('inputs', {})
            # Inputs must remain the same through extraction and aggregation.
            for name, digest in inputs.items():
                if file_sha256(base / row['relative'] / name) != digest:
                    raise ValueError('TRS input changed during scoring: ' + str(base / row['relative'] / name))
            source_manifest[sid][side] = inputs
    for scope, point in points.items():
        point['bootstrap'] = intervals.get(scope, dict(status='NOT_REQUESTED'))
    if not declared:
        for point in [*points.values(), *gt_points.values()]:
            point['independent_clusters'] = None
            point['cluster_status'] = 'NOT_AVAILABLE'
    point = points.pop('overall')
    ids = sorted(source_manifest)
    result = dict(point, scope='population', independent_scene_TRS=False, scene_ids=ids,
        kernel_population_hash=point.get('population_hash'), population_hash=manifest_hash(ids),
        selected_scene_hash=manifest_hash(rows), relative_scene_paths=[row['relative'] for row in rows],
        trajectory_snapshot_hash=manifest_hash({sid: source_manifest[sid]['candidate'] for sid in ids}),
        reference_snapshot_hash=manifest_hash({sid: source_manifest[sid]['reference'] for sid in ids}),
        protocol_sha256=manifest_hash(protocol), source_manifest=source_manifest,
        candidate_root=str(candidate_root), reference_root=str(reference_root),
        recording_manifest_sha256=file_sha256(scene_manifest) if scene_manifest is not None else None,
        code_sha256={**{name: file_sha256(Path(__file__).parent / name)
                       for name in ('population.py', 'input_gt.py', 'trs.py', 'trs_distribution.py')},
                     'rounded_footprint.py': file_sha256(ROOT / 'driveaudit_eval/diagnostics/rounded_footprint.py')},
        datasets=points, gt_self=gt_points.pop('overall'),
        scenes=[item['metrics'] for item in gens], reference_scenes=[item['metrics'] for item in refs],
        ci95=intervals.get('overall', {}).get('ci95'),
        geometry_label='rounded-footprint geometric overlap', physical_collision='NOT_ASSESSED')
    write_json(output / 'result.json', result)
    return result


def prepare_gt(scene_rows, output, protocol, *, root=ROOT, resume=False):
    """Fresh GT features, G2 and actual GT-self TRS; no CAT-K or agent inference."""
    rows, output, protocol = _validate_rows(scene_rows), Path(output), _protocol(protocol)
    results = _measure(rows, output, protocol, Path(root), 'gt', resume)
    points = _points(rows, results, results, protocol)
    result = dict(status='COMPLETE' if all(item['status'] == 'available' for item in points.values()) else 'PARTIAL',
                  overall=points.pop('overall'), datasets=points, scenes=[item['metrics'] for item in results])
    write_json(output / 'evidence/trajectory/gt_summary.json', result)
    return result


def _bootstrap_weights(rows, replicates=10000, seed=20260916):
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(rows)), np.float64)
    evidence, counts = {}, {}
    for dataset in sorted({row['dataset'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['dataset'] == dataset]
        groups = sorted({rows[i]['cluster_id'] for i in indices})
        selected = rng.integers(len(groups), size=(replicates, len(groups)))
        counts[dataset] = len(groups)
        multiplicity = np.zeros((replicates, len(groups)))
        np.add.at(multiplicity, (np.arange(replicates)[:, None], selected), 1.)
        scene_multiplicity = multiplicity[:, [groups.index(rows[i]['cluster_id']) for i in indices]]
        weights[:, indices] = scene_multiplicity / scene_multiplicity.sum(axis=1)[:, None] * len(indices) / len(rows)
        evidence[dataset + '__cluster_draws'] = selected
        evidence[dataset + '__cluster_ids'] = np.asarray(groups)
    return weights, evidence, counts


def _bootstrap_distribution(refs, gens, weights, bins):
    """Pool each resampled population then call the unchanged scoring kernel."""
    pooled = []
    invalid = np.zeros(len(weights), bool)
    for side in (refs, gens):
        blocks = {'state': {}, 'transition': {feature: {} for feature in FEATURES}}
        for feature in FEATURES:
            for block, delta in [('state', None)] + [('transition', key) for key in bins]:
                pick = lambda row: row[block][feature] if delta is None else row[block][feature].get(delta)
                support = np.array([ref['state'][feature] is not None and pick(ref) is not None for ref in refs])
                complete = np.array([pick(row) is not None and not (row.get('_support', {}).get(feature, {}).get(
                    'model_missing_observations', 0) if delta is None else row.get('_support', {}).get(feature, {}).get(
                    'transitions', {}).get(delta, {}).get('model_missing_pairs', 0)) for row in side])
                denominator = weights[:, support].sum(axis=1)
                invalid |= denominator == 0
                invalid |= (weights[:, support & ~complete].sum(axis=1) > 0)
                usable = support & complete
                values = (weights[:, usable] @ np.asarray([pick(row) for row, use in zip(side, usable) if use]).reshape(int(usable.sum()), -1)
                          if usable.any() else np.zeros((len(weights), 1)))
                values = np.divide(values, denominator[:, None], out=np.zeros_like(values), where=denominator[:, None] > 0)
                if delta is None:
                    blocks[block][feature] = values
                else:
                    blocks[block][feature][delta] = values
        pooled.append(blocks)
    scores = np.full(len(weights), np.nan)
    for draw in np.flatnonzero(~invalid):
        pair = []
        for blocks in pooled:
            pair.append(dict(scene_id='resampled_population',
                state={feature: blocks['state'][feature][draw] for feature in FEATURES},
                transition={feature: {delta: blocks['transition'][feature][delta][draw] for delta in bins} for feature in FEATURES}))
        scores[draw] = score_population([pair[0]], [pair[1]], ['resampled_population'], bins)['score']
    return scores


def _bootstrap(rows, refs, gens, protocol, output):
    weights, arrays, counts = _bootstrap_weights(rows)
    indexed_ref = {item['metrics']['scene_id']: item for item in refs}
    indexed_gen = {item['metrics']['scene_id']: item for item in gens}
    summaries = {}
    for scope in ['overall'] + sorted(counts):
        indices = [i for i, row in enumerate(rows) if scope == 'overall' or row['dataset'] == scope]
        chosen = [rows[i]['scene_id'] for i in indices]
        local_weights = weights[:, indices].copy()
        local_weights /= local_weights.sum(axis=1)[:, None]
        ref = [indexed_ref[sid] for sid in chosen]
        gen = [indexed_gen[sid] for sid in chosen]
        if any(item['metrics'].get('actor_exposure_rate') is None for item in ref + gen):
            summaries[scope] = dict(status='NOT_ESTIMABLE_MISSING_SUPPORT', valid_replicates=0)
            continue
        distribution = _bootstrap_distribution([item['histogram'] for item in ref],
                                               [item['histogram'] for item in gen], local_weights, protocol['transition_bins'])
        delta = np.array([g['metrics']['actor_exposure_rate'] - r['metrics']['actor_exposure_rate'] for r, g in zip(ref, gen)])
        plus, minus, net = [local_weights @ vector for vector in (np.maximum(delta, 0.), np.maximum(-delta, 0.), delta)]
        geometry = geometry_factor(plus)
        values = dict(D=distribution, E_plus=plus, E_minus=minus, net_delta=net, G=geometry, TRS=distribution * geometry)
        arrays.update({scope + '__' + key: value for key, value in values.items()})
        cluster_counts = counts if scope == 'overall' else {scope: counts[scope]}
        valid = int(np.isfinite(distribution).sum())
        estimable = valid == 10000 and min(cluster_counts.values()) >= 2
        summaries[scope] = dict(status=('UNSTABLE_SMALL_CLUSTER_COUNT' if min(cluster_counts.values()) < 10 else 'available')
            if estimable else 'NOT_ESTIMABLE_INSUFFICIENT_CLUSTER_OR_FEATURE_SUPPORT',
            replicates=10000, valid_replicates=valid, seed=20260916, cluster_counts=cluster_counts,
            ci95={key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in values.items()} if estimable else None,
            limitations='Percentile interval; the actual independent recording count remains small; unrecorded higher-level dependence is not covered.')
        print(f'TRAJECTORY bootstrap {scope}: {summaries[scope]["status"]}, draws={valid}/10000', flush=True)
    arrays['scene_ids'] = np.asarray([row['scene_id'] for row in rows])
    np.savez_compressed(output / 'evidence/trajectory/joint_bootstrap.npz', **arrays)
    return summaries


def run(scene_rows, output, protocol, *, root=ROOT, resume=True):
    """Measure CAT-K trajectory only using GT prepared in this output directory."""
    rows, output, protocol = _validate_rows(scene_rows), Path(output), _protocol(protocol)
    index = json.loads((output / 'evidence/trajectory/gt/index.json').read_text())
    reference = []
    for row in rows:
        sid = row['scene_id']
        try:
            metadata, _ = _load_cache(output / 'evidence/trajectory/gt' / (sid + '.npz'), index[sid]['sha256'])
            current_identity, _ = _identity(Path(root) / 'challenge/input-gt' / row['relative'], row, protocol)
            if metadata['identity'] != current_identity:
                raise ValueError('GT evidence identity differs from current inputs/protocol/code')
            reference.append(metadata)
        except (OSError, KeyError, ValueError) as exc:
            reference.append(dict(metrics=dict(row, status='ERROR', error=str(exc))))
    generated = _measure(rows, output, protocol, Path(root), 'catk', resume)
    gt_points, catk_points = _points(rows, reference, reference, protocol), _points(rows, reference, generated, protocol)
    intervals = _bootstrap(rows, reference, generated, protocol, output)
    for scope, point in catk_points.items():
        point['bootstrap'] = intervals[scope]
    geometry = []
    for ref, gen in zip(reference, generated):
        a, b = ref['metrics'], gen['metrics']
        delta = b['actor_exposure_rate'] - a['actor_exposure_rate'] if a.get('actor_exposure_rate') is not None and b.get('actor_exposure_rate') is not None else None
        geometry.append(dict(scene_id=a['scene_id'], dataset=a['dataset'], cluster_id=a['cluster_id'],
            gt_actor_exposure_rate=a.get('actor_exposure_rate'), catk_actor_exposure_rate=b.get('actor_exposure_rate'),
            E_plus=max(delta, 0.) if delta is not None else None, E_minus=max(-delta, 0.) if delta is not None else None,
            net_delta=delta, status=b['status']))
    write_csv(output / 'catk/scene_geometry.csv', geometry)
    result = dict(status='COMPLETE' if all(item['status'] == 'available' for item in catk_points.values()) else 'PARTIAL',
        gt=dict(overall=gt_points.pop('overall'), datasets=gt_points),
        catk=dict(overall=catk_points.pop('overall'), datasets=catk_points),
        scenes=[item['metrics'] for item in generated], selected_scene_hash=content_hash(rows),
        geometry_label='rounded-footprint geometric overlap', physical_collision='NOT_ASSESSED',
        dimensions='GT first available annotation length/width; identical scalar dimensions for both branches',
        eligibility='GT vehicle valid at history boundary and at least three valid history observations',
        time_support='native GT future validity; generated support loss makes the affected score unavailable',
        inference='NOT_RUN_BY_DESIGN', video_and_agents='NOT_RUN_BY_DESIGN',
        protocol_sha256=content_hash(protocol), bootstrap_population='actual selected scene and recording mapping',
        dataset_weights={dataset: sum(row['dataset'] == dataset for row in rows) / len(rows) for dataset in sorted({row['dataset'] for row in rows})})
    write_json(output / 'catk/trs.json', result)
    return result

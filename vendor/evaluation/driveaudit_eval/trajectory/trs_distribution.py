"""Frozen native-time common-six distribution kernels; road-edge is diagnostic."""
from __future__ import annotations
import math
import numpy as np
from driveaudit_eval.diagnostics.rounded_footprint import signed_distance

ALPHA = 0.1
MIN_HISTORY_VALID = 3
CONTINUOUS_WEIGHTS = {
    'linear_speed': .05, 'linear_acceleration': .05,
    'angular_speed': .05, 'angular_acceleration': .05,
    'distance_to_nearest_object': .10, 'time_to_collision': .10,
}
FEATURES = tuple(CONTINUOUS_WEIGHTS)
BIN_CONFIG = {
    'linear_speed': (0.,25.,10), 'linear_acceleration': (-12.,12.,11),
    'angular_speed': (-.628,.628,11), 'angular_acceleration': (-3.14,3.14,11),
    'distance_to_nearest_object': (-5.,40.,10), 'time_to_collision': (0.,5.,10),
}

def fixed_edges(feature):
    low, high, bins = BIN_CONFIG[feature]
    return np.linspace(low, high, bins + 1)


def jsd_similarity(p, q):
    """JSD uses natural logarithms; identical distributions score one."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("JSD distributions must be one-dimensional and shape-matched")
    if np.any(p < 0) or np.any(q < 0) or not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("JSD distributions must be non-negative and normalized")
    m = (p + q) / 2.0
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / m[mask])))
    jsd = (kl(p) + kl(q)) / 2.0
    return float(np.clip(1.0 - jsd / math.log(2.0), 0.0, 1.0))


def eligible_tracks(tracks, min_history_valid=MIN_HISTORY_VALID):
    """Causal non-ego vehicle population at the history/current boundary."""
    result = []
    for track in tracks:
        history = int(track.metadata.get("history_frames") or 0)
        current = history - 1
        if track.agent_type != "vehicle" or current < 0 or current >= len(track.valid_mask):
            continue
        if bool(track.valid_mask[current]) and int(np.sum(track.valid_mask[:history])) >= min_history_valid:
            result.append(track)
    return result


def _timeline(times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('expected a nonempty finite native timestamp vector')
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError('native timestamps must increase strictly')
    contiguous = dt <= 1.5 * np.median(dt) + 1e-9 if len(dt) else np.empty(0, bool)
    return times, dt, contiguous


def extract_features(boxes, valid, times, history_frames, actor_ids, provider=None):
    """Return seven N x futureT arrays, recomputed from poses and native times.

    Nearest distance uses the shared sharp XY footprint (c=0), independently
    of the frozen G2 c=.7 geometry. TTC is the existing radial-closing proxy.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    times, dt, contiguous = _timeline(times)
    if (boxes.ndim != 3 or boxes.shape[-1] != 5 or
            valid.shape != boxes.shape[:2] or boxes.shape[1] != len(times)):
        raise ValueError('expected boxes N x T x 5, validity N x T and times T')
    n, total = valid.shape
    if len(actor_ids) != n or len(set(actor_ids)) != n or not 1 <= history_frames < total:
        raise ValueError('actor IDs must be unique and history must include current state')
    if not np.isfinite(boxes[valid]).all() or np.any(boxes[valid, 3:] <= 0):
        raise ValueError('valid poses must be finite with positive dimensions')
    velocity = np.full((n, total, 2), np.nan)
    angular = np.full((n, total), np.nan)
    adjacent = valid[:, 1:] & valid[:, :-1] & contiguous
    velocity[:, 1:] = np.where(adjacent[..., None], np.diff(boxes[..., :2], axis=1) / dt[None, :, None], np.nan)
    yaw_delta = (np.diff(boxes[..., 2], axis=1) + np.pi) % (2 * np.pi) - np.pi
    angular[:, 1:] = np.where(adjacent, yaw_delta / dt, np.nan)
    speed = np.linalg.norm(velocity, axis=-1)

    def derivative(values):
        out = np.full((n, total), np.nan)
        out[:, 1:] = np.where(adjacent, np.diff(values, axis=1) / dt, np.nan)
        return out[:, history_frames:]

    future = slice(history_frames, None)
    features = {
        'linear_speed': speed[:, future], 'linear_acceleration': derivative(speed),
        'angular_speed': angular[:, future], 'angular_acceleration': derivative(angular),
    }
    b, active, vel = boxes[:, future], valid[:, future], velocity[:, future]
    nearest = np.full(active.shape, np.inf)
    ttc_min = np.full(active.shape, 5.)
    counts = active.sum(axis=0)
    complete_velocity = (~active | np.isfinite(vel).all(axis=-1)).all(axis=0)
    left, right = np.triu_indices(n, 1)
    # ponytail: exact O(N^2 T) pairs; blocks cap memory, spatial indexing if N grows.
    for offset in range(0, len(left), 256):
        ia, ib = left[offset:offset + 256], right[offset:offset + 256]
        pair_valid = active[ia] & active[ib]
        if not pair_valid.any():
            continue
        distances = np.full(pair_valid.shape, np.inf)
        distances[pair_valid] = signed_distance(b[ia][pair_valid], b[ib][pair_valid], 0.)
        np.minimum.at(nearest, ia, distances)
        np.minimum.at(nearest, ib, distances)
        delta = b[ib, :, :2] - b[ia, :, :2]
        center_distance = np.linalg.norm(delta, axis=-1)
        relative = vel[ib] - vel[ia]
        closing = np.divide(-np.sum(delta * relative, axis=-1), center_distance,
                            out=np.full(pair_valid.shape, np.nan), where=center_distance > 1e-9)
        pair_ttc = np.full(pair_valid.shape, 5.)
        approaching = pair_valid & np.isfinite(closing) & (closing > 1e-9)
        pair_ttc[approaching] = np.clip(distances[approaching] / closing[approaching], 0., 5.)
        pair_ttc[pair_valid & (center_distance <= 1e-9)] = 0.
        np.minimum.at(ttc_min, ia, pair_ttc)
        np.minimum.at(ttc_min, ib, pair_ttc)
    nearest[~np.isfinite(nearest)] = np.nan
    features['distance_to_nearest_object'] = nearest
    features['time_to_collision'] = np.where(active & (counts > 1) & complete_velocity, ttc_min, np.nan)
    road = np.full(active.shape, np.nan)
    if provider is not None:
        extent = np.asarray(getattr(provider, 'extent', []), dtype=float)
        if extent.shape != (4,) or not np.isfinite(extent).all():
            raise ValueError('road support requires an explicit finite map extent')
        points = b[..., :2]
        inside = (active & (points[..., 0] >= extent[0]) & (points[..., 0] < extent[1]) &
                  (points[..., 1] >= extent[2]) & (points[..., 1] < extent[3]))
        if inside.any():
            result = np.asarray(provider.distance_to_road_boundary(points[inside]), float)
            if result.shape != (int(inside.sum()),):
                raise ValueError('road query returned an unexpected shape')
            road[inside] = np.where(np.isfinite(result), result, np.nan)
    features['distance_to_road_edge'] = road
    return features


def _endpoint_indices(times, delta, tolerance, horizon):
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('delta must be positive and tolerance nonnegative')
    pairs = []
    for i, timestamp in enumerate(times):
        target = timestamp + delta
        if timestamp < -1e-9 or target > horizon + 1e-9:
            continue
        right = np.searchsorted(times, target)
        candidates = [j for j in (right - 1, right)
                      if i < j < len(times) and times[j] <= horizon + 1e-9]
        if candidates:
            j = min(candidates, key=lambda j: abs(times[j] - target))
            if abs(times[j] - target) <= tolerance + 1e-9:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _pair_mask(valid, pairs, contiguous):
    if not len(pairs):
        return np.zeros((len(valid), 0), bool)
    bad = np.pad(np.cumsum(~valid, axis=1), ((0, 0), (1, 0)))
    gaps = np.r_[0, np.cumsum(~contiguous)]
    i, j = pairs.T
    return ((bad[:, j + 1] - bad[:, i]) == 0) & ((gaps[j] - gaps[i]) == 0)


def transition_indices(times, valid, delta, tolerance, horizon):
    """Match endpoints on native time, retaining only wholly supported segments."""
    times, _, contiguous = _timeline(times)
    valid = np.asarray(valid, bool)
    if valid.shape != times.shape:
        raise ValueError('expected one validity flag per native time')
    pairs = _endpoint_indices(times, float(delta), tolerance, horizon)
    return pairs[_pair_mask(valid[None], pairs, contiguous)[0]]


def _distribution(counts, values, edges, feature):
    n, b = int(counts.sum()), int(counts.size)
    values = np.asarray(values, float)
    zero_bin = int(np.searchsorted(edges, 0., side='right'))
    if edges[-1] == 0.:
        zero_bin = len(edges) - 1
    value_bins = np.searchsorted(edges, values, side='right')
    value_bins[values == edges[-1]] = len(edges) - 1
    no_conflict = (values >= 5.) if feature == 'time_to_collision' else np.zeros(len(values), bool)
    audit = {'N': n, 'B': b, 'alpha': ALPHA, 'total_pseudocount': b * ALPHA,
             'prior_mass_fraction': float(b * ALPHA / (n + b * ALPHA)),
             'underflow': int(np.sum(values < edges[0])),
             'overflow': int(np.sum((values > edges[-1]) | no_conflict)),
             'zero_bin_count': int(np.sum(value_bins == zero_bin)),
             'no_conflict_count': int(no_conflict.sum())}
    return ((counts + ALPHA) / (n + b * ALPHA) if n else None), audit


def make_histograms(features, times_future_relative, time_horizon, transition_bins,
                    transition_tolerance, reference_masks=None):
    """Smooth raw counts once; retain every generated loss of reference support."""
    times, _, contiguous = _timeline(times_future_relative)
    if not np.isfinite(time_horizon) or time_horizon <= 0:
        raise ValueError('positive finite time horizon required')
    time_count = max(1, int(math.ceil(time_horizon - 1e-12)))
    in_window = (times >= -1e-9) & (times <= time_horizon + 1e-9)
    candidates = {str(d): _endpoint_indices(times, float(d), transition_tolerance,
                                           min(time_horizon, times[-1])) for d in transition_bins}
    result = {key: {} for key in ('state', 'transition', 'marginal', '_support', 'reference_masks')}
    result['_audit'] = {key: {} for key in ('state', 'transition', 'marginal')}
    shape = None
    for feature in CONTINUOUS_WEIGHTS:
        values = np.asarray(features[feature], float)
        if values.ndim != 2 or values.shape[1] != len(times) or (shape is not None and values.shape != shape):
            raise ValueError('all feature arrays must share shape N x native futureT')
        shape = values.shape
        expected = np.isfinite(values) & in_window if reference_masks is None else np.asarray(reference_masks[feature], bool) & in_window
        if expected.shape != values.shape:
            raise ValueError('reference feature mask shape mismatch')
        actual = expected & np.isfinite(values)
        result['reference_masks'][feature] = expected
        missing = int(np.sum(expected & ~actual))
        result['_support'][feature] = {
            'reference_observations': int(expected.sum()), 'observations': int(actual.sum()),
            'model_missing_observations': missing, 'vehicles': int(actual.any(axis=1).sum()),
            'status': 'model_missing' if missing else ('available' if actual.any() else 'reference_unavailable'),
            'transitions': {},
        }
        edges = fixed_edges(feature)
        flattened = values[actual]
        indices = np.searchsorted(edges, flattened, side='right')
        indices[flattened == edges[-1]] = len(edges) - 1
        if feature == 'time_to_collision':
            indices[flattened >= edges[-1]] = len(edges)
        counts = np.bincount(indices, minlength=len(edges) + 1).astype(float)
        result['marginal'][feature], result['_audit']['marginal'][feature] = _distribution(counts, flattened, edges, feature)
        state_counts = np.zeros((time_count, len(edges) + 1), float)
        time_indices = np.clip(np.floor(np.maximum(times, 0.)).astype(int), 0, time_count - 1)
        np.add.at(state_counts, (np.broadcast_to(time_indices, values.shape)[actual], indices), 1)
        result['state'][feature], result['_audit']['state'][feature] = _distribution(state_counts, flattened, edges, feature)
        result['transition'][feature], result['_audit']['transition'][feature] = {}, {}
        for delta, pairs in candidates.items():
            pair_expected = _pair_mask(expected, pairs, contiguous)
            pair_valid = _pair_mask(actual, pairs, contiguous)
            changes = ((values[:, pairs[:, 1]] - values[:, pairs[:, 0]])[pair_valid]
                       if len(pairs) else np.empty(0))
            delta_edges = np.asarray(transition_bins[delta][feature], float)
            if delta_edges.ndim != 1 or len(delta_edges) < 2 or not np.all(np.diff(delta_edges) > 0):
                raise ValueError('frozen transition edges must increase strictly')
            indices = np.searchsorted(delta_edges, changes, side='right')
            indices[changes == delta_edges[-1]] = len(delta_edges) - 1
            counts = np.bincount(indices, minlength=len(delta_edges) + 1).astype(float)
            distribution, audit = _distribution(counts, changes, delta_edges, 'transition')
            result['transition'][feature][delta] = distribution
            result['_audit']['transition'][feature][delta] = audit
            result['_support'][feature]['transitions'][delta] = {
                'reference_pairs': int(pair_expected.sum()), 'pairs': int(pair_valid.sum()),
                'model_missing_pairs': int(np.sum(pair_expected & ~pair_valid)),
            }
    return result


def strict_zero_geometric_mean(feature_scores):
    rows = [(w, feature_scores.get(f)) for f, w in CONTINUOUS_WEIGHTS.items()]
    if not rows or any(v is None or not np.isfinite(v) or not 0 <= v <= 1 for _, v in rows):
        return {'status': 'low_coverage_or_missing', 'score': None, 'feature_scores': feature_scores}
    denominator = sum(w for w, _ in rows)
    score = 0. if any(w > 0 and v == 0 for w, v in rows) else math.exp(sum(w * math.log(v) for w, v in rows) / denominator)
    return {'status': 'available', 'score': float(score), 'feature_scores': feature_scores,
            'available_weight': denominator}


def score_population(reference_rows, generated_rows, scene_ids, transition_bins):
    """Reference-fixed feature support, equal-scene pooling and strict-zero D."""
    ref, gen = ({r['scene_id']: r for r in rows} for rows in (reference_rows, generated_rows))
    if len(ref) != len(reference_rows) or len(gen) != len(generated_rows):
        raise ValueError('duplicate scene IDs in population')
    scores, marginal_scores, state_scores, support = {}, {}, {}, {}
    for feature in CONTINUOUS_WEIGHTS:
        requested = list(scene_ids.get(feature, ()) if isinstance(scene_ids, dict) else scene_ids)
        if len(requested) != len(set(requested)):
            raise ValueError('duplicate scene in frozen feature support')
        ids = sorted(s for s in requested if s in ref and ref[s]['state'].get(feature) is not None)
        missing = [s for s in ids if s not in gen or gen[s]['state'].get(feature) is None or
                   gen[s].get('_support', {}).get(feature, {}).get('model_missing_observations', 0)]
        missing_reference = [s for s in requested if s not in ref or
                             (isinstance(scene_ids, dict) and ref[s]['state'].get(feature) is None)]
        if not ids or missing or missing_reference:
            scores[feature] = marginal_scores[feature] = state_scores[feature] = None
            support[feature] = {'status': 'missing_generated_or_reference', 'scenes': len(ids),
                                'missing_scene_ids': missing, 'missing_reference_scene_ids': missing_reference}
            continue

        def similarity(key, selected, delta=None):
            def hist(row):
                h = row[key][feature]
                return h[str(delta)] if delta is not None else h
            p = np.mean([hist(ref[s]) for s in selected], axis=0).ravel()
            q = np.mean([hist(gen[s]) for s in selected], axis=0).ravel()
            return jsd_similarity(p / p.sum(), q / q.sum())

        state = similarity('state', ids)
        marginal_scores[feature] = (similarity('marginal', ids)
                                    if all(ref[s].get('marginal', {}).get(feature) is not None and
                                           gen[s].get('marginal', {}).get(feature) is not None for s in ids) else None)
        deltas, transition_scenes = {}, {}
        for delta in transition_bins:
            dkey = str(delta)
            tids = [s for s in ids if ref[s]['transition'][feature].get(dkey) is not None]
            if not tids or any(gen[s]['transition'][feature].get(dkey) is None or
                               gen[s].get('_support', {}).get(feature, {}).get('transitions', {}).get(dkey, {}).get('model_missing_pairs', 0)
                               for s in tids):
                break
            deltas[dkey] = similarity('transition', tids, dkey)
            transition_scenes[dkey] = len(tids)
        state_scores[feature] = state
        if len(deltas) != len(transition_bins):
            scores[feature] = None
            support[feature] = {'status': 'transition_support_missing', 'scenes': len(ids), 'state_similarity': state}
        else:
            transition = float(np.mean(list(deltas.values())))
            scores[feature] = math.sqrt(state * transition)
            support[feature] = {'status': 'available', 'scenes': len(ids), 'state_similarity': state,
                                'transition_similarity': transition, 'delta_scores': deltas,
                                'transition_scenes': transition_scenes, 'marginal_similarity': marginal_scores[feature]}
    return {**strict_zero_geometric_mean(scores), 'support': support,
            'sequence_state_score': strict_zero_geometric_mean(state_scores)['score'],
            'marginal_score': strict_zero_geometric_mean(marginal_scores)['score']}

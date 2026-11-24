"""Paper MFid/Cov ratios over the unchanged TCVV-O evidence populations.

Eligible windows are history-anchored actor/camera/start keys with at least one
projectable candidate interval. Geometry observability is measured before camera
selection. Camera windows not selected for RGB scoring remain in the denominator.
These raw ratios do not certify the provisional verifier for paper ranking.
"""


def _key(row):
    return row['agent_id'], row['camera_name'], row['frame_index']


def _index(rows, name):
    indexed = {}
    for row in rows:
        key = _key(row)
        if (not all(isinstance(value, str) and value for value in key[:2])
                or type(key[2]) is not int or key[2] < 0):
            raise ValueError(f'{name}: invalid actor/camera/start identity')
        if key in indexed:
            raise ValueError(f'{name}: duplicate eligible window {key}')
        indexed[key] = row
    return indexed


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def compute(rows, interval_diagnostics, *, selected_geometry, history_validation):
    """Return raw P/S and S/W, exact counts, and the separate verifier status.

    Inputs must be the complete primary (history-anchored) populations emitted by
    one scene's TCVV-O backend, without sampling or dropping failed visual rows.
    Missing/duplicate/inconsistent populations raise rather than count as zeros.
    """
    eligible = _index(interval_diagnostics, 'interval diagnostics')
    selected = _index(selected_geometry, 'selected geometry')
    evidence = _index(rows, 'visual evidence')
    observable = set()
    for key, row in eligible.items():
        delta = row['selected_delta_frames']
        if delta is not None:
            if type(delta) is not int or delta <= 0:
                raise ValueError('selected_delta_frames must be a positive integer or null')
            observable.add(key)
    if not set(selected) <= observable:
        raise ValueError('Selected geometry is outside the observable eligible population')
    if set(evidence) != set(selected):
        raise ValueError('Visual evidence does not cover every selected geometry window exactly once')
    counts = dict(eligible=len(eligible), geometry_observable=len(observable),
                  selected_for_visual_scoring=len(selected), visually_scorable=0,
                  supported=0, contradicted=0, ambiguous=0)
    state_counts = {'observable_supported': 'supported',
                    'observable_contradicted': 'contradicted',
                    'observable_ambiguous': 'ambiguous'}
    for key, row in evidence.items():
        geometry = selected[key]
        end = key[2] + eligible[key]['selected_delta_frames']
        if (row.get('history_anchored') is not True or geometry.get('history_anchored') is not True
                or row.get('target_frame_index') != end or geometry.get('target_frame_index') != end
                or row.get('geometry_observable') is not True
                or geometry.get('geometry_observability_status') != 'geometry_observable'):
            raise ValueError(f'Visual evidence and primary selected geometry disagree: {key}')
        state = row['motion_state']
        if state not in state_counts and state not in {'visually_unscorable', 'visually_unscorable_contamination'}:
            raise ValueError(f'Unknown selected-window motion state: {state}')
        scorable = state in state_counts
        if row.get('visually_scorable') is not scorable:
            raise ValueError('Visual scoring flag disagrees with motion state')
        if scorable:
            counts['visually_scorable'] += 1
            counts[state_counts[state]] += 1
    passed = history_validation.get('passed')
    if type(passed) is not bool or not isinstance(history_validation.get('status'), str):
        raise ValueError('An explicit history validation result is required')
    w, o, s, p = (counts[k] for k in ('eligible', 'geometry_observable', 'visually_scorable', 'supported'))
    status = 'valid_low_sample' if passed else history_validation['status']
    return {
        'implementation_version': 'tcvv_o_paper_evidence_ratios_v1',
        'population': 'history_anchored_actor_camera_start_with_projectable_candidate_interval',
        'aggregation': 'pooled_window_counts', 'units': 'ratio',
        'MFid': _ratio(p, s), 'Cov': _ratio(s, w),
        'geometry_observable_coverage': _ratio(o, w),
        'visual_scorable_coverage_given_observable': _ratio(s, o),
        'contradiction_rate': _ratio(counts['contradicted'], s),
        'ambiguity_rate': _ratio(counts['ambiguous'], s),
        'counts': counts,
        'denominators': {'MFid': s, 'Cov': w, 'geometry_observable_coverage': w,
                         'visual_scorable_coverage_given_observable': o,
                         'contradiction_rate': s, 'ambiguity_rate': s},
        'availability_reasons': {'MFid': None if s else 'no_visually_scorable_windows',
                                 'Cov': None if w else 'no_eligible_windows'},
        'formal_motion_fidelity_status': status,
        'history_model_gate_passed': passed,
        'thresholds_provisional': history_validation.get('thresholds_provisional', True),
        'validated_MFid': None, 'paper_eligible': False,
        'validation_reason': ('independent_calibration_and_coverage_validation_required' if passed
                              else 'history_hypothesis_model_validation_failed'),
        'history_validation': history_validation,
    }

"""Image-Trajectory Consistency: shared dispatch, independent scientific backends."""
from copy import deepcopy
from importlib import import_module, util
import logging
from pathlib import Path

OBSERVABILITY_STATES = ('supported', 'contradicted', 'ambiguous', 'unobservable', 'unscorable')
BACKENDS = ('gt_vtc', 'gt_vtc_v2', 'mtpc', 'tcvv', 'tcvv_h', 'tcvv_o',
            'tcvv_bench', 'image_trajectory_consistency', 'image_trajectory_consistency_v1')


def normalize_state(state):
    """Normalize vocabulary only; thresholds and original evidence are untouched."""
    if state in OBSERVABILITY_STATES:
        return state
    if state == 'not_observable':
        return 'unobservable'
    if state == 'weakly_supported':
        return 'ambiguous'
    if state.startswith('geometry_unobservable'):
        return 'unobservable'
    if state.startswith(('visually_unscorable', 'unscorable')):
        return 'unscorable'
    for prefix in ('observable_', 'motion_', 'presence_'):
        if state.startswith(prefix) and state[len(prefix):] in OBSERVABILITY_STATES:
            return state[len(prefix):]
    return None


def _states(value):
    if isinstance(value, list):
        return [_states(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _states(item) for key, item in value.items()}
    states = {key: normalize_state(item) for key, item in value.items()
              if isinstance(item, str) and (key.endswith(('state', 'status')) or key == 'observability')
              and normalize_state(item) is not None}
    if states:
        result['normalized_observability'] = states
    return result


def _input_paths(config):
    """Explicit original input fields; output/baseline destinations are never inputs."""
    paths = list(config.get('required_paths', []))
    for group in ('inputs', 'dataset'):
        for key, value in config.get(group, {}).items():
            if isinstance(value, str):
                paths.append(value)
    backend = config.get('backend', {})
    for key in ('checkpoint', 'repo', 'config', 'cache', 'frozen_parameters', 'gt_vtc_v1_results'):
        if isinstance(backend.get(key), str):
            paths.append(backend[key])
    pairing = config.get('pairing', {})
    manifest = pairing.get('manifest', config.get('inputs', {}).get('pair_manifest'))
    if isinstance(manifest, str):
        paths.append(manifest)
    elif isinstance(manifest, dict):
        pair_id = pairing.get('pair_id', pairing.get('default_pair_id'))
        for pair in manifest.get('pairs', []):
            if pair_id and pair.get('pair_id') != pair_id:
                continue
            for group in ('image_sequence', 'trajectory_sequence', 'calibration', 'ego_pose'):
                for key, value in pair.get(group, {}).items():
                    # Optional GT is checked by paired-GT protocol, not submission-only.
                    if key != 'gt_future_root' and (key.endswith(('_root', '_path')) or key == 'path') and isinstance(value, str):
                        paths.append(value)
    for group in (config.get('inputs', {}), backend):
        for key, value in group.items():
            if key.endswith('_config') and isinstance(value, dict):
                paths.extend(_input_paths(value))
    return list(dict.fromkeys(paths))


def _run(name, config, output):
    packages = {'gt_vtc': 'driveaudit_eval.research.consistency.observed_flow', 'gt_vtc_v2': 'driveaudit_eval.research.consistency.geometry_flow', 'tcvv': 'driveaudit_eval.research.consistency.trajectory_verification', 'tcvv_h': 'driveaudit_eval.research.consistency.hypothesis_verification', 'tcvv_o': 'driveaudit_eval.research.consistency.observable_verification', 'tcvv_bench': 'driveaudit_eval.research.consistency.verification_benchmark', 'image_trajectory_consistency': 'driveaudit_eval.research.consistency.perception_trajectory', 'image_trajectory_consistency_v1': 'driveaudit_eval.research.consistency.projection_consistency'}
    package = packages.get(name, 'driveaudit_eval.consistency.' + name)
    mode = config.pop('run_mode', 'full')
    force, resume = config.pop('force', False), config.pop('resume', False)
    if name == 'image_trajectory_consistency':
        pairing = config.get('pairing', {})
        args = (config, mode, output, pairing.get('manifest'), pairing.get('default_pair_id', pairing.get('pair_id')))
        if config.get('evaluation_protocol') and mode in {'audit', 'full', 'submission_only', 'distribution_reference', 'paired_gt'}:
            return import_module(f'{package}.protocol_runner').run_protocols(*args)
        if config.get('backend', {}).get('name') == 'bevformer':
            return import_module(f'{package}.bevformer_evaluator').evaluate_bevformer(*args)
        return import_module(f'{package}.evaluator').evaluate(*args)
    if name == 'image_trajectory_consistency_v1':
        config.setdefault('outputs', {})['root'] = str(output)
        runner = import_module(f'{package}.aggregation').EvaluationRunner(
            config, force, resume, config.get('scene_ids'), config.get('max_workers', 1))
        if mode in {'evaluate-one', 'evaluate-ten', 'corruption'}:
            return runner.evaluate(mode == 'corruption')
        if mode not in {'audit', 'projection', 'aggregate', 'full'}:
            raise ValueError(f'Unsupported run_mode for {name}: {mode}')
        return getattr(runner, mode)()
    config['output'] = str(output)
    module = import_module(f'{package}.aggregate')
    if name in {'gt_vtc', 'tcvv_h', 'tcvv_o'}:
        return module.run(config, mode, force, resume)
    if name in {'mtpc', 'tcvv'}:
        return module.run(config, mode)
    runner = module.Runner(config, force, resume)
    if name == 'gt_vtc_v2':
        if mode in {'audit', 'timestamps', 'geometry'}:
            return runner.core()
        if mode == 'history_calibration':
            return runner.history_calibration()
        if mode not in {'future_evaluation', 'camera_diagnostics', 'oracle_leakage', 'ablation', 'full'}:
            raise ValueError(f'Unsupported run_mode for {name}: {mode}')
        return runner.full()
    methods = {'audit': 'audit', 'discover': 'audit', 'build_pairs': 'pairs',
               'background_compensation': 'background', 'current_holdout': 'holdout',
               'calibrate': 'validate', 'validate': 'validate', 'multiscene': 'validate',
               'human_export': 'validate', 'full': 'validate'}
    if mode not in methods:
        raise ValueError(f'Unsupported run_mode for {name}: {mode}')
    result = getattr(runner, methods[mode])()
    return result if isinstance(result, dict) else {'result': result}



def _backend_result(payload):
    """Execution finishing cannot promote a failed scientific gate to acceptance."""
    labels = {key: payload[key] for key in ('status', 'implementation_status', 'benchmark_readiness_status')
              if isinstance(payload.get(key), str)}
    failed = []
    for group in ('hard_gates', 'gates'):
        for key, gate in payload.get(group, {}).items():
            if isinstance(gate, dict) and gate.get('passed') is False and gate.get('blocks_primary', True):
                failed.append(f'{group}.{key}: {gate.get("status", "failed")}')
    status = 'COMPLETED'
    lowered = [value.lower() for value in labels.values()]
    if any(value.startswith('blocked') for value in lowered):
        status = 'BLOCKED'
    elif any(value.startswith(('skipped', 'not_available')) for value in lowered):
        status = 'SKIPPED'
    elif failed or any(value.startswith(('invalid', 'insufficient', 'diagnostic', 'prototype', 'limited',
                                         'partial', 'unscorable', 'unobservable', 'not_run', 'failed')) for value in lowered):
        status = 'PARTIAL'
    result = {'status': status, 'execution_status': 'COMPLETED',
              'scientific_status': payload.get('status', payload.get('benchmark_readiness_status',
                                   'NOT_ACCEPTED' if failed else 'NOT_ASSESSED')),
              'backend_statuses': labels, 'failed_gates': failed, 'result': _states(payload)}
    if status != 'COMPLETED':
        result['reason'] = payload.get('reason') or '; '.join(failed) or '; '.join(f'{k}={v}' for k, v in labels.items())
    return result


def evaluate(config, context) -> dict:
    if not config.get('enabled', True):
        return {'status': 'DISABLED'}
    if config.get('mode') == 'input_gt_diagnostic':
        from driveaudit_eval.research.consistency.input_diagnostic import evaluate as evaluate_input_gt_diagnostic
        return evaluate_input_gt_diagnostic(config, context)
    if config.get('mode') == 'input_gt':
        from driveaudit_eval.consistency.input_gt_full import evaluate as evaluate_input_gt
        return evaluate_input_gt(config, context)
    root = Path(context.get('root', '.')).resolve()
    logger = context.get('logger') or logging.getLogger(__name__)
    results = {}
    for name, settings in config.get('backends', {}).items():
        if not settings.get('enabled', False):
            results[name] = {'status': 'DISABLED'}
            continue
        try:
            if name not in BACKENDS:
                raise ValueError(f'Unknown consistency backend: {name}')
            missing = []
            dependencies = list(settings.get('dependencies', []))
            if settings.get('backend', {}).get('name') == 'bevformer':
                dependencies.extend(['torch', 'mmcv', 'mmdet', 'mmdet3d'])
            for dependency in dict.fromkeys(dependencies):
                try:
                    available = util.find_spec(dependency) is not None
                except ModuleNotFoundError:
                    available = False
                if not available:
                    missing.append(f'dependency unavailable: {dependency}')
            for value in _input_paths(settings):
                path = Path(value)
                path = path if path.is_absolute() else root / path
                if not path.exists():
                    missing.append(f'input unavailable: {path}')
            if missing:
                results[name] = {'status': 'SKIPPED', 'reason': '; '.join(missing), 'missing': missing}
            elif context.get('dry_run', False):
                results[name] = {'status': 'DRY_RUN', 'reason': 'Inputs and declared dependencies checked; model not executed'}
            else:
                logger.info('[CONSISTENCY] Running %s', name)
                output = Path(context['output_dir']) / 'consistency' / name
                backend_config = deepcopy(settings)
                backend_config['_config_source'] = str(Path(context['output_dir']) / 'resolved_config.yaml')
                payload = _run(name, backend_config, output)
                results[name] = _backend_result(payload)
        except (ImportError, FileNotFoundError) as error:
            if isinstance(error, ImportError) and (getattr(error, 'name', '') or '').startswith(('evaluation', 'tools')):
                logger.exception('[CONSISTENCY] Internal import failure in %s', name)
                results[name] = {'status': 'FAILED', 'reason': f'{type(error).__name__}: {error}'}
            else:
                results[name] = {'status': 'SKIPPED', 'reason': f'{type(error).__name__}: {error}'}
        except Exception as error:
            logger.exception('[CONSISTENCY] Failed %s', name)
            results[name] = {'status': 'FAILED', 'reason': f'{type(error).__name__}: {error}'}
        logger.info('[CONSISTENCY] %s: %s%s', name, results[name]['status'],
                    f" ({results[name]['reason']})" if 'reason' in results[name] else '')
    active = [row['status'] for row in results.values() if row['status'] != 'DISABLED']
    status = ('FAILED' if 'FAILED' in active else 'BLOCKED' if 'BLOCKED' in active
              else 'PARTIAL' if 'PARTIAL' in active or len(set(active)) > 1
              else active[0] if active else 'DISABLED')
    return {'status': status, 'backends': results, 'observability_states': list(OBSERVABILITY_STATES)}

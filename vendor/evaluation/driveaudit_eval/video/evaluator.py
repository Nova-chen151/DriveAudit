"""Video evaluation through the original locked dataset-level metric engine."""
from copy import deepcopy
import importlib
import logging
from pathlib import Path

RESULT_KEYS = {
    'fid': 'fid', 'fvd': 'fvd', 'temporal_warp': 'temporal_consistency',
    'history_future': 'history_future_continuity', 'loftr': 'multi_view_consistency',
    'epipolar': 'geometry_consistency',
}


def _skip(reason):
    return {'status': 'SKIPPED', 'enabled': True, 'skipped': True, 'reason': reason,
            'metric_validity': 'unavailable', 'metric_confidence': 'unavailable'}


def _check(config, root):
    results = {}
    paths = config.get('paths', {})
    def exists(value):
        return bool(value) and (root / Path(value).expanduser()).exists()
    dependencies = {}
    embedded = {}
    if exists(paths.get('generated_root')) and any(
        config['metrics'].get(name, {}).get('enabled') and not paths.get(key)
        for name, key in [('history_future', 'history_root'), ('epipolar', 'metadata_root')]
    ):
        try:
            from driveaudit_eval.video.io import discover_samples
            resolved = {key: root / Path(paths[key]).expanduser() if paths.get(key) else None
                        for key in ('dataset_root', 'generated_root', 'history_root', 'metadata_root', 'real_reference_root')}
            samples = discover_samples(**resolved, cameras=config['dataset']['cameras'],
                                       max_scenes=config['dataset'].get('max_scenes'))
            embedded['history_root'] = any(any(sample.history_images.values()) for sample in samples)
            embedded['metadata_root'] = any(sample.camera_metadata for sample in samples)
        except (ImportError, OSError, ValueError):
            pass
    for name, options in config['metrics'].items():
        if not options.get('enabled', False):
            results[name] = {'status': 'DISABLED', 'enabled': False, 'skipped': True, 'reason': 'disabled'}
            continue
        issues = []
        if name in ('fid', 'fvd', 'loftr', 'epipolar') and config.get('runtime', {}).get('skip_heavy', False):
            issues.append('runtime.skip_heavy is enabled')
        if name not in RESULT_KEYS:
            issues.append('historical optional metric is not implemented in the formal engine')
        required = ['generated_root']
        if name in ('fid', 'fvd'):
            required.append('real_reference_root')
        if name == 'history_future':
            required.append('history_root')
        if name == 'epipolar':
            required.append('metadata_root')
        issues += [f'missing input paths.{key}: {paths.get(key)}' for key in required if not exists(paths.get(key)) and not embedded.get(key)]
        modules = ['numpy', 'PIL']
        if name in ('fid', 'fvd'):
            modules += ['scipy', 'torch']
        if name == 'fid':
            modules += ['torchvision', 'pyiqa']
        if name in ('loftr', 'epipolar'):
            modules += ['torch', 'kornia', 'cv2']
        if name in ('temporal_warp', 'history_future'):
            modules += ['cv2']
            if options.get('flow_backend', 'opencv') not in ('opencv', 'fallback_opencv'):
                issues.append(f"unsupported flow backend: {options['flow_backend']}")
        for module in modules:
            if module not in dependencies:
                try:
                    importlib.import_module(module)
                    dependencies[module] = None
                except Exception as exc:
                    dependencies[module] = f'{module}: {type(exc).__name__}: {exc}'
            if dependencies[module]:
                issues.append(dependencies[module])
        if name in ('fid', 'fvd', 'loftr', 'epipolar') and not dependencies.get('torch'):
            torch = importlib.import_module('torch')
            if str(config.get('runtime', {}).get('device', 'cpu')).startswith('cuda') and not torch.cuda.is_available():
                issues.append('configured CUDA device is unavailable')
        if name == 'fid':
            if options.get('backend', 'pytorch_fid_inception_v3') != 'pytorch_fid_inception_v3':
                issues.append('canonical FID requires backend=pytorch_fid_inception_v3')
            weight = options.get('weights_path') or config.get('weights', {}).get('fid_inception_v3')
            if not exists(weight):
                issues.append(f'missing canonical FID checkpoint: {weight}; automatic downloads are disabled')
            else:
                options['weights_path'] = str(root / Path(weight).expanduser())
        if name in ('fvd', 'loftr', 'epipolar'):
            key = 'fvd' if name == 'fvd' else 'loftr'
            weight = config['metrics'][key].get('weights_path') or config.get('weights', {}).get('fvd_i3d_torchscript' if key == 'fvd' else 'loftr_outdoor')
            if not exists(weight):
                issues.append(f'missing {key} checkpoint: {weight}')
            else:
                config['metrics'][key]['weights_path'] = str(root / Path(weight).expanduser())
        results[name] = _skip('; '.join(issues)) if issues else {'status': 'READY'}
    if results.get('epipolar', {}).get('status') == 'READY' and results.get('loftr', {}).get('status') != 'READY':
        results['epipolar'] = _skip('epipolar requires enabled, available LoFTR matches')
    return results


def evaluate(config, context) -> dict:
    if not config.get('enabled', True):
        return {'status': 'DISABLED'}
    oracle = None
    if config.get('mode') == 'input_gt':
        from driveaudit_eval.video.input_gt import prepare
        config, oracle = prepare(config, context)
    from driveaudit_eval.video.config import DEFAULT_CONFIG, deep_update
    cfg = deep_update(DEFAULT_CONFIG, deepcopy(config))
    root = Path(context['root']).resolve()
    checks = _check(cfg, root)
    if context.get('dry_run'):
        return {'status': 'DRY_RUN', 'metrics': checks, 'model_forwards': False, 'ground_truth_self_evaluation': oracle}
    ready = [name for name, check in checks.items() if check['status'] == 'READY']
    quality_enabled = any(item.get('enabled', False) for item in cfg.get('visual_quality', {}).values() if isinstance(item, dict))
    if not ready and not quality_enabled:
        return {'status': 'SKIPPED', 'reason': 'no enabled metric has available inputs and dependencies', 'metrics': checks}
    from driveaudit_eval.video.evaluation_engine import EvaluationEngine
    logger = context.get('logger', logging.getLogger(__name__))
    for name in cfg['metrics']:
        cfg['metrics'][name]['enabled'] = name in ready
    cfg['paths']['output_dir'] = str(Path(context['output_dir']) / 'video_details')
    engine = EvaluationEngine(cfg, root)
    engine.state.set_deterministic()
    engine.load_dataset_manifest()
    engine.build_execution_graph()
    engine.state.graph = engine.graph.describe()
    for name, check in checks.items():
        if name in RESULT_KEYS and check['status'] == 'SKIPPED':
            engine.state.results[RESULT_KEYS[name]] = check
    logger.info('[VIDEO] Running %s', ', '.join(ready))
    engine.graph.execute(engine.state)
    summary = engine.state.results['summary']
    for name in ready:
        result = summary[RESULT_KEYS[name]]
        result['status'] = result.get('status', 'SKIPPED' if result.get('skipped') else 'COMPLETED')
        checks[name] = result
    summary['metrics'] = checks
    if oracle:
        summary['ground_truth_self_evaluation'] = oracle
    states = {item['status'] for item in checks.values()}
    summary['status'] = 'FAILED' if 'FAILED' in states else 'PARTIAL' if 'SKIPPED' in states else 'COMPLETED'
    return summary

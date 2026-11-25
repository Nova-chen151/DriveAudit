"""Submitted trajectory evaluation using the unchanged numeric kernels."""
from importlib import import_module, util
import json
from pathlib import Path

GROUPS = ('integrity', 'kinematics', 'map_compliance', 'collision', 'interaction', 'reference', 'distribution')


def _call(module, function, *args):
    return getattr(import_module('.' + module, __package__), function)(*args)


def _path(value, root):
    return root / value if value else None


def evaluate(config, context):
    if not config.get('enabled', True):
        return {'status': 'disabled'}
    root = Path(context['root'])
    inputs = config.get('inputs', {})
    manifest = _path(inputs.get('manifest'), root)
    submission = _path(inputs.get('submission'), root)
    checks = {'numpy': util.find_spec('numpy') is not None, 'scipy': util.find_spec('scipy') is not None,
              'shapely': util.find_spec('shapely') is not None}
    if config.get('mode') == 'input_gt':
        from driveaudit_eval.trajectory.input_gt import prepare
        scenes = prepare(config, context)
    elif manifest and manifest.is_file():
        scenes = json.loads(manifest.read_text()).get('scenes', [])
    elif submission and submission.is_file():
        scenes = [{'submission_file': str(submission), 'gt_file': inputs.get('ground_truth'), 'map_file': inputs.get('map')}]
    else:
        return {'status': 'skipped', 'reason': 'trajectory inputs.manifest or inputs.submission is missing', 'dependencies': checks}
    if not scenes:
        return {'status': 'skipped', 'reason': 'trajectory manifest contains no scenes', 'dependencies': checks}
    file_checks = []
    for scene in scenes:
        for key in ('submission_file', 'gt_file', 'map_file'):
            if key in ('submission_file', 'gt_file') and '_tracks' in scene:
                file_checks.append({'scene_id': scene.get('scene_id'), 'field': key, 'path': None,
                                    'available': True, 'storage': 'memory'})
                continue
            path = _path(scene.get(key), root)
            file_checks.append({'scene_id': scene.get('scene_id'), 'field': key, 'path': str(path) if path else None,
                                'available': path.is_file() if path else False})
    if context.get('dry_run'):
        return {'status': 'dry_run', 'inputs': file_checks, 'dependencies': checks, 'scene_count': len(scenes)}
    if not checks['numpy']:
        return {'status': 'skipped', 'reason': 'numpy dependency is not available'}
    from driveaudit_eval.trajectory.io import load_tracks
    from driveaudit_eval.trajectory.validate_submission import select_protocol
    from driveaudit_eval.evaluation.detail_retention import enabled, compact_trajectory_result, verify_metric_detail_archives
    compact_details = enabled(config)
    rows = []
    for scene in scenes:
        checkpoint = Path(context['output_dir']) / 'trajectory_scenes' / (scene.get('scene_id', 'scene').replace('/', '__') + '.json')
        if scene.get('input_gt_scene') and config.get('resume', True) and checkpoint.is_file():
            previous = json.loads(checkpoint.read_text())
            if previous.get('status') == 'completed' and previous.get('input_gt_signature') == scene['input_gt_signature']:
                if '_retained_details' in previous:
                    verify_metric_detail_archives(previous, context['output_dir'], include_quality_children=False, include_native=False)
                elif compact_details:
                    previous = compact_trajectory_result(previous, context['output_dir'])
                context['logger'].info('[TRAJECTORY] Reusing verified scene checkpoint %s', scene['scene_id'])
                rows.append(previous)
                continue
        try:
            if '_tracks' in scene:
                all_tracks = references = scene['_tracks']
            else:
                all_tracks = load_tracks(_path(scene['submission_file'], root), 'generated', config.get('legacy_onsite', {}))
                reference_path = _path(scene.get('gt_file'), root)
                references = load_tracks(reference_path, 'ground_truth', config.get('legacy_onsite', {})) if reference_path and reference_path.is_file() else []
            ids = [scene['scene_id']] if scene.get('scene_id') else sorted({t.scene_id for t in all_tracks})
            for scene_id in ids:
                tracks = [t for t in all_tracks if t.scene_id == scene_id and t.metadata.get('predict', True)]
                truth = [t for t in references if t.scene_id == scene_id and t.metadata.get('predict', True)]
                errors = [f'{t.agent_id}:{error}' for t in tracks for error in t.validate(scene.get('dt'))]
                if len({t.agent_id for t in tracks}) != len(tracks):
                    errors.append('duplicate_agent_ids')
                if errors or not tracks:
                    rows.append({'scene_id': scene_id, 'status': 'failed', 'reason': 'invalid trajectory schema', 'errors': errors})
                    continue
                reference_errors = [f'{t.agent_id}:{error}' for t in truth for error in t.validate(scene.get('dt'))]
                if len({t.agent_id for t in truth}) != len(truth):
                    reference_errors.append('duplicate_reference_agent_ids')
                protocol, reasons = ('A', reference_errors) if reference_errors else select_protocol(tracks, truth)
                metrics, cache = {}, {}
                def kin(values, key):
                    if scene.get('input_gt_scene') and key.startswith('gt_'):
                        key = key[3:]
                    if key not in cache:
                        cache[key] = _call('kinematic_metrics', 'evaluate_scene_kinematics', values, config['kinematics'])
                    return cache[key]
                def collision(values, key):
                    key = 'gt_collision' if key.startswith('gt_') and not scene.get('input_gt_scene') else 'collision'
                    if key not in cache:
                        cache[key] = _call('collision_metrics', 'evaluate_collisions', values, config.get('collision'))
                    return cache[key]
                def interaction(values, key):
                    if scene.get('input_gt_scene') and key.startswith('gt_'):
                        key = key[3:]
                    if key not in cache:
                        cache[key] = _call('interaction_metrics', 'evaluate_interactions', values, collision(values, key + '_collision'), config['interaction'])
                    return cache[key]
                def map_rules(values):
                    if not scene.get('map_file'):
                        return {'status': 'not_available', 'reason': 'map_file is missing', 'per_agent': {}}
                    if scene.get('input_gt_scene'):
                        from driveaudit_eval.metrics.capabilities import map_capabilities, MAP_REASON
                        if not map_capabilities(scene['input_gt_scene'])['drivable_area']:
                            return {'status': 'not_available', 'reason': MAP_REASON, 'per_agent': {}}
                        from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
                        if 'map_provider' not in cache:
                            native_config = dict(config.get('native_map') or {})
                            if native_config:
                                native_config.setdefault('cache_dir', str(Path(context['output_dir']) / 'trajectory_native_map_cache'))
                            cache['map_provider'] = InputGTMapProvider(scene['input_gt_scene'], native_config=native_config)
                        return _call('map_rule_metrics', 'evaluate_scene_map_provider', values, cache['map_provider'], config['map'])
                    return _call('map_rule_metrics', 'evaluate_scene_map_rules', values, _path(scene.get('map_file'), root), config['map'])
                for group in GROUPS:
                    if not config.get('metrics', {}).get(group, {}).get('enabled', True):
                        metrics[group] = {'status': 'disabled'}
                        continue
                    context['logger'].info('[TRAJECTORY] %s: %s', scene_id, group)
                    if group in ('reference', 'distribution') and protocol != 'B':
                        metrics[group] = {'status': 'failed' if reference_errors else 'skipped', 'reason': '; '.join(reasons)}
                        continue
                    if group == 'map_compliance' and not scene.get('map_file'):
                        metrics[group] = {'status': 'skipped', 'reason': 'map_file is missing'}
                        continue
                    try:
                        if group == 'integrity':
                            value = _call('validity_metrics', 'evaluate_scene_validity', tracks, config['validity'])
                        elif group == 'kinematics':
                            value = kin(tracks, 'kin')
                        elif group == 'collision':
                            value = collision(tracks, 'collision')
                        elif group == 'interaction':
                            value = interaction(tracks, 'interaction')
                        elif group == 'map_compliance':
                            value = map_rules(tracks)
                            value = {k: v for k, v in value.items() if not k.startswith('_')}
                        elif group == 'reference':
                            value = _call('paired_metrics', 'evaluate_paired', tracks, truth)
                        else:
                            # Missing geometry affects interaction/lane distributions, not speed distributions.
                            distribution_dependencies = {}
                            def optional_geometry(function, *args):
                                try:
                                    return function(*args)
                                except (ImportError, FileNotFoundError) as exc:
                                    distribution_dependencies[function.__name__] = str(exc)
                                    return {}
                            value = _call('distribution_metrics', 'evaluate_distributions', tracks, truth, kin(tracks, 'kin'), kin(truth, 'gt_kin'),
                                          optional_geometry(interaction, tracks, 'interaction'), optional_geometry(interaction, truth, 'gt_interaction'), optional_geometry(map_rules, tracks), optional_geometry(map_rules, truth), config['distribution'], 1)
                            value['unavailable_dependencies'] = distribution_dependencies
                        if isinstance(value, dict) and value.get('status') == 'not_available':
                            metrics[group] = dict(value, status='skipped')
                        else:
                            metrics[group] = {'status': 'completed', 'values': value}
                    except (ImportError, FileNotFoundError) as exc:
                        metrics[group] = {'status': 'skipped', 'reason': f'{type(exc).__name__}: {exc}'}
                    except Exception as exc:
                        metrics[group] = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
                if config.get('protocol') == 'universal_v1':
                    try:
                        from driveaudit_eval.trajectory.universal_metrics import evaluate_universal
                        metrics['universal'] = {'status': 'completed', 'values': evaluate_universal(tracks, config, collision(tracks, 'collision'))}
                    except (ImportError, FileNotFoundError, ValueError) as exc:
                        metrics['universal'] = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
                statuses = {v['status'] for v in metrics.values()}
                status = 'failed' if 'failed' in statuses else 'partial' if 'skipped' in statuses and 'completed' in statuses else 'skipped' if 'skipped' in statuses else 'completed' if 'completed' in statuses else 'disabled'
                scene_result = {'scene_id': scene_id, 'status': status, 'protocol': protocol, 'metrics': metrics,
                             'input_gt_provenance': scene.get('input_gt_provenance'),
                             'input_gt_signature': scene.get('input_gt_signature')}
                if compact_details:
                    scene_result = compact_trajectory_result(scene_result, context['output_dir'])
                rows.append(scene_result)
                if scene.get('input_gt_scene'):
                    from driveaudit_eval.utils.io import write_json
                    write_json(Path(context['output_dir']) / 'trajectory_scenes' / (scene_id.replace('/', '__') + '.json'), rows[-1])
        except (ImportError, FileNotFoundError) as exc:
            rows.append({'scene_id': scene.get('scene_id'), 'status': 'skipped', 'reason': f'{type(exc).__name__}: {exc}'})
        except Exception as exc:
            rows.append({'scene_id': scene.get('scene_id'), 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
    states = {r['status'] for r in rows}
    status = 'failed' if 'failed' in states else 'completed' if states == {'completed'} else 'skipped' if states == {'skipped'} else 'disabled' if states == {'disabled'} else 'partial'
    return {'status': status, 'metric_type': 'offline_submitted_trajectory', 'scenes': rows,
            'distribution_scope': 'per_scene; low-sample markers retained; no pooled distribution claim'}

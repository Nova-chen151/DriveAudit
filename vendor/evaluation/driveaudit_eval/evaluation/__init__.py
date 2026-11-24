"""Public API: one preset, one run, metrics and auditable paired final scores."""
from copy import deepcopy
from pathlib import Path
import hashlib
import os
import shutil
import sys

from driveaudit_eval.config import load_config, project_root
from driveaudit_eval.metrics.registry import CORE_IDS, OFFICIAL_IDS, METRICS
from driveaudit_eval.utils.io import write_json


def export_public(summary, work, output, config, dry_run=False):
    """Pack computed evidence before removing this run's temporary detail files."""
    if config.get('output', {}).get('retention') == 'metrics':
        from .retention import export_scores
        return export_scores(summary, work, output, config, dry_run)
    from driveaudit_eval.metrics.leaderboard import export_leaderboard
    core = deepcopy(summary.get('core_metrics', {}))
    core.setdefault('status', 'dry_run' if dry_run else 'unavailable')
    core.setdefault('model_row', dict(model=config['evaluation'].get('leaderboard', {}).get('model', 'submission'),
                                     **dict.fromkeys(OFFICIAL_IDS)))
    core.setdefault('metric_statuses', dict.fromkeys(OFFICIAL_IDS, 'not_computed'))
    testing = summary.get('testing_effectiveness', {})
    from driveaudit_eval.utils.io import write_csv
    from driveaudit_eval.metrics.testing_effectiveness import QUALITY_FIELDS
    from driveaudit_eval.metrics.quality_retention import QUALITY_VERSION, EXPORT_NAMES
    write_csv(output/'testing_effectiveness.csv',[dict(r,qtes_definition_version=QUALITY_VERSION) for r in testing.get('comparison',{}).get('rows',[])],
              headers=('dataset','scene','agent','rpdms_gt','rpdms_generated','testing_effectiveness','validity','invalid_reason','qtes_definition_version'))
    row = core['model_row']
    final = {'model':row['model'], **{k:row.get(k) for k in QUALITY_FIELDS},
             **testing.get('quality',{}), **testing.get('final',{})}
    final['epi3'] = final.pop('epi_3px')
    final.setdefault('status','not_computed')
    final['qtes_definition_version'] = QUALITY_VERSION
    headers = ['model','clip_iqa','dino_t','epi3','tvc','trs']
    headers += [name+'_'+suffix for name in EXPORT_NAMES.values() for suffix in ('generated','gt','retention')]
    headers += [group+'_'+kind+'_score' for kind in ('absolute','retention')
                for group in ('visual_quality','scene_fidelity','quality')]
    headers += [group+'_retention_coverage' for group in ('visual_quality','scene_fidelity','quality')]
    headers += ['testing_effectiveness','qtes','validity','invalid_reason','qtes_definition_version']
    write_csv(output/'final_score.csv',[final],headers=headers)
    for name,schema in [('gt_agent_baseline','qtes_gt_baseline_v1'),('gt_quality_baseline','qtes_gt_quality_baseline_v1')]:
        baseline = work/(name+'.json')
        if baseline.is_file():
            (output/(name+'.json')).write_bytes(baseline.read_bytes())
        else:
            write_json(output/(name+'.json'),dict(schema=schema,entries={},status='not_computed'))
    core['qtes_definition_version'] = QUALITY_VERSION
    payload = dict(core, results={key: value for key, value in summary.items() if key != 'core_metrics'})
    write_json(output / 'metrics.json', payload)
    export_leaderboard(core.get('model_rows') or [core['model_row']], output, markdown_name='leaderboard.md')
    # Preserve exact prediction, input-audit and per-frame evidence as UTF-8 text.
    # There is no second result tree; hashes allow lossless extraction and replay.
    artifacts = {}
    duplicate_results = {'summary.json', 'metrics.json', 'resolved_config.yaml',
                         *(name + '.json' for name in summary)}
    for path in sorted(work.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.csv', '.yaml', '.log', '.txt'):
            continue
        relative = path.relative_to(work).as_posix()
        if relative in duplicate_results:
            continue
        raw = path.read_bytes()
        artifacts[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), content_utf8=raw.decode('utf-8'))
    package = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob('*.py')) if 'research' not in path.parts}
    if config['evaluation']['profile'] == 'research':
        hashes.update({str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((package / 'research').rglob('*.py'))})
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ('numpy', 'scipy', 'PyYAML'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    manifest = dict(profile=config['evaluation']['profile'], dry_run=dry_run,
                    python_version=sys.version, package_versions=versions,
                    resolved_config=config, implementation_sha256=hashes,
                    qtes_definition_version=QUALITY_VERSION,
                    metric_registry={key: METRICS[key] for key in (*OFFICIAL_IDS,'quality_absolute_score','quality_retention_score')},
                    component_statuses={key: value.get('status') for key, value in summary.items()},
                    evidence_root=str(work), artifacts=artifacts,
                    evidence_storage='artifacts[path].content_utf8 preserves exact bytes and sha256; original paths describe execution time',
                    scientific_scope='offline agent proxy; GT self-comparison is an implementation check')
    write_json(output / 'run_manifest.json', manifest)
    return payload


def _evaluate(config=None, *, dataset=None, agent=None, output=None, profile=None,
             section='all', dry_run=False):
    """Evaluate selected scenes; return metrics, statuses and individual agent results.

    ``dataset`` filters configured scenes by dataset (or selects one scene path).
    ``agent`` accepts one registered ID or a list; defaults use all six agents.
    Existing output directories are never overwritten.
    """
    from driveaudit_eval.agents.registry import FORMAL_AGENTS
    from driveaudit_eval.evaluation.profiles import prepare_config
    from driveaudit_eval.evaluation.runner import run
    root = project_root()
    default = root / 'tools/configs/default.yaml'
    cfg = load_config(config if config is not None else
                      default if default.is_file() else {'defaults': 'leaderboard'}, profile=profile)
    settings = cfg.setdefault('evaluation', {})
    settings['profile'] = profile or settings.get('profile', 'leaderboard')
    if settings['profile'] not in ('leaderboard', 'research'):
        raise ValueError('Public profiles are leaderboard and research')
    if dataset:
        if settings.get('reuse_results'):
            raise ValueError('Dataset filtering cannot reuse pre-aggregated metrics; select scenes before creating verified reuse results')
        selected = [dataset] if isinstance(dataset, str) else list(dataset)
        known = {'nuscenes': 'nuscenes300', 'nuplan': 'nuplan', 'av2': 'av2', 'pandaset': 'pandaset'}
        for name in ('video', 'trajectory', 'consistency', 'agent'):
            item = settings.setdefault(name, {})
            paths = item.get('scene_paths', [])
            matches = []
            for selection in selected:
                key = str(selection).lower()
                if key in known:
                    matches.extend(p for p in paths if any(part.startswith(known[key] + '_challenger') for part in Path(p).parts))
                elif (root / selection).is_dir():
                    matches.append(str(selection))
                else:
                    raise ValueError('Unknown dataset or missing scene directory: ' + str(selection))
            if not matches:
                raise ValueError('No configured scenes match dataset in ' + name)
            item['scene_paths'] = list(dict.fromkeys(matches))
    if agent:
        selected = [agent] if isinstance(agent, str) else list(agent)
        if not selected or len(set(selected)) != len(selected) or any(name not in FORMAL_AGENTS for name in selected):
            raise ValueError('agent must select unique IDs from ' + ', '.join(FORMAL_AGENTS))
        cfg['agents'] = {'include': selected}
        settings['agents'] = {'include': selected}
    cfg = prepare_config(cfg)
    for name in ('video', 'trajectory', 'consistency', 'agent'):
        item = cfg['evaluation'].get(name, {})
        if (section in ('all', name) and item.get('enabled', True)
                and str(item.get('mode', '')).startswith('input_gt') and not item.get('scene_paths')):
            raise ValueError(name + ': configure data.scenes or supply a scene directory with dataset')
    destination = Path(output) if output is not None else Path(cfg.get('output', {}).get('root', 'outputs')) / cfg.get('output', {}).get('run_name', 'evaluation')
    destination = (root / destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    resolved = deepcopy(cfg)
    compact = cfg.get('output', {}).get('retention') == 'metrics'
    from contextlib import nullcontext
    from tempfile import TemporaryDirectory
    with (TemporaryDirectory(prefix='.work-', dir=destination) if compact else nullcontext(None)) as scratch:
        cfg['output'] = dict(cfg.get('output', {}), root=scratch or str(destination), run_name='details')
        previous = Path.cwd()
        previous_bytecode = os.environ.get('PYTHONDONTWRITEBYTECODE')
        try:
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
            os.chdir(root)
            summary, work = run(cfg, root, section, dry_run,
                                export_outputs=cfg['evaluation']['profile'] == 'research' and not compact)
        except BaseException as exc:
            if compact:
                import json
                work = Path(scratch)/'details'
                partial = {p.stem: json.loads(p.read_text()) for p in work.glob('*.json')
                           if p.stem in ('video', 'trajectory', 'consistency', 'agent', 'core_metrics',
                                         'population_trs', 'traffic_and_responsibility', 'episode_rollout', 'testing_effectiveness')}
                partial.setdefault('core_metrics', {}).update(status='failed')
                write_json(destination/'failure.json', dict(status='failed', reason=f'{type(exc).__name__}: {exc}'))
                failure_config = deepcopy(resolved)
                failure_config['paper_export'] = {'enabled': False}
                export_public(partial, work, destination, failure_config, dry_run)
            raise
        finally:
            os.chdir(previous)
            if previous_bytecode is None:
                os.environ.pop('PYTHONDONTWRITEBYTECODE',None)
            else:
                os.environ['PYTHONDONTWRITEBYTECODE'] = previous_bytecode
        payload = export_public(summary, work, destination, resolved, dry_run)
        if not compact and cfg['evaluation']['profile'] == 'leaderboard':
            shutil.rmtree(work)
    return payload


def evaluate(config=None, *, output=None, replace=False, **kwargs):
    """Run the benchmark; CLI task outputs replace only after successful staging."""
    if not replace:
        return _evaluate(config, output=output, **kwargs)
    import fcntl
    import tempfile
    cfg = load_config(config if config is not None else project_root()/'tools/configs/default.yaml',
                      profile=kwargs.get('profile'))
    configured = cfg.get('output', {})
    destination = Path(output) if output is not None else Path(configured.get('root', 'outputs'))/configured.get('run_name', 'evaluation')
    destination = (project_root()/destination).resolve()
    reuse = cfg.get('evaluation', {}).get('agent', {}).get('reuse_predictions_from')
    if reuse:
        source = (project_root()/reuse).resolve()
        if source == destination or destination in source.parents:
            raise ValueError('Prediction reuse must reference a separate result directory, not the output being replaced')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Single task writer. Work and recovery stay on the destination filesystem.
    with (destination.parent/('.'+destination.name+'.lock')).open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluation holds this task output lock') from exc
        temporary = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent))
        staging = temporary/'current'
        previous = temporary/'previous'
        try:
            result = _evaluate(cfg, output=staging, **kwargs)
            if kwargs.get('dry_run', False):
                return result
            if result.get('status') != 'completed' or any(
                    str(item.get('status', '')).lower() in ('failed', 'skipped', 'blocked')
                    for item in result.get('results', {}).values()):
                raise RuntimeError('Task failed; previous published output retained')
            for path in staging.rglob('*'):
                if path.is_symlink():
                    source = path.resolve()
                    if source == destination or destination in source.parents:
                        raise ValueError('Retained evidence refers to the output being replaced: '+str(path))
            write_json(temporary/'transaction.json', {'destination': destination, 'staging': staging,
                       'previous': previous, 'recovery': 'If destination is absent, restore previous before starting another run.'})
            if destination.exists():
                os.replace(destination, previous)
            os.replace(staging, destination)
        except BaseException:
            if previous.exists() and not destination.exists():
                os.replace(previous, destination)
            if cfg.get('output', {}).get('retention') == 'metrics' and staging.is_dir():
                failed = destination.parent/(destination.name+'.failed-'+temporary.name.rsplit('-', 1)[-1])
                os.replace(staging, failed)
            raise
        finally:
            # If rollback itself fails, retain the only previous result for recovery.
            if not previous.exists() or destination.exists():
                shutil.rmtree(temporary)
        return result

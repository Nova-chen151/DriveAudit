"""Keep computed scores and plans; move owned results, never copy input files."""
import hashlib
import json
import os
from pathlib import Path

from driveaudit_eval.utils.io import jsonable, write_csv, write_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def scene_key(value):
    value = str(value).replace('__', '/')
    path = Path(value)
    return path.parent.parent.name + '/' + path.name if path.parent.name == 'scenes' else value


def export_scene_scores(summary, output, config):
    """Index scene aggregates; full metric records remain in summary/window CSVs."""
    summary = jsonable(summary)
    expected = {scene_key(p) for section in ('video', 'trajectory', 'consistency', 'agent')
                for p in config.get('evaluation', {}).get(section, {}).get('scene_paths', [])}
    records = []
    def add(section, rows):
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = row.get('scene_id', row.get('scene'))
            if not isinstance(sid, str):
                continue
            sid = scene_key(sid)
            agent = row.get('agent', row.get('agent_name', section[6:] if section.startswith('agent.') else ''))
            expected.add(sid)
            def visit(value, path='', status=row.get('status', 'reported')):
                if isinstance(value, dict):
                    status = value.get('status', status)
                    for key, child in value.items():
                        if key not in ('provenance', 'input_gt_provenance', 'source_manifest', 'config'):
                            visit(child, (path+'.' if path else '')+key, status)
                elif value is None or isinstance(value, (int, float, bool)):
                    records.append(dict(scene_id=sid, agent=agent, section=section, metric=path, value=value, status=status))
                # Per-frame arrays are already preserved in canonical results/CSVs.
            visit(row)
    for section in ('trajectory', 'consistency', 'traffic_and_responsibility'):
        add(section, summary.get(section, {}).get('scenes', []))
    video = summary.get('video', {})
    add('video.visual_quality', video.get('visual_quality', {}).get('aggregate_metrics', []))
    core = summary.get('core_metrics', {})
    for group in ('tvc', 'trs', 'rpdms_scenes'):
        add('core.'+group, core.get(group, []))
    agents = summary.get('agent', {}).get('agents', {})
    if isinstance(agents, dict):  # Dry runs report selected agent names, without scores.
        for agent, result in agents.items():
            add('agent.'+agent, result.get('per_scene', []))
    add('episode', summary.get('episode_rollout', {}).get('episodes', []))
    testing = summary.get('testing_effectiveness', {})
    for group in ('comparison', 'quality', 'final'):
        source = testing.get(group, {})
        add('testing.'+group, source.get('scenes', source.get('rows', [])))
    present = {row['scene_id'] for row in records}
    records.extend(dict(scene_id=sid, section='evaluation', metric=None, value=None, status='unavailable')
                   for sid in sorted(expected-present))
    write_csv(output/'scene_metrics.csv', records, ['scene_id', 'agent', 'section', 'metric', 'value', 'status'])
    return sorted(expected)


def export_scores(summary, work, output, config, dry_run=False):
    work, output = Path(work).resolve(), Path(output).resolve()
    core = summary.get('core_metrics', {})
    payload = dict(core, status=core.get('status', 'dry_run' if dry_run else 'unavailable'),
                   results={key: value for key, value in summary.items() if key != 'core_metrics'})
    output.mkdir(parents=True, exist_ok=True)
    # One canonical detailed score document, without an embedded second file tree.
    (output/'summary.json').write_text(json.dumps(jsonable(summary), ensure_ascii=False,
        allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8')
    write_json(output/'metrics.json', core)
    write_csv(output/'final_score.csv', [core.get('model_row', {})])
    scenes = export_scene_scores(summary, output, config)
    artifacts, warnings = {}, []
    patterns = (
        'resolved_config.yaml', 'execution_timeline.json', 'evaluation.log',
        'gt_agent_baseline.json', 'gt_quality_baseline.json',
        'input_gt_video/provenance.json', 'input_gt_video/generated_video_validation_gate.json',
        'video_details/*.csv', 'video_details/sampling_manifest.json',
        'real_agents/input_manifest.json', 'real_agents/scene_status.json',
        'real_agents/*/predictions.jsonl', 'real_agents/*/model_identity.json',
        'real_agents/*/inference_status.json', 'real_agents/*/agent_input_manifest.json',
        'real_agents/*/sensor_access_audit.jsonl', 'real_agents/*/prediction_reuse.json',
        'real_agents/*/agent_details/*/per_frame_metrics.csv',
        'real_agents/*/agent_details/*/per_scene_metrics.csv',
        'consistency/*/*/full_chain/derived/source_manifest.json',
        'consistency/*/*/full_chain/retained_details/*.json.gz',
        'consistency/*/*/full_chain/retained_details/*.tar.gz',
        'consistency/*/*/full_chain/retained_details/*_manifest.json',
        'trajectory_scenes/*.details.json.gz',
        '_stages/*/stage.log',
    )
    roots = [work] + list((work/'gt_baselines').glob('*')) + list((work/'gt_quality').glob('*'))
    sources = [(base, base.relative_to(work), base != work, False) for base in roots]
    # Interrupted workers may have flushed plans/scores before publishing aliases.
    for base in roots:
        for stage in (base/'_stages').glob('*'):
            if stage.is_dir() and not any((base/item.name).is_symlink()
                    and (base/item.name).resolve() == item.resolve()
                    for item in stage.iterdir() if item.name not in ('resolved_config.yaml', 'job.json', 'result.json', 'stage.log')):
                sources.append((stage, base.relative_to(work), False, True))
    for base, prefix, include_summary, unpublished in sources:
        if not base.is_dir():
            continue
        for log in (base/'real_agents').glob('*/server.log'):
            status = log.with_name('inference_status.json')
            try:
                if status.is_file() and json.loads(status.read_text()).get('status') == 'COMPLETED':
                    continue
            except (OSError, ValueError, AttributeError) as exc:
                warnings.append(dict(source=str(prefix/status.relative_to(base)),
                                     reason='Incomplete inference status: '+str(exc)))
            target = output/prefix/log.relative_to(base).with_name('server_error_tail.log')
            target.parent.mkdir(parents=True, exist_ok=True)
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-65536))
                target.write_bytes(stream.read())
        for raw in (base/'real_agents').glob('*/raw_outputs.jsonl'):
            rows = []
            with raw.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        profile = row.get('output', {}).get('profile')
                    except (ValueError, AttributeError) as exc:
                        warnings.append(dict(source=str(prefix/raw.relative_to(base)), line=line_number,
                                             reason='Incomplete inference profile: '+str(exc)))
                        continue
                    if profile is not None:
                        rows.append(dict(scene_id=row['scene_id'], frame_id=row['frame_id'],
                                         agent=raw.parent.name, profile=profile))
            if rows:
                target = output/prefix/raw.relative_to(base).with_name('inference_profile.jsonl')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(''.join(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False)+'\n' for row in rows), encoding='utf-8')
        extra = ('summary.json',) if include_summary else ()
        if unpublished:
            extra += ('trajectory_scenes/*.json', 'consistency/*/*/full_chain/chain_progress.json',
                      'real_agents/results.json')
        for pattern in patterns + extra:
            for source in sorted(base.glob(pattern)):
                if not source.is_file():
                    continue
                relative = prefix/source.relative_to(base)
                target = output/relative
                if str(relative) in artifacts:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                resolved = source.resolve()
                if work in resolved.parents:
                    os.replace(resolved, target)
                else:
                    # Explicit prediction reuse is a reference, never a copied file.
                    target.symlink_to(resolved)
                artifacts[str(relative)] = dict(sha256=digest(target), bytes=target.stat().st_size,
                                                external_reference=target.is_symlink())
    if not dry_run and config.get('paper_export', {}).get('enabled'):
        from tools.export_paper_metrics import export_single_run
        export_single_run(output, config)
    package = Path(__file__).resolve().parents[1]
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'run_manifest.json':
            artifacts.setdefault(str(path.relative_to(output)), dict(sha256=digest(path), bytes=path.stat().st_size))
    write_json(output/'run_manifest.json', dict(schema='driveaudit_eval_metrics_retention_v1',
        retention='metrics', dry_run=dry_run, scene_ids=scenes,
        component_statuses={key: value.get('status') for key, value in summary.items() if isinstance(value, dict)},
        implementation_sha256={str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))},
        artifacts=artifacts, export_warnings=warnings, same_run_resume_supported=False, rgb_files_retained=0,
        evidence_storage='Score JSON/CSV and model plans retained once; paths in source records describe execution time.',
        population_only_metrics=['FID', 'FVD', 'KID', 'CMMD', 'TRS', 'QTES'],
        scene_score_scope='scene_metrics.csv indexes scalar scene aggregates; summary.json and per_frame_metrics.csv retain all measured detail; population scores are not assigned to individual scenes'))
    return payload

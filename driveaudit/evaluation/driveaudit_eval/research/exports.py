"""Optional research tables and historical evidence exports."""
import csv
import json
import zipfile
from driveaudit_eval.utils.io import jsonable, write_json
from driveaudit_eval.utils.io import write_csv

def export_results(summary, output_dir, config=None):
    settings = (config or {}).get('evaluation', {})
    profile = settings.get('profile', 'legacy')
    if profile != 'legacy':
        return export_frozen(summary, output_dir, config)
    return export_legacy(summary, output_dir)


def export_frozen(summary, output_dir, config):
    from driveaudit_eval.metrics.registry import METRICS
    from driveaudit_eval.metrics.leaderboard import export_leaderboard
    from driveaudit_eval.evaluation.profiles import public_values
    from driveaudit_eval.metrics.core import build_core
    import shutil
    settings = config['evaluation']
    core = summary.get('core_metrics') or build_core(summary, output_dir, config)
    directory = output_dir/'all_metrics_export'
    directory.mkdir(exist_ok=True)
    if settings.get('profile') == 'research':
        csv.field_size_limit(2**31-1)
        export_legacy(summary, output_dir)
        diagnostic = directory/'diagnostics'
        diagnostic.mkdir(exist_ok=True)
        for path in list(directory.iterdir()):
            if path.is_file():
                if path.suffix == '.zip':
                    path.unlink()  # This just-created duplicate archive is rebuilt below.
                else:
                    shutil.move(str(path), diagnostic/path.name)
        if not settings.get('legacy_export', {}).get('enabled', False):
            for path in diagnostic.iterdir():
                if path.suffix == '.csv':
                    with path.open(newline='', encoding='utf-8') as handle:
                        reader = csv.DictReader(handle)
                        headers = list(public_values(dict.fromkeys(reader.fieldnames or [])))
                        rows = []
                        from driveaudit_eval.metrics.registry import DEPRECATED_ALIASES
                        for row in reader:
                            if any(part in DEPRECATED_ALIASES for part in row.get('path', '').split('/')):
                                continue
                            row = public_values(row)
                            for key, value in row.items():
                                if value and value[:1] in ('{', '['):
                                    try:
                                        row[key] = public_values(json.loads(value))
                                    except (ValueError, TypeError):
                                        pass
                            rows.append(row)
                    write_csv(path, rows, headers)
                elif path.suffix == '.json':
                    write_json(path, public_values(json.loads(path.read_text(encoding='utf-8'))))
                elif path.suffix == '.jsonl':
                    records = [public_values(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in records), encoding='utf-8')
    write_json(directory/'core_metrics.json', core)
    write_json(directory/'metric_registry.json', METRICS)
    write_json(directory/'metric_status.json', core.get('metric_statuses', {}))
    rows = core.get('model_rows') or [core['model_row']]
    if settings.get('leaderboard', {}).get('generate_rank', True):
        export_leaderboard(rows, directory)
        shutil.copyfile(directory/'LEADERBOARD.md', output_dir/'leaderboard.md')
        shutil.copyfile(directory/'leaderboard.csv', output_dir/'leaderboard.csv')
    write_json(output_dir/'metrics.json', core)
    for level in ('windows', 'scenes', 'agents'):
        write_csv(directory/f'rpdms_{level}.csv', core.get('rpdms_'+level, []))
    catalog = ['# Frozen metric registry', '', '| ID | Name | Group | Tier | Direction | Aggregation |', '|---|---|---|---|---|---|']
    catalog.extend('| '+' | '.join(str(m[k]) for k in ('metric_id','display_name','group','tier','direction','aggregation'))+' |' for m in METRICS.values())
    (directory/'metric_catalog.md').write_text('\n'.join(catalog)+'\n', encoding='utf-8')
    with zipfile.ZipFile(directory/'all_metrics_and_results.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, str(path.relative_to(output_dir)))
    return directory


def export_legacy(summary, output_dir):
    directory = output_dir/'all_metrics_export'
    directory.mkdir(exist_ok=True)
    visual = summary.get('video', {}).get('visual_quality', {})
    write_csv(directory/'video_visual_quality_frame_metrics.csv', visual.get('frame_metrics', []),
              ['scene_id', 'camera', 'frame_id'])
    write_csv(directory/'video_visual_quality_scene_metrics.csv', visual.get('aggregate_metrics', []),
              ['level', 'scene_id', 'camera'])
    write_json(directory/'video_distribution_quality.json', visual.get('distribution', {}))
    traffic = summary.get('traffic_and_responsibility', {})
    actor_rows, scene_rows, event_rows, collision_rows, collision_summaries = [], [], [], [], []
    for scene in traffic.get('scenes', []):
        scene_rows.append({key: value for key, value in scene.items() if key not in ('actors', 'collision_responsibility', 'provenance')})
        for actor in scene['actors']:
            actor_rows.append(actor)
            for rule in actor['rules'].values():
                event_rows.extend(rule['events'])
        responsibility = scene.get('collision_responsibility', {})
        collision_rows.extend(responsibility.get('events', []))
        collision_summaries.append(dict(scene_id=scene['scene_id'], **responsibility.get('summary', {})))
    windows = []
    for model in traffic.get('agent_scenes', []):
        windows.extend(model['windows'])
        collision_rows.extend(model['collision_events'])
        collision_summaries.append(dict(scene_id=model['scene_id'], agent=model['agent'], **model.get('collision_summary', {})))
        for window in model['windows']:
            actor_rows.append(dict(window['traffic'], agent=window['agent'], decision_frame_id=window['frame_id'], scope='agent_open_loop_plan'))
            for rule in window['traffic']['rules'].values():
                event_rows.extend(dict(event, agent=window['agent'], decision_frame_id=window['frame_id']) for event in rule['events'])
    write_csv(directory/'traffic_compliance_actor_metrics.csv', actor_rows, ['scene_id', 'actor_id'])
    write_csv(directory/'traffic_compliance_scene_metrics.csv', scene_rows, ['scene_id'])
    write_csv(directory/'traffic_compliance_event_metrics.csv', event_rows, ['scene_id', 'actor_id', 'rule_id'])
    write_csv(directory/'collision_responsibility_events.csv', collision_rows, ['scene_id', 'collision_event_id'])
    write_csv(directory/'collision_responsibility_summary.csv', collision_summaries, ['scene_id'])
    write_csv(directory/'agent_traffic_compliance_window_metrics.csv', windows, ['scene_id', 'agent', 'frame_id'])
    write_csv(directory/'agent_traffic_compliance_overall_metrics.csv', traffic.get('agent_overall', []), ['agent'])
    write_json(directory/'traffic_and_responsibility_summary.json', traffic)
    for level in ('frame', 'scene'):
        merged = []
        for path in sorted((output_dir/'real_agents').glob(f'*/agent_details/*/per_{level}_metrics.csv')):
            with path.open(newline='', encoding='utf-8') as handle:
                merged.extend(csv.DictReader(handle))
        write_csv(directory/f'agent_all_{level}_metrics.csv', merged, ['agent', 'scene_id'])
    v2 = traffic.get('responsibility_v2')
    if v2:
        from driveaudit_eval.research.responsibility_reporting import export_v2
        export_v2(v2, directory)
    v3 = traffic.get('responsibility_v3')
    if v3:
        from driveaudit_eval.research.responsibility_reporting import export_v3
        export_v3(v3, directory)
    episode_catalog = ''
    if summary.get('episode_rollout', {}).get('status') == 'completed':
        from driveaudit_eval.research.episode_reporting import export as export_episode
        episode_catalog = export_episode(summary['episode_rollout'], directory)
        if summary['episode_rollout'].get('config', {}).get('terminal_execution_mode') == 'horizon_commit':
            from driveaudit_eval.research.episode_diagnostics import export as export_episode_v2
            episode_catalog += '\n'+export_episode_v2(summary['episode_rollout'], directory)
    if summary.get('episode_rollout',{}).get('config',{}).get('route_projection',{}).get('enabled',False):
        from driveaudit_eval.research.route_reporting import export as export_final_route
        episode_catalog += '\n'+export_final_route(summary['episode_rollout'],directory,summary)
    rows = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, path+[str(key)])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path+[str(index)])
        else:
            rows.append(dict(path='/'.join(path), value=value))
    visit(jsonable(summary), [])
    write_csv(directory/'all_output_values.csv', rows, ['path', 'value'])
    catalog = ['# Metric catalog', '',
        'All output leaves (including metadata) are preserved in all_output_values.csv.',
        'Missing values are null; consult status and reason. Original module JSON remains authoritative.', '',
        '| Module | Result keys |', '|---|---|']
    for name, result in summary.items():
        catalog.append('| '+name+' | '+', '.join(result)+' |')
    from driveaudit_eval.metrics.registry import METRICS
    catalog.extend(['', '| Metric | Group | Tier | Aggregation |', '| --- | --- | --- | --- |'])
    catalog.extend('| ' + ' | '.join(str(metric[key]) for key in ('display_name', 'group', 'tier', 'aggregation')) + ' |'
                   for metric in METRICS.values())
    catalog.append('Responsibility is model-based pairwise evidence, not legal liability. Detailed field contracts are in docs/metrics.md and docs/RESPONSIBILITY.md.')
    if episode_catalog:
        catalog.extend(['', episode_catalog])
    (directory/'metric_catalog.md').write_text('\n'.join(catalog)+'\n', encoding='utf-8')
    with zipfile.ZipFile(directory/'all_metrics_and_results.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, 'all_metrics_export/'+path.name)
        for path in sorted(output_dir.rglob('*')):
            relative = path.relative_to(output_dir)
            if relative.parts[0] != 'all_metrics_export' and path.is_file() and path.suffix in ('.json', '.csv', '.jsonl', '.md', '.yaml'):
                archive.write(path, str(relative))
    return directory

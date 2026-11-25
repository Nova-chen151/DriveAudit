"""Paired camera-only GT inference, baseline reuse and quality-gated final score."""
from copy import deepcopy
import json
from pathlib import Path

from driveaudit_eval.agents.registry import FORMAL_AGENTS
from driveaudit_eval.agents.provenance import (digest, model_identity, protocol_identity,
                                         request_seeds, audit_inference)
from driveaudit_eval.metrics.testing_effectiveness import (BaselineCache, QUALITY_FIELDS,
    content_hash, final_score, paired_comparison, quality_validity, record_errors, unit)
from driveaudit_eval.utils.io import write_json


def scene_key(value):
    value = str(value).replace('__', '/')
    path = Path(value)
    return path.parent.parent.name + '/' + path.name if path.parent.name == 'scenes' else value


def reference_paths(config, root):
    settings = config['evaluation']
    options = settings.get('testing_effectiveness', {})
    gt_root = Path(root)/options.get('gt_root', 'challenge/input-gt')
    explicit = options.get('gt_scenes', {})
    result = []
    for value in settings['agent']['scene_paths']:
        path = Path(root)/value
        sid = path.parent.parent.name + '/' + path.name
        result.append(str(Path(root)/explicit[sid]) if sid in explicit else
                      str(gt_root/path.parent.parent.name/'scenes'/path.name))
    return result


def scene_evidence(scene):
    """Decode the full six-camera bundle and validate native timestamp support."""
    from PIL import Image
    from driveaudit_eval.data.scene_input import CANONICAL_CAMERA_NAMES
    import numpy as np
    directory = Path(scene['scene_path'])
    calibration = json.loads((directory/'calibration/cameras.json').read_text())['cameras']
    sizes = {c['camera']:c['challenge_image_size'] for c in calibration}
    timeline = json.loads((directory/'timeline.json').read_text())['frames']
    manifest = json.loads((directory/'future/rgb_manifest.json').read_text())['images']
    rows = {(r['frame_index'],r['camera']):r for r in manifest}
    expected = {(i,c) for i in range(10,90) for c in CANONICAL_CAMERA_NAMES}
    if len(rows) != len(manifest) or set(rows) != expected:
        raise ValueError('Incomplete or duplicate RGB timestamps: ' + str(directory))
    if set(sizes) != set(CANONICAL_CAMERA_NAMES) or len(calibration) != 6:
        raise ValueError('Six-camera protocol invalid')
    images = {}
    for camera in CANONICAL_CAMERA_NAMES:
        exposures = []
        for kind, indices in (('history',range(10)),('future',range(10,90))):
            paths = {p.stem:p for p in (directory/kind/'rgb'/camera).glob('*.jpg')}
            if set(paths) != {f'{i:06d}' for i in indices}:
                raise ValueError('Required RGB incomplete: ' + str(directory/kind/camera))
            for i in indices:
                path = paths[f'{i:06d}']
                with Image.open(path) as image:
                    image.load()
                    if list(image.size) != sizes[camera]:
                        raise ValueError('RGB size differs from calibration')
                images[str(path.relative_to(directory))] = digest(path)
                timing = rows[i,camera] if i >= 10 else timeline[i]['cameras'][camera]
                exposure = timing.get('exposure_time_us')
                if isinstance(exposure,bool) or not isinstance(exposure,(int,float)) or not np.isfinite(exposure):
                    raise ValueError('Missing/nonfinite native exposure')
                exposures.append(exposure)
        # Repeated stored native exposures are allowed and remain explicit.
        if np.any(np.diff(exposures) < 0):
            raise ValueError('Nonmonotonic camera timestamps')
    metadata = {str(p.relative_to(directory)):digest(p) for folder in ('calibration','map')
                for p in sorted((directory/folder).rglob('*')) if p.is_file()}
    for name in ('timeline.json','scene_context.json','provenance.json','generation_provenance.json',
                 'future/rgb_manifest.json','history/agents.csv','future/future_trajectories.csv'):
        if (directory/name).is_file():
            metadata[name] = digest(directory/name)
    conditioning = []
    for frame in scene['frames']:
        item = deepcopy(frame)
        item['model_input'].pop('camera_images')
        conditioning.append(item)
    history = dict(images={k:v for k,v in images.items() if k.startswith('history/')},
        timeline=timeline[:10], agents=digest(directory/'history/agents.csv'),
        cameras=digest(directory/'calibration/cameras.json'),
        camera_poses=digest(directory/'calibration/history_camera_poses.json'),
        ego_poses=json.loads((directory/'calibration/reference_ego_poses.json').read_text())['ego_to_anchor'][:10],
        maps={k:v for k,v in metadata.items() if k.startswith('map/')})
    return dict(scene_hash=content_hash(dict(images=images, metadata=metadata)),
                rgb_hash=content_hash(images), conditioning_hash=content_hash(conditioning),
                history_hash=content_hash(history),
                required_rgb_complete=True, six_camera_valid=True, timestamps_valid=True)


def scoring_hash(config):
    package = Path(__file__).resolve().parent
    # Agent score identity excludes presentation and visual-quality aggregation.
    paths = [p for group in ('agents','data','trajectory','responsibility','utils')
             for p in (package/group).rglob('*.py')]
    paths += [package/name for name in ('metrics/core.py','traffic_compliance.py',
              'traffic_evaluator.py','episode_rollout.py','route_projection.py')]
    sources = {str(p.relative_to(package)):digest(p) for p in sorted(paths)}
    agent = {k:v for k,v in config['evaluation']['agent'].items()
             if k not in ('scene_paths','agents','resume','reuse_predictions_from')}
    return content_hash(dict(agent=agent, core=config['evaluation']['core_metrics'],
        traffic=config.get('traffic_compliance'), responsibility=config.get('responsibility'),
        responsibility_v2=config.get('responsibility_v2'), source=sources))


def record_identity(agent, scene, windows, evidence, gt_evidence, identity, scorer, pairing_version=None):
    selected = [w['frame_id'] for w in windows if w['scene_id'] == scene['scene_id']]
    frames = [f for f in scene['frames'] if f['frame_id'] in selected]
    if len(frames) != len(selected) or not frames:
        raise ValueError('Invalid frozen decision support')
    return dict(dataset=scene['dataset'], scene=scene_key(scene['scene_id']), agent=agent,
        checkpoint_hash=identity['checkpoint_hash'], agent_config_hash=identity['agent_config_hash'],
        camera_only_protocol_hash=protocol_identity(agent), scene_hash=gt_evidence['scene_hash'],
        scoring_hash=scorer, conditioning_hash=evidence['history_hash' if pairing_version == 'paper_population_v1' else 'conditioning_hash'],
        decision_frames=[f['frame_id'] for f in frames], decision_timestamps=[f['timestamp'] for f in frames],
        request_seeds=request_seeds(agent,scene['scene_id'],frames), rgb_hash=evidence['rgb_hash'],
        required_rgb_complete=evidence['required_rgb_complete'], six_camera_valid=evidence['six_camera_valid'],
        timestamps_valid=evidence['timestamps_valid'])


def measured_record(identity, summary, directory, scene, model):
    result = dict(identity)
    directory = Path(directory)
    status = json.loads((directory/identity['agent']/'inference_status.json').read_text())
    if status.get('manifest_sha256') != digest(directory/'input_manifest.json'):
        raise ValueError('Input manifest is not bound to the recorded inference')
    audit = audit_inference(directory, identity['agent'], scene, model)
    windows = [w for w in summary['core_metrics']['rpdms_windows'] if w['agent'] == identity['agent']
               and scene_key(w['scene_id']) == identity['scene']]
    if sorted(w['frame_id'] for w in windows) != sorted(identity['decision_frames']):
        raise ValueError('RPDMS windows differ from paired decision support')
    scores = [unit(w.get('rpdms'), 'window_rpdms') for w in windows]
    # Every frozen decision is required on both sides; no model-specific intersection.
    score = sum(scores)/len(scores)
    result.update(audit, rpdms=score, window_rpdms=[dict(frame_id=w['frame_id'],rpdms=w['rpdms']) for w in windows])
    return result


def quality_support(summary, expected, output):
    """Reject scene omissions in model aggregates; retain existing component kernels."""
    reasons = []
    core = summary.get('core_metrics', {})
    for key in ('tvc','trs'):
        rows = core.get(key, [])
        keys = [scene_key(r['scene_id']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            reasons.append(key + ':scene_support_mismatch')
        for row in rows:
            if row.get(key) is None or any(v is None for v in row.get('components',{}).values()):
                reasons.append(key + ':missing_scene_component:' + scene_key(row['scene_id']))
    video = summary.get('video', {})
    vq = video.get('visual_quality', {})
    frames = vq.get('frame_metrics', [])
    keys = [(scene_key(r['scene_id']),r['camera'],r['frame_id']) for r in frames]
    if len(keys) != len(set(keys)) or {k[0] for k in keys} != expected:
        reasons.append('clip_iqa:scene_support_mismatch')
    if any(r.get('clipiqa') is None for r in frames):
        reasons.append('clip_iqa:missing_frame_score')
    pairs = vq.get('metrics',{}).get('dino_temporal',{}).get('pairs',[])
    for interval in ('short','medium','long'):
        if {scene_key(r['scene_id']) for r in pairs if r['interval'] == interval} != expected:
            reasons.append('dino_t:missing_scene_interval:' + interval)
    # The metric engine exposes measured per-frame geometry in its standard CSV.
    import csv
    csvs = list(Path(output).glob('video_details/**/camera_pair_metrics.csv'))
    if len(csvs) != 1:
        reasons.append('epi3:missing_frame_evidence')
    else:
        with csvs[0].open() as stream:
            geometry = list(csv.DictReader(stream))
        valid = [r for r in geometry if r.get('epipolar_inlier_ratio_3px') not in ('',None,'nan')]
        if {scene_key(r['scene_id']) for r in valid} != expected:
            reasons.append('epi3:missing_scene_score')
        try:
            from driveaudit_eval.metrics.core_metric_gate import audit_video_sources
            validation = json.loads((Path(output)/'input_gt_video/generated_video_validation_gate.json').read_text())
            provenance = json.loads((Path(output)/'input_gt_video/provenance.json').read_text())
            audit = audit_video_sources(validation,provenance,vq,geometry,expected)
            reasons += [key + ':invalid_source_support' for key,value in audit['source_valid'].items() if not value]
        except (KeyError,ValueError,FileNotFoundError) as exc:
            reasons.append('quality_source_audit:' + str(exc))
    return reasons


def evaluate_testing(summary, output, config, context):
    from driveaudit_eval.agents.real_agents import prepare
    from driveaudit_eval.evaluation.runner import run
    options = config['evaluation'].get('testing_effectiveness', {})
    version = options.get('version', 'gt_relative_v1')
    if version not in ('gt_relative_v1', 'paper_population_v1'):
        raise ValueError('Unknown QTES definition version: '+str(version))
    output = Path(output)
    gt_directory = output/'gt_inputs'
    gt_directory.mkdir()
    agent_config = deepcopy(config['evaluation']['agent'])
    agent_config['scene_paths'] = reference_paths(config,context['root'])
    agent_config['agents'] = list(FORMAL_AGENTS)
    agent_config.pop('reuse_predictions_from',None)
    gt_manifest = prepare(agent_config,context,gt_directory)
    generated_manifest = json.loads((output/'real_agents/input_manifest.json').read_text())
    gt_scenes = {scene_key(s['scene_id']):s for s in gt_manifest['scenes']}
    generated_scenes = {scene_key(s['scene_id']):s for s in generated_manifest['scenes']}
    if len(gt_scenes) != len(gt_manifest['scenes']) or set(gt_scenes) != set(generated_scenes):
        raise ValueError('GT/generated scene identity mismatch')
    if gt_manifest['common_windows'] != generated_manifest['common_windows']:
        raise ValueError('GT/generated frozen decision windows mismatch')
    expected = [(s['dataset'],sid) for sid,s in gt_scenes.items()]
    cache = BaselineCache(Path(context['root'])/options.get('baseline_cache','cache/gt_agent_baseline.json')
                          if options.get('retain_intermediates', True) else None)
    scorer = scoring_hash(config)
    models = {agent:model_identity(agent) for agent in FORMAL_AGENTS}
    gt_evidence = {sid:scene_evidence(s) for sid,s in gt_scenes.items()}
    generated_evidence = {sid:scene_evidence(s) for sid,s in generated_scenes.items()}
    gt, generated, pending = [], [], {}
    cache_hits = 0
    for sid, scene in generated_scenes.items():
        if scene['dataset'] != gt_scenes[sid]['dataset'] or scene['scene_token'] != gt_scenes[sid]['scene_token']:
            raise ValueError('Dataset/source-scene pairing mismatch')
        pairing_key = 'history_hash' if version == 'paper_population_v1' else 'conditioning_hash'
        if gt_evidence[sid][pairing_key] != generated_evidence[sid][pairing_key]:
            raise ValueError('GT/generated paired '+pairing_key+' differs')
        for agent in FORMAL_AGENTS:
            identity = record_identity(agent,scene,generated_manifest['common_windows'],generated_evidence[sid],gt_evidence[sid],models[agent],scorer,version)
            generated.append(measured_record(identity,summary,output/'real_agents',scene,models[agent]))
            identity = record_identity(agent,gt_scenes[sid],gt_manifest['common_windows'],gt_evidence[sid],gt_evidence[sid],models[agent],scorer,version)
            cached = cache.get(identity)
            if cached is not None:
                gt.append(cached)
                cache_hits += 1
            else:
                pending.setdefault(agent,[]).append((sid,identity))
    for agent, missing in pending.items():
        context['logger'].info('[QTES] Computing GT baseline %s on %d scene(s)',agent,len(missing))
        baseline_config = deepcopy(config)
        baseline_config['evaluation']['testing_effectiveness'] = {'enabled':False}
        baseline_config['evaluation']['core_metrics']['trs']['fresh_population'] = False
        baseline_config['evaluation'].pop('reuse_results',None)
        baseline_config['output'] = dict(config.get('output', {}), root=str(output/'gt_baselines'), run_name=agent)
        baseline_config['agents'] = {'include':[agent]}
        baseline_config['evaluation']['agents'] = {'include':[agent]}
        for section in ('video','trajectory','consistency','agent'):
            baseline_config['evaluation'][section]['scene_paths'] = [gt_scenes[sid]['scene_path'] for sid,_ in missing]
        baseline_config['evaluation']['agent'] = dict(agent_config, agents=[agent],
            scene_paths=[gt_scenes[sid]['scene_path'] for sid,_ in missing])
        baseline_summary, directory = run(baseline_config,context['root'],section='agent',export_outputs=False)
        for sid,identity in missing:
            row = measured_record(identity,baseline_summary,directory/'real_agents',gt_scenes[sid],models[agent])
            cache.put(row)
            gt.append(row)
    comparison = paired_comparison(gt,generated,expected,eps=options.get('eps',1e-6))
    if version == 'paper_population_v1':
        comparison['input_identity'] = {sid:dict(generated=generated_evidence[sid],reference=gt_evidence[sid]) for sid in gt_scenes}
    from driveaudit_eval.quality import evaluate_paired_quality
    quality = evaluate_paired_quality(summary,output,config,context,gt_scenes,generated_scenes,gt_evidence,generated_evidence)
    selection_errors = []
    # All sections must evaluate the same scene set; partial/reused foreign quality cannot rescue a run.
    expected_paths = {str((Path(context['root'])/p).resolve()) for p in config['evaluation']['agent']['scene_paths']}
    for section in ('video','trajectory','consistency'):
        actual = {str((Path(context['root'])/p).resolve()) for p in config['evaluation'][section]['scene_paths']}
        if actual != expected_paths:
            selection_errors.append(section + ':different_scene_selection')
    if version == 'paper_population_v1':
        from driveaudit_eval.metrics.population_testing import aggregate
        if selection_errors:
            raise ValueError(';'.join(selection_errors))
        final = aggregate(summary.get('core_metrics',{}).get('trs_population',{}),comparison,quality,sorted(gt_scenes))
    else:
        reasons = comparison['invalid_reason'] + quality['invalid_reason'] + selection_errors
        final = final_score(comparison['statistics']['mean'],quality['quality_retention_score'],reasons)
    selected_entries = {cache.key(row):dict(record=row,sha256=content_hash(row)) for row in gt}
    write_json(output/'gt_agent_baseline.json',dict(schema='qtes_gt_baseline_v1',entries=selected_entries))
    return dict(status=final['status'], comparison=comparison, quality=quality, final=final,
                generated_records=generated, gt_records=gt, cache_hits=cache_hits, cache_misses=len(gt)-cache_hits,
                gt_rgb_self_comparison=all(gt_evidence[sid]['rgb_hash']==generated_evidence[sid]['rgb_hash'] for sid in gt_scenes),
                scope='Paired offline RPDMS; GT RGB self-comparison verifies implementation, not generation efficacy')


def attach_testing(summary, output, config, context):
    """Always emit an explicit invalid result if any evidence or baseline is absent."""
    try:
        result = evaluate_testing(summary,output,config,context)
    except Exception as exc:
        context['logger'].exception('[QTES] INVALID')
        quality = quality_validity(summary.get('core_metrics',{}).get('model_row',{}))
        reason = type(exc).__name__ + ':' + str(exc)
        manifest_path = Path(output)/'real_agents/input_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        expected = [(s['dataset'],scene_key(s['scene_id'])) for s in manifest.get('scenes',[])]
        comparison = paired_comparison([],[],expected,eps=config['evaluation'].get('testing_effectiveness',{}).get('eps',1e-6))
        result = dict(status='invalid',quality=quality,final=final_score(None,None,[reason]),
                      comparison=comparison)
        if config['evaluation'].get('testing_effectiveness',{}).get('version') == 'paper_population_v1':
            result.update(status='unavailable',final=dict(qtes=None,testing_effectiveness=None,
                quality_retention_score=None,qtes_definition_version='paper_population_v1',
                validity=0,invalid_reason=[reason],status='unavailable'))
    core = summary.setdefault('core_metrics',{})
    final = result['final']
    core.setdefault('model_row',{}).update({key:final[key] for key in ('testing_effectiveness','qtes','validity','invalid_reason','qtes_definition_version')})
    core['model_row'].update({key:value for key,value in result['quality'].items()
                              if key.endswith(('_absolute_score','_retention_score','_retention_coverage'))})
    result['qtes_definition_version'] = final['qtes_definition_version']
    core.setdefault('metric_statuses',{}).update(testing_effectiveness='valid' if final['testing_effectiveness'] is not None else 'unavailable',
                                             qtes=final['status'])
    if final['validity'] == 0:
        core['status'] = 'invalid'
    summary['testing_effectiveness'] = result
    write_json(Path(output)/'testing_effectiveness.json',result)
    return result

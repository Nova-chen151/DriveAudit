"""Quality evidence adapter and separate, hash-bound paired GT cache."""
from copy import deepcopy
import csv
import hashlib
import importlib
import json
from pathlib import Path
from statistics import mean

from driveaudit_eval.agents.provenance import digest
from driveaudit_eval.metrics.core import number, build_core, _valid_source, _parent_ok
from driveaudit_eval.metrics.core_metric_gate import audit_video_sources
from driveaudit_eval.metrics.quality_retention import (QUALITY_VERSION, QUALITY_FIELDS, EXPORT_NAMES,
    QualityBaselineCache, aggregate_retention, quality_retention, reference_audit, motion_guard)
from driveaudit_eval.metrics.testing_effectiveness import content_hash, quality_validity
from driveaudit_eval.utils.io import write_json


def decoded_hashes(scene):
    from PIL import Image
    hashes = {}
    for path in sorted((Path(scene)/'future/rgb').glob('*/*.jpg')):
        with Image.open(path) as image:
            pixels = image.convert('RGB')
            hashes[path.parent.name+'/'+str(int(path.stem))] = hashlib.sha256(pixels.tobytes()).hexdigest()
    return hashes


def scene_quality(summary, output, scenes, fields=QUALITY_FIELDS):
    """Preserve existing frame mean, equal temporal-span means and pair mean."""
    from driveaudit_eval.testing import scene_key
    core, video = summary['core_metrics'], summary['video']
    vq = video.get('visual_quality',{})
    with (Path(output)/'video_details/camera_pair_metrics.csv').open() as stream:
        geometry = list(csv.DictReader(stream))
    validation = json.loads((Path(output)/'input_gt_video/generated_video_validation_gate.json').read_text())
    provenance = json.loads((Path(output)/'input_gt_video/provenance.json').read_text())
    expected = set(scenes)
    for rows,key in ((validation['scenes'],'scene_id'),(provenance['scenes'],'scene_id')):
        ids = [scene_key(r[key]) for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=expected:
            raise ValueError('Quality scene support mismatch: '+key)
    if validation['status'] != 'VALID' or any(r['status'] != 'VALID' or r['decoded_frames'] != 540 for r in validation['scenes']):
        raise ValueError('Quality RGB validation failed')
    for key in ('tvc','trs'):
        ids = [scene_key(r['scene_id']) for r in core.get(key,[])]
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ValueError('Duplicate or foreign quality scene: '+key)
    for rows in (vq.get('frame_metrics',[]),vq.get('metrics',{}).get('dino_temporal',{}).get('pairs',[]),geometry):
        if any(scene_key(r['scene_id']) not in expected for r in rows):
            raise ValueError('Foreign scene in quality measurements')
    records = {}
    for sid, scene in scenes.items():
        if scene.get('scene_path'):
            recorded = next(r for r in provenance['scenes'] if scene_key(r['scene_id']) == sid)['rgb_manifest']
            current = json.loads((Path(scene['scene_path'])/'future/rgb_manifest.json').read_text())
            if recorded != current:
                raise ValueError('Quality RGB manifest differs from current scene: '+sid)
        frames = [r for r in vq.get('frame_metrics',[]) if scene_key(r['scene_id']) == sid]
        pairs = [r for r in vq.get('metrics',{}).get('dino_temporal',{}).get('pairs',[]) if scene_key(r['scene_id']) == sid]
        epi = [r for r in geometry if scene_key(r['scene_id']) == sid]
        subset_vq = dict(frame_metrics=frames,metrics=dict(dino_temporal=dict(pairs=pairs)))
        support = audit_video_sources(
            dict(scenes=[r for r in validation['scenes'] if scene_key(r['scene_id']) == sid]),
            dict(scenes=[r for r in provenance['scenes'] if scene_key(r['scene_id']) == sid]),subset_vq,epi,{sid})
        def average(values):
            values = [number(v) for v in values]
            return mean(values) if values and all(v is not None for v in values) else None
        span_means = [average(r.get('cosine_similarity') for r in pairs if r['interval']==span)
                      for span in ('short','medium','long')]
        scores = dict(clip_iqa=average(r.get('clipiqa') for r in frames),dino_t=average(span_means),
                      epi_3px=average(float(r['epipolar_inlier_ratio_3px']) for r in epi
                                     if number(r.get('epipolar_inlier_ratio_3px')) is not None))
        coverage = dict(clip_iqa=support['image_count']/480,
                        dino_t=len({(r['camera'],r['interval']) for r in pairs})/18,
                        epi_3px=support['epi_coverage'])
        sources = dict(clip_iqa=vq.get('metrics',{}).get('clipiqa'),
                       dino_t=vq.get('metrics',{}).get('dino_temporal'),
                       epi_3px=video.get('geometry_consistency'))
        statuses = {k:'valid' if support['source_valid'][k] and scores[k] is not None
                    and _parent_ok(video) and _valid_source(sources[k])
                    and not str(sources[k].get('metric_validity','')).startswith('invalid')
                    else 'invalid' for k in scores}
        for key in (k for k in ('tvc','trs') if k in fields):
            row = next((r for r in core.get(key,[]) if scene_key(r['scene_id']) == sid),{})
            scores[key],coverage[key] = row.get(key),row.get(key+'_coverage',0)
            statuses[key] = row.get(key+'_status','invalid')
        records[sid] = dict(dataset=scene['dataset'],scene_id=sid,scores=scores,score_space='raw',
            metric_coverage=coverage,dimension_status=statuses,source_support=support,temporal_pairs=pairs)
        if 'trs' not in fields:
            tvc = next((r for r in core.get('tvc',[]) if scene_key(r['scene_id']) == sid),{})
            records[sid]['tvc_component_mask'] = sorted(k for k,v in tvc.get('components',{}).items() if v is not None)
    return records


def quality_signature(config):
    """Only raw quality kernels/config/resources identify a GT quality baseline."""
    package = Path(__file__).resolve().parent
    paths = [p for group in ('video','consistency','trajectory','data','utils')
             for p in (package/group).rglob('*.py')]
    paths += [package/'metrics/core.py',package/'metrics/core_metric_gate.py',package/'quality.py']
    sources = {str(p.relative_to(package)):digest(p) for p in sorted(paths)}
    sections = {key:{k:v for k,v in config['evaluation'][key].items()
                if k not in ('scene_paths','reference_scene_paths','resume','paths','output','scene_workers')}
                for key in ('video','trajectory','consistency')}
    resources = {}
    def weights(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('weights_path','checkpoint') and isinstance(item,str) and Path(item).is_file():
                    resources[item] = digest(item)
                weights(item)
        elif isinstance(value,list):
            for item in value: weights(item)
    weights(sections)
    # CLIP-IQA's defaults live outside the project and are not explicit config keys.
    for name in ('pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth','clip/RN50.pt'):
        import os
        path = Path(os.environ.get('TORCH_HOME', Path.home()/'.cache/torch'))/'hub'/name
        if path.is_file(): resources[str(path)] = digest(path)
    return dict(metric_implementation_hash=content_hash(dict(sources=sources,resources=resources)),
        metric_config_hash=content_hash(dict(sections=sections,core={key:config['evaluation']['core_metrics'][key]
                                                                  for key in ('tvc','trs')})))


def quality_identity(config, scene, evidence, signature=None):
    return dict(dataset=scene['dataset'],scene_id=scene['scene_id'].replace('__','/'),
        scene_hash=evidence['scene_hash'],rgb_hash=evidence['rgb_hash'],
        manifest_hash=digest(Path(scene['scene_path'])/'future/rgb_manifest.json'),
        **(signature or quality_signature(config)))



def evaluate_quality_only(config, paths, output, context):
    """Three existing evaluator entrypoints; Agent inference is never dispatched."""
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    result = {}
    cfg = deepcopy(config)
    if cfg['evaluation'].get('testing_effectiveness',{}).get('version') == 'paper_population_v1':
        cfg['evaluation']['core_metrics']['trs']['fresh_population'] = False
        # Paper local quality does not consume trajectory-family scores or TRS.
        # The primary evaluation already computes and binds population TRS and
        # every trajectory family; keep that independent evidence authoritative.
        cfg['evaluation']['trajectory']['enabled'] = False
        # This reference measures local QTES fields (CLIP-IQA, DINO, Epi, TVC).
        # TVC consumes unchanged MTPC primary metrics. Independent TCVV research
        # diagnostics are measured by each primary evaluation, including GT.
        consistency = cfg['evaluation']['consistency']
        consistency['profile'] = 'core'
        for name, backend in consistency.get('backends', {}).items():
            backend['enabled'] = name == 'mtpc'
    local = dict(context,output_dir=output)
    for section in ('video','trajectory','consistency'):
        options = cfg['evaluation'][section]
        options.update(scene_paths=[str(p) for p in paths],resume=False)
        options.pop('reference_scene_paths',None)
        if section == 'video':
            # Distribution metrics do not contribute to quality retention.
            for name in ('fvd','fid','temporal_warp','history_future'):
                options.setdefault('metrics',{}).setdefault(name,{})['enabled'] = False
            for name in ('cmmd','kid','musiq','dover'):
                options.setdefault('visual_quality',{}).setdefault(name,{})['enabled'] = False
    if cfg.get('execution', {}).get('enabled', False):
        from driveaudit_eval.evaluation.parallel import run_stages
        cfg['evaluation']['agent']['enabled'] = False
        cfg.setdefault('traffic_compliance', {})['enabled'] = False
        cfg.setdefault('episode_rollout', {})['enabled'] = False
        result = run_stages(cfg, local)
    else:
        for section in ('video','trajectory','consistency'):
            options = cfg['evaluation'][section]
            result[section] = importlib.import_module('driveaudit_eval.'+section+'.evaluator').evaluate(options,local)
            write_json(output/(section+'.json'),result[section])
    result['core_metrics'] = build_core(result,output,cfg)
    write_json(output/'summary.json',result)
    return result


def evaluate_paired_quality(summary, output, config, context, gt_scenes, generated_scenes, gt_evidence, generated_evidence):
    if config['evaluation'].get('testing_effectiveness',{}).get('version') == 'paper_population_v1':
        from driveaudit_eval.metrics.population_testing import evaluate_local_quality
        return evaluate_local_quality(summary,output,config,context,gt_scenes,generated_scenes,gt_evidence,generated_evidence)
    options = config['evaluation'].get('testing_effectiveness',{}).get('quality_retention',{})
    if options.get('version',QUALITY_VERSION) != QUALITY_VERSION:
        raise ValueError('Unsupported QTES quality definition version')
    cache = QualityBaselineCache(Path(context['root'])/options.get('baseline_cache','cache/gt_quality_baseline.json')
                                if options.get('retain_intermediates', True) else None)
    generated = scene_quality(summary,output,generated_scenes)
    reference = {}; hits = 0
    signature = quality_signature(config)
    for sid,scene in gt_scenes.items():
        identity = quality_identity(config,scene,gt_evidence[sid],signature)
        record = cache.get(identity)
        if record is not None:
            hits += 1
        else:
            if gt_evidence[sid]['scene_hash'] == generated_evidence[sid]['scene_hash']:
                record = deepcopy(generated[sid])
                source = 'identical_GT_scene_and_RGB; existing_measured_generated_quality'
            else:
                work = Path(output)/'gt_quality'/sid.replace('/','__')
                measured = evaluate_quality_only(config,[scene['scene_path']],work,context)
                record = scene_quality(measured,work,{sid:scene})[sid]
                source = str(work)
            record.update(identity,decoded_rgb_hashes=decoded_hashes(scene['scene_path']),source=source)
            record.update({EXPORT_NAMES[k]+'_gt':v for k,v in record['scores'].items()})
            cache.put(record)
        reference[sid] = record
    audit_path = options.get('reference_validation')
    if audit_path:
        audit = json.loads((Path(context['root'])/audit_path).read_text())
        expected_hash = options.get('reference_validation_hash')
        if expected_hash and digest(Path(context['root'])/audit_path) != expected_hash:
            raise ValueError('GT reference validation hash mismatch')
    else:
        # Initial construction is independent of generated models and records its scope.
        audit = reference_audit(list(reference.values()),options.get('epsilons'))
    rows = []
    for sid,gen in generated.items():
        ref = reference[sid]
        pairs = ref['temporal_pairs']
        pair_key = lambda p:(p['camera'],p['frame_id'],p['next_frame_id'],p['interval'])
        if {pair_key(p) for p in pairs} != {pair_key(p) for p in gen['temporal_pairs']}:
            gen['dimension_status']['dino_t'] = 'invalid'
        hashes = (ref['decoded_rgb_hashes'] if gt_evidence[sid]['rgb_hash'] == generated_evidence[sid]['rgb_hash']
                  else decoded_hashes(generated_scenes[sid]['scene_path']))
        guard = motion_guard(ref['decoded_rgb_hashes'],hashes,pairs)
        rows.append(quality_retention(gen,ref,audit,options.get('epsilons'),guard))
    result = {**quality_validity(summary['core_metrics']['model_row']),**aggregate_retention(rows)}
    result.update(reference_audit=audit,cache_hits=hits,cache_misses=len(reference)-hits)
    write_json(Path(output)/'gt_quality_baseline.json',dict(schema=cache.SCHEMA,
        qtes_definition_version=QUALITY_VERSION,
        entries={cache.key(r):dict(record=r,sha256=content_hash(r)) for r in reference.values()}))
    return result

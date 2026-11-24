"""Small, additive Metric Freeze v1 aggregators. Original scores are never mutated."""
import csv
import math
from pathlib import Path
from driveaudit_eval.metrics.registry import CORE_IDS, METRICS
from driveaudit_eval.metrics.capabilities import MAP_REASON

TVC_FIELDS = {
    'support': 'presence.persistent_agent_visual_support_rate',
    'center': 'projection_geometry.normalized_projected_center_error',
    'motion': 'image_motion.motion_direction_agreement',
    'identity': 'identity.identity_persistence',
    'multiview': 'multiview.camera_support_recall',
}
from driveaudit_eval.agents.registry import FORMAL_AGENTS

from driveaudit_eval.utils.aggregation import metric_number as number

def bounded(value, lower=0, upper=1):
    value = number(value)
    if value is not None and not lower <= value <= upper:
        raise ValueError('Metric outside [%s,%s]: %s' % (lower,upper,value))
    return value

from driveaudit_eval.utils.aggregation import metric_mean as mean

def _weighted(name, keys, components, weights, coverage_threshold):
    if set(components)-set(keys):
        raise ValueError('Unknown %s dimensions' % name)
    weights = dict.fromkeys(keys,1/len(keys)) if weights is None else dict(weights)
    if set(weights) != set(keys):
        raise ValueError('Weights must specify every dimension')
    weights = {k: number(v) for k,v in weights.items()}
    if any(v is None or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Weights must be finite, nonnegative and have positive sum')
    threshold = bounded(coverage_threshold)
    if threshold is None: raise ValueError('Coverage threshold required')
    scores = {k:bounded(components.get(k)) for k in keys}
    available = sum(weights[k] for k in keys if scores[k] is not None)
    coverage = available/sum(weights.values())
    score = sum(weights[k]*scores[k] for k in keys if scores[k] is not None)/available if available else None
    return {name:bounded(score), name+'_coverage':coverage, name+'_status':
            'unavailable' if score is None else ('low_coverage' if coverage < threshold else 'valid'),
            'components':scores, 'weights':weights}

def calculate_tvc(components, weights=None, coverage_threshold=.8):
    return _weighted('tvc',tuple(TVC_FIELDS),components,weights,coverage_threshold)

def compute_rpdms(window, responsibility_ratio=None, require_responsibility=False):
    values = {k:bounded(window.get(k)) for k in ('NC','DAC','TTC_score','Comfort','EP')}
    if any(v is None for v in values.values()):
        return {'rpdms':None,'rnc':None,'rpdms_status':'unavailable','responsibility_coverage':0.0,
                'reason': MAP_REASON if values['DAC'] is None else 'required_pdms_component_not_available'}
    if values['NC'] not in (0,1): raise ValueError('Window NC must be binary')
    ratio = bounded(responsibility_ratio)
    if require_responsibility and values['NC'] == 0 and ratio is None:
        return {'rpdms':None,'rnc':None,'rpdms_status':'unavailable','responsibility_coverage':0.0,
                'reason':'required_responsibility_not_available'}
    rnc = 1.0 if values['NC']==1 else (1-ratio if ratio is not None else 0.0)
    base = values['DAC']*(5*values['TTC_score']+2*values['Comfort']+5*values['EP'])/12
    original = number(window.get('PDMS_open_loop'))
    if original is not None and not math.isclose(original,values['NC']*base,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Original PDMS differs from frozen formula')
    score = original if values['NC']==1 and original is not None else rnc*base
    return {'rpdms':bounded(score),'rnc':rnc,'rpdms_status':'valid',
            'ego_responsibility_ratio':ratio if values['NC']==0 else None,
            'responsibility_coverage':float(ratio is not None) if values['NC']==0 else None}

def _get(value,path):
    for key in path.split('.'):
        value = value.get(key,{}) if isinstance(value,dict) else {}
    return None if value == {} else value

def _valid_source(value):
    if not isinstance(value,dict): return {}
    status=str(value.get('status','')).lower()
    if status and status not in ('valid','valid_low_sample','completed','available','valid_low_coverage','ok','success'):
        return {}
    if value.get('skipped') or value.get('enabled') is False: return {}
    return value

def _parent_ok(value):
    status=str(value.get('status','')).lower() if isinstance(value,dict) else ''
    return not status.startswith(('failed','blocked','invalid','disabled','skipped','unavailable','not_available','dry_run'))

def _scene_id(value):
    return str(value).replace('__','/')

def _tvc(scene, cfg):
    backend = _get(scene,'backends.mtpc') or {}
    result = backend.get('result',{}) if _parent_ok(scene) and _parent_ok(backend) else {}
    primary = result.get('primary_metrics',{})
    if result.get('status') not in ('valid','valid_low_sample','completed'):
        primary = {}
    fields = cfg.get('fields',TVC_FIELDS)
    if fields != TVC_FIELDS: raise ValueError('TVC v1 source fields are frozen')
    if cfg.get('motion_transform','cosine_to_unit') != 'cosine_to_unit': raise ValueError('Frozen motion transform')
    if cfg.get('center_transform','one_minus_clip') != 'one_minus_clip' or cfg.get('center_clip',[0,1]) != [0,1]: raise ValueError('Frozen center transform')
    raw = {k:number(_get(primary,p)) for k,p in fields.items()}
    scores = dict(raw)
    if raw['center'] is not None:
        if raw['center'] < 0: raise ValueError('Negative normalized center error')
        scores['center'] = 1-min(raw['center'],1)
    if raw['motion'] is not None: scores['motion']=(bounded(raw['motion'],-1,1)+1)/2
    return {'scene_id':scene.get('scene'), **calculate_tvc(scores,cfg.get('weights'),cfg.get('coverage_threshold',.8)), 'raw_components':raw,'source_fields':fields}

def _population_trs(trajectory, cfg, output_dir=None, scene_paths=None, precomputed=None):
    """Measure an explicit fresh population, or verify sealed release evidence."""
    from driveaudit_eval.trajectory.trs import INPUT, VERSION, Q, ANCHOR, recompute, RESULT
    import json
    if cfg.get('protocol', VERSION) != VERSION:
        raise ValueError('TRS protocol must match the authoritative policy')
    if cfg.get('q', Q) != Q or cfg.get('anchor_excess_fraction', ANCHOR) != ANCHOR:
        raise ValueError('TRS q/anchor must match the authoritative policy')
    unavailable = {'TRS': None, 'status': 'NOT_COMPUTED',
                   'reason': 'exact_release_population_and_trajectory_provenance_required'}
    if cfg.get('fresh_population', False):
        from driveaudit_eval.trajectory.population import ROOT, evaluate_paired
        from driveaudit_eval.utils.io import file_sha256
        if trajectory.get('status') in ('disabled', 'skipped', 'dry_run'):
            return dict(unavailable, reason='trajectory_stage_not_run')
        paths = scene_paths or [row.get('input_gt_provenance', {}).get('scene_path')
                               for row in trajectory.get('scenes', [])]
        if not paths or not all(paths) or output_dir is None or not cfg.get('input_root') or not cfg.get('reference_root'):
            return dict(unavailable, reason='fresh_TRS_requires_selected_scenes_and_paired_roots')
        paths = [(ROOT / path).resolve() for path in paths]
        try:
            for row in trajectory.get('scenes', []):
                provenance = row.get('input_gt_provenance') or {}
                if not provenance:
                    continue
                scene = Path(provenance['scene_path']).resolve()
                if scene not in paths:
                    raise ValueError('Trajectory provenance is outside selected TRS population')
                for name, digest in provenance.get('files', {}).items():
                    if file_sha256(scene / name) != digest:
                        raise ValueError('Trajectory inputs changed after evaluation: ' + str(scene / name))
            if precomputed is not None:
                if precomputed.get('TRS') is None:
                    return dict(precomputed, TRS=None)
                from driveaudit_eval.metrics.testing_effectiveness import content_hash
                source = precomputed['source_manifest']
                selected = sorted((ROOT / cfg['input_root'] / row['relative']).resolve() for row in source.values())
                if selected != sorted(paths):
                    raise ValueError('Scheduled TRS population differs from selected trajectory scenes')
                for side, root_key, hash_key in (('candidate','input_root','trajectory_snapshot_hash'),
                                                  ('reference','reference_root','reference_snapshot_hash')):
                    if content_hash({sid:row[side] for sid,row in source.items()}) != precomputed[hash_key]:
                        raise ValueError('Scheduled TRS input manifest hash mismatch')
                    for row in source.values():
                        base = ROOT / cfg[root_key] / row['relative']
                        for name, digest in row[side].items():
                            if file_sha256(base / name) != digest:
                                raise ValueError('TRS input changed during evaluation: '+str(base/name))
                return precomputed
            return evaluate_paired(paths, ROOT / cfg['input_root'], ROOT / cfg['reference_root'],
                Path(output_dir) / 'trs_population',
                ROOT / cfg['protocol_path'] if cfg.get('protocol_path') else None,
                resume=cfg.get('resume', True), bootstrap=cfg.get('bootstrap', False),
                scene_manifest=ROOT / cfg['scene_manifest'] if cfg.get('scene_manifest') else None)
        except (OSError, ValueError, KeyError) as exc:
            return dict(unavailable, reason=f'fresh_TRS_input_error: {type(exc).__name__}: {exc}')
    rows = trajectory.get('scenes', []) if _parent_ok(trajectory) else []
    if len(rows) != 500 or not (RESULT / 'evidence/release_manifest.json').is_file():
        return unavailable
    release = json.loads((RESULT / 'evidence/release_manifest.json').read_text())
    seen = set()
    for row in rows:
        provenance = row.get('input_gt_provenance') or {}
        path = Path(provenance.get('scene_path', ''))
        try:
            relative = str(path.resolve().relative_to(INPUT.resolve()))
        except ValueError:
            return unavailable
        if relative in seen or relative not in release['scene_ids'] or not _parent_ok(row):
            return unavailable
        seen.add(relative)
        for name in ('future/future_trajectories.csv','timeline.json','history/agents.csv','generation_input.json'):
            if provenance.get('files', {}).get(name) != release['required_file_hashes'].get(relative+'/'+name):
                return unavailable
    if seen != set(release['scene_ids']):
        return unavailable
    result, intervals, _ = recompute()
    return dict(result, ci95=intervals, scope='population', independent_scene_TRS=False)

def build_core(summary, output_dir, config):
    """Aggregate live summary and original per-window CSVs; return JSON-ready result."""
    cfg = config.get('evaluation',{}).get('core_metrics',{})
    video=summary.get('video',{})
    if not _parent_ok(video): video={}
    vq=video.get('visual_quality',{})
    spans=_valid_source(_get(vq,'metrics.dino_temporal')).get('metrics',{})
    dino=[number(_get(spans,s+'.dino_temporal_'+s+'_mean')) for s in ('short','medium','long')]
    consistency=summary.get('consistency',{})
    consistency_scenes=consistency.get('scenes',[consistency] if 'backends' in consistency else [])
    tvc=[_tvc(s,cfg.get('tvc',{})) for s in consistency_scenes] if _parent_ok(consistency) else []
    trajectory=summary.get('trajectory',{})
    population_trs = _population_trs(trajectory, cfg.get('trs', {}), output_dir,
                                     config.get('evaluation', {}).get('trajectory', {}).get('scene_paths'),
                                     precomputed=summary.get('population_trs'))
    trs = []  # Population TRS is never fabricated as independent per-scene observations.
    model={'model':config.get('evaluation',{}).get('leaderboard',{}).get('model',cfg.get('model_name','GT reference (offline replay)')),
           'fvd':_valid_source(video.get('fvd')).get('fvd'), 'cmmd':_valid_source(_get(vq,'distribution.cmmd')).get('cmmd'),
           'clip_iqa':_valid_source(_get(vq,'metrics.clipiqa')).get('clipiqa_mean'),
           'dino_t':mean(dino) if all(v is not None for v in dino) else None,
           'epi_3px':_valid_source(video.get('geometry_consistency')).get('epipolar_inlier_ratio_3px'),
           'tvc':mean(s['tvc'] for s in tvc),'tvc_coverage':mean(s['tvc_coverage'] for s in tvc) or 0,
           'trs':population_trs['TRS'],'trs_coverage':1. if population_trs['TRS'] is not None else 0.}
    traffic=summary.get('traffic_and_responsibility',{})
    responsibility=traffic.get('responsibility_v3',{})
    events=responsibility.get('events',[]) if _parent_ok(traffic) and _parent_ok(responsibility) else []
    joined={}
    for event in events:
        if not event.get('agent_name') or event.get('decision_frame_id') is None: continue
        key=(_scene_id(event['scene_id']),event['agent_name'],int(event['decision_frame_id']))
        joined.setdefault(key,[]).append(event)
    windows=[]; scene_rows=[]; agent_rows=[]
    agent_source=summary.get('agent',{})
    agents=agent_source.get('agents',{}) if _parent_ok(agent_source) else {}
    include=config.get('agents',{}).get('include') or config.get('evaluation',{}).get('agents',{}).get('include') or list(FORMAL_AGENTS)
    for agent in include:
        if agent not in agents or not _parent_ok(agents[agent]):
            agent_rows.append({'agent':agent,'rpdms':None,'rpdms_mean':None,'status':'unavailable','reason':'agent_source_missing_or_failed'})
            continue
        path=Path(output_dir)/'real_agents'/agent/'agent_details'/agent/'per_frame_metrics.csv'
        if not path.exists():
            agent_rows.append({'agent':agent,'rpdms':None,'rpdms_mean':None,'status':'unavailable','reason':'per_window_source_missing'})
            continue
        with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
        expected=agents[agent].get('valid_windows')
        if expected is not None and len(rows)!=expected: raise ValueError('Per-window row count differs from agent summary')
        seen=set()
        for row in rows:
            key=(_scene_id(row['scene_id']),agent,int(row['frame_id']))
            if key in seen: raise ValueError('Duplicate planning window')
            seen.add(key)
            matches=joined.get(key,[])
            # Multiple collisions use maximum ego burden; any unavailable event keeps conservative penalty.
            ratios=[bounded(e.get('ego_responsibility_ratio')) if _parent_ok(e) and e.get('responsibility_status','valid')=='valid' else None for e in matches]
            ratio=max(ratios) if ratios and all(r is not None for r in ratios) else None
            r=compute_rpdms(row,ratio,require_responsibility=cfg.get('rpdms', {}).get('require_responsibility', False))
            windows.append({'agent':agent,'scene_id':key[0],'frame_id':key[2],**r,
                'responsibility_confidence':mean(e.get('responsibility_confidence') for e in matches),
                'responsibility_event_ids':[e['event_id'] for e in matches],
                **{k:number(row.get(k)) for k in ('PDMS_open_loop','NC','DAC','TTC_score','Comfort','EP')}})
            if cfg.get('protocol') == 'universal_v1':
                windows[-1].update({k:number(row.get(k)) for k in
                    ('dac_map_coverage','observable_samples','total_samples','drivable_samples')})
                w = windows[-1]
                if w['NC'] == 0:
                    w['ego_responsibility_ratio'] = ratio
                    w['responsibility_coverage'] = float(ratio is not None)
                if (row.get('dac_protocol') != 'observed_common_v1' or w['dac_map_coverage'] is None
                    or w['observable_samples'] is None or w['total_samples'] is None
                    or not 0 <= w['observable_samples'] <= w['total_samples'] or w['total_samples'] <= 0
                    or not math.isclose(w['dac_map_coverage'],w['observable_samples']/w['total_samples'],abs_tol=1e-12)):
                    raise ValueError('Universal RPDMS requires measured common-map support provenance')
                if w['dac_map_coverage'] < .8 and w['rpdms'] is not None:
                    raise ValueError('RPDMS with insufficient common-map coverage')
                counts = [w[k] for k in ('observable_samples','total_samples','drivable_samples')]
                if (any(v is None or v != int(v) for v in counts)
                    or not 0 <= w['drivable_samples'] <= w['observable_samples']
                    or (w['DAC'] is not None and (w['observable_samples'] == 0 or not math.isclose(
                        w['DAC'],w['drivable_samples']/w['observable_samples'],abs_tol=1e-12)))):
                    raise ValueError('Universal DAC differs from measured footprint counts')
        aw=[w for w in windows if w['agent']==agent]
        for sid in sorted({w['scene_id'] for w in aw}):
            sw=[w for w in aw if w['scene_id']==sid]
            score = mean(w['rpdms'] for w in sw)
            if agents[agent].get('scene_status') and any(w['rpdms'] is None for w in sw):
                score = None
            scene_rows.append({'agent':agent,'scene_id':sid,'rpdms_mean':score,'windows':len(sw)})
        episodes=[e for e in summary.get('episode_rollout',{}).get('episodes',[]) if e.get('agent')==agent or e.get('agent_name')==agent]
        collision_windows=[w for w in aw if w['NC']==0]
        score=mean(s['rpdms_mean'] for s in scene_rows if s['agent']==agent)
        agent_rows.append({'agent':agent,'rpdms':score,'rpdms_mean':score,
            'route_completion_ratio':mean(e.get('route_completion_ratio') for e in episodes),
            'windows':len(aw),'collision_windows':len(collision_windows),
            'responsibility_coverage':mean(w['responsibility_coverage'] for w in collision_windows),
            'responsibility_confidence':mean(w['responsibility_confidence'] for w in collision_windows)})
    if cfg.get('protocol') == 'universal_v1':
        # Identical decision support for all agents, never model-specific deletion.
        configured_frames = config['evaluation']['agent'].get('score_frame_ids')
        if not configured_frames:
            raise ValueError('Universal core requires explicit frozen decision frames')
        for sid in sorted({w['scene_id'] for w in windows}):
            frame_ids = configured_frames.get(sid, configured_frames.get(sid.replace('/','__'))) if isinstance(configured_frames, dict) else configured_frames
            if not frame_ids:
                raise ValueError('Missing frozen decision frames for '+sid)
            planned = set(frame_ids)
            support = [set(w['frame_id'] for w in windows if w['scene_id']==sid
                       and w['agent']==agent and w['rpdms'] is not None and w['frame_id'] in planned) for agent in include]
            common = set.intersection(*support) if support else set()
            coverage = len(common)/len(planned) if planned else 0.
            for row in scene_rows:
                if row['scene_id'] != sid: continue
                row.update(common_window_count=len(common), common_window_coverage=coverage)
                row['rpdms_mean'] = mean(w['rpdms'] for w in windows if w['scene_id']==sid
                    and w['agent']==row['agent'] and w['frame_id'] in common) if coverage>=.8 else None
        for row in agent_rows:
            selected = [r for r in scene_rows if r['agent']==row['agent']]
            row['rpdms'] = row['rpdms_mean'] = mean(r['rpdms_mean'] for r in selected)
            row['common_window_count'] = sum(r['common_window_count'] for r in selected)
            row['common_window_coverage'] = mean(r['common_window_coverage'] for r in selected)
    agent_coverage=sum(a['rpdms'] is not None for a in agent_rows)/len(include) if include else 0.0
    route_coverage=sum(a.get('route_completion_ratio') is not None for a in agent_rows)/len(include) if include else 0.0
    model['rpdms']=mean(a['rpdms'] for a in agent_rows) if agent_coverage==1 else None
    cross_dataset = route_cross_dataset = None
    if any(agent.get('scene_status') for agent in agents.values()):
        from driveaudit_eval.metrics.leaderboard import build_cross_dataset_leaderboards
        cross_source = {'agents': {agent: {
            'scene_status': agents.get(agent, {}).get('scene_status', []),
            'per_scene': [{'scene_id': row['scene_id'], 'rpdms': row['rpdms_mean']}
                          for row in scene_rows if row['agent'] == agent]}
            for agent in include}}
        cross_dataset = build_cross_dataset_leaderboards(
            cross_source, Path(output_dir) / 'rpdms_cross_dataset', expected_agents=include, score='rpdms')
        common_scores = {row['agent']: row['rpdms'] for row in cross_dataset['full_agent']['rows']}
        for row in agent_rows:
            row['rpdms'] = row['rpdms_mean'] = common_scores[row['agent']]
            row['rpdms_datasets'] = cross_dataset['full_agent']['datasets']
        model['rpdms'] = mean(row['rpdms'] for row in cross_dataset['full_agent']['rows'])
        rollout = summary.get('episode_rollout', {})
        route_source = {'agents': {agent: {
            'scene_status': agents.get(agent, {}).get('scene_status', []), 'per_scene': []} for agent in include}}
        for agent in include:
            episodes = [episode for episode in rollout.get('episodes', [])
                        if (episode.get('agent') or episode.get('agent_name')) == agent and episode.get('scene_id')]
            for sid in sorted({_scene_id(episode['scene_id']) for episode in episodes}):
                selected = [episode for episode in episodes if _scene_id(episode['scene_id']) == sid]
                values = [number(episode.get('route_completion_ratio'))
                          if _parent_ok(rollout) and _parent_ok(episode) else None for episode in selected]
                route_source['agents'][agent]['per_scene'].append({
                    'scene_id': sid, 'route_completion_ratio': mean(values) if all(value is not None for value in values) else None})
        route_cross_dataset = build_cross_dataset_leaderboards(
            route_source, Path(output_dir) / 'route_cross_dataset', expected_agents=include, score='route_completion_ratio')
        common_routes = {row['agent']: row['route_completion_ratio'] for row in route_cross_dataset['full_agent']['rows']}
        for row in agent_rows:
            row['route_completion_ratio'] = common_routes[row['agent']]
            row['route_datasets'] = route_cross_dataset['full_agent']['datasets']
        agent_coverage = sum(row['rpdms'] is not None for row in agent_rows)/len(include) if include else 0.
        route_coverage = sum(row['route_completion_ratio'] is not None for row in agent_rows)/len(include) if include else 0.
    model['route_completion_ratio']=mean(a.get('route_completion_ratio') for a in agent_rows) if route_coverage==1 else None
    if not _parent_ok(summary.get('episode_rollout',{})): model['route_completion_ratio']=None
    statuses={}
    for key in CORE_IDS:
        value=number(model.get(key)); lower,upper=METRICS[key]['range']
        if value is not None and ((lower is not None and value<lower) or (upper is not None and value>upper)):
            raise ValueError('Core range audit failed: '+key)
        model[key]=value
        statuses[key]='unavailable' if value is None else 'valid'
        if key in ('tvc','trs') and value is not None and model[key+'_coverage']<cfg.get(key,{}).get('coverage_threshold',.8): statuses[key]='low_coverage'
    return {'status':'completed' if all(s=='valid' for s in statuses.values()) else 'partial',
            'version':'universal_v1' if cfg.get('protocol')=='universal_v1' else '1.0',
            **({'dac_protocol':'observed_common_v1','common_support':'frozen decision intersection across all six agents; minimum coverage 0.8'}
               if cfg.get('protocol')=='universal_v1' else {}),
            'agent_coverage':agent_coverage,'route_agent_coverage':route_coverage,'expected_agents':list(include),
            'formal_agent_set':set(include)==set(FORMAL_AGENTS),'model_row':model,'metric_statuses':statuses,
            'tvc':tvc,'trs':trs,'trs_population':population_trs,'rpdms_windows':windows,'rpdms_scenes':scene_rows,'rpdms_agents':agent_rows,
            'scope':'One evaluated video/trajectory submission; six downstream agents are not six generation models.',
            'rpdms_multi_collision_policy':'maximum ego responsibility across window events; ' +
                ('missing collision responsibility is unavailable' if cfg.get('rpdms', {}).get('require_responsibility', False)
                 else 'any unavailable ratio gives RNC=0'),
            'rpdms_cross_dataset':cross_dataset,
            'route_cross_dataset':route_cross_dataset,
            'aggregation': ('TVC scene equal; TRS frozen population; RPDMS and route scene equal then common-dataset equal then agent equal; explicit metric-specific common subsets'
                            if cross_dataset is not None else
                            'TVC scene equal; TRS frozen population; RPDMS window mean then scene equal then agent equal; route episode then agent equal')}

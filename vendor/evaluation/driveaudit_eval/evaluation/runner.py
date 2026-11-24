"""Dispatch four independent evaluators; no model imports at startup."""
import importlib
import logging
from datetime import datetime
from pathlib import Path

from driveaudit_eval.utils.io import write_json

SECTIONS = ('video', 'trajectory', 'consistency', 'agent')


def evaluate_stage(name, config, context, summary):
    """One unchanged evaluator call, shared by sequential and process workers."""
    settings = config['evaluation']
    logger = context['logger']
    dry_run = context.get('dry_run', False)
    logger.info('[%s] %s', name.upper(), 'Checking inputs and dependencies' if dry_run else 'Running')
    try:
        if name in SECTIONS:
            cfg = settings.get(name, {'enabled': False})
            reuse = settings.get('reuse_results', {}).get(name)
            if reuse and not dry_run:
                import json
                from driveaudit_eval.utils.io import file_sha256
                if name == 'agent':
                    raise ValueError('Reuse agent predictions through the verified adapter; never reuse agent scores')
                root = Path(context['root'])
                source = root / reuse['path']
                if file_sha256(source) != reuse['sha256']:
                    raise ValueError('Reused result hash mismatch: '+str(source))
                if not reuse['input_sha256']:
                    raise ValueError('Reused results require a nonempty input hash manifest')
                for relative, expected in reuse['input_sha256'].items():
                    if file_sha256(root / relative) != expected:
                        raise ValueError('Reused input hash mismatch: '+relative)
                result = json.loads(source.read_text())
            else:
                evaluate = importlib.import_module('driveaudit_eval.' + ('agents' if name == 'agent' else name) + '.evaluator').evaluate
                result = evaluate(cfg, context)
        elif name == 'traffic_and_responsibility':
            from driveaudit_eval.traffic_evaluator import evaluate
            result = evaluate(config, context, summary)
        elif name == 'episode_rollout':
            from driveaudit_eval.episode_rollout import evaluate
            result = evaluate(config, context, summary)
            if settings.get('profile') == 'research' and config.get('episode_rollout', {}).get('route_projection', {}).get('enabled'):
                from driveaudit_eval.research.route_sensitivity import attach_route_sensitivity
                attach_route_sensitivity(result, config, Path(context['root']))
        elif name == 'population_trs':
            from driveaudit_eval.metrics.core import _population_trs
            result = _population_trs(summary.get('trajectory', {}), settings['core_metrics']['trs'],
                context['output_dir'], settings.get('trajectory', {}).get('scene_paths'))
        else:
            raise ValueError('Unknown evaluation stage: '+name)
        if not isinstance(result, dict):
            raise TypeError('Evaluator must return a dict')
        result['status'] = str(result.get('status', 'failed')).lower()
        if dry_run and result['status'] == 'ready':
            result['status'] = 'dry_run'
    except (ImportError, FileNotFoundError) as exc:
        result = dict(status='skipped' if name in SECTIONS else 'failed', reason=f'{type(exc).__name__}: {exc}')
        logger.warning('[%s] %s', name.upper(), result['reason'])
    except Exception as exc:
        result = dict(status='failed', reason=f'{type(exc).__name__}: {exc}')
        logger.exception('[%s] FAILED', name.upper())
    logger.info('[%s] %s %s', name.upper(), result['status'], result.get('reason', ''))
    return result


def run(config, root, section='all', dry_run=False, *, export_outputs=True):
    if section not in (*SECTIONS, 'all'):
        raise ValueError('Unknown evaluation section: ' + section)
    from driveaudit_eval.evaluation.profiles import prepare_config, CONTROL_KEYS
    config=prepare_config(config)
    if config.get('agents',{}).get('include') is not None:
        included=config['agents']['include']
        if not isinstance(included,list) or not included or len(set(included))!=len(included):
            raise ValueError('agents.include must be a nonempty unique list')
        config['evaluation']['agent']['agents']=included
    settings = config.get('evaluation')
    if not isinstance(settings, dict):
        raise ValueError('Config must contain evaluation mapping with video/trajectory/consistency/agent')
    unknown = set(settings) - set(SECTIONS) - CONTROL_KEYS
    if unknown:
        raise ValueError('Unknown evaluation sections: ' + ', '.join(sorted(unknown)))
    root = Path(root).resolve()
    output = config.get('output', {})
    run_name = output.get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('output.run_name must be a single directory name')
    if dry_run:
        run_name += '_dry_run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = root / output.get('root', 'outputs/evaluation') / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / 'evaluation.log', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    context = dict(root=root, output_dir=output_dir, dry_run=dry_run, logger=logger)
    summary = {}
    try:
        testing = settings.get('testing_effectiveness', {}).get('enabled', False)
        if testing:
            from driveaudit_eval.testing import reference_paths
            settings['video']['reference_scene_paths'] = reference_paths(config,root)
        import yaml
        (output_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        if config.get('execution', {}).get('enabled', False) and dry_run:
            from driveaudit_eval.evaluation.parallel import run_stages
            run_stages(config, context, section=section, dry_run=True)
        if config.get('execution', {}).get('enabled', False) and not dry_run:
            from driveaudit_eval.evaluation.parallel import run_stages
            summary = run_stages(config, context, section=section)
        else:
            for name in SECTIONS:
                cfg = settings.get(name, {'enabled': False})
                if not isinstance(cfg, dict):
                    raise ValueError(name + ' settings must be a mapping')
                if not cfg.get('enabled', True) or section not in ('all', name):
                    summary[name] = {'status': 'disabled', 'reason': 'disabled by config or section selection'}
                    continue
                summary[name] = evaluate_stage(name, config, context, summary)
                write_json(output_dir / (name + '.json'), summary[name])
            for name, enabled in (
                    ('traffic_and_responsibility', section in ('all', 'trajectory', 'agent') and config.get('traffic_compliance', {}).get('enabled', False)),
                    ('episode_rollout', section in ('all', 'agent') and config.get('episode_rollout', {}).get('enabled', False))):
                if enabled and not dry_run:
                    summary[name] = evaluate_stage(name, config, context, summary)
                    write_json(output_dir / (name + '.json'), summary[name])
        if not dry_run and settings.get('core_metrics', {}).get('enabled', settings.get('profile', 'legacy') != 'legacy'):
            from driveaudit_eval.metrics.core import build_core
            summary['core_metrics'] = build_core(summary, output_dir, config)
        if testing and not dry_run:
            from driveaudit_eval.testing import attach_testing
            attach_testing(summary, output_dir, config, context)
        write_json(output_dir / 'summary.json', summary)
        if not dry_run and export_outputs:
            from driveaudit_eval.research.exports import export_results
            export_results(summary, output_dir, config=config)
            if config.get('paper_export', {}).get('enabled', False):
                from tools.export_paper_metrics import export_single_run
                paper = export_single_run(output_dir, config)
                logger.info('[PAPER] %d numeric cells; %d eligible cells', paper['numeric_cells'], paper['paper_ready_cells'])
        logger.info('Evaluation summary: %s', output_dir / 'summary.json')
        return summary, output_dir
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def recover_agent_scoring(config, context, agent_result):
    """Rescore verified same-run predictions on shared GT support; return summaries.

    Only the existing scorer writes per-model agent_details. The caller owns
    publishing shared results/comparison/scene_status after inference stops.
    """
    import json
    from copy import deepcopy
    import numpy as np
    from driveaudit_eval.agents.input_gt import load_scene, sample, prediction_windows, evaluate_predictions
    from driveaudit_eval.agents.real_agents import _same_run_predictions
    from driveaudit_eval.metrics.testing_effectiveness import content_hash
    output = Path(context['output_dir'])/'real_agents'
    manifest = json.loads((output/'input_manifest.json').read_text())
    windows = manifest['common_windows']
    if prediction_windows(config, context) != windows:
        raise ValueError('Scoring recovery requested windows differ from original inference')
    dt, horizon = float(config.get('dt', .5)), float(config.get('horizon_s', 3.))
    max_gap = config.get('max_interpolation_gap_s', .25)
    offsets = np.arange(1, round(horizon/dt)+1)*dt
    requested, supported, unsupported = {}, {}, []
    for scene in manifest['scenes']:
        sid = scene['scene_id']
        requested[sid], supported[sid] = [], []
        _, times, ego, _, _ = load_scene(Path(scene['scene_path']))
        for window in (w for w in windows if w['scene_id'] == sid):
            requested[sid].append(window['frame_id'])
            query = np.r_[window['start_time_s'], window['start_time_s']+offsets]
            _, valid = sample(times, ego, query, max_gap)
            if valid.all():
                supported[sid].append(window['frame_id'])
            else:
                unsupported.append(dict(window, unsupported_query_times_s=query[~valid].tolist(),
                    reason=f'native_ego_interpolation_gap_exceeds_frozen_{max_gap:g}s'))
    support = dict(requested_windows=len(windows), supported_windows=sum(map(len, supported.values())),
        requested_frame_ids=requested, supported_frame_ids=supported, unsupported=unsupported,
        threshold_s=max_gap, dt_s=dt, horizon_s=horizon,
        inference_windows_unchanged=True, computed_from_GT_only=True)
    support_hash = content_hash(support)
    result = deepcopy(agent_result)
    result['scoring_time_support'] = support
    required = {(w['scene_id'], w['frame_id']) for w in windows}
    for name, previous in list(result['agents'].items()):
        try:
            records = _same_run_predictions(output, name, manifest)
            if records is None:
                raise ValueError('Scoring recovery requires completed same-run inference')
            inferred = {(r['scene_id'], r['frame_id']) for r in records if r.get('inference_success') is True}
            if not required.issubset(inferred):
                raise ValueError('Scoring recovery requires predictions for all original windows')
            receipt = json.loads((output/name/'inference_status.json').read_text())
            scored = evaluate_predictions(dict(config, agent=name, score_frame_ids=supported),
                dict(context, output_dir=output/name), records) if support['supported_windows'] else dict(
                    status='PARTIAL', valid_windows=0, per_scene=[], reason='No supported GT scoring windows')
            if scored.get('valid_windows') != support['supported_windows']:
                raise ValueError('Recovered scoring did not cover the complete shared GT support')
            scored.update(inference=receipt, requested_windows=len(windows), inferred_windows=len(required & inferred),
                scored_windows=scored['valid_windows'], unsupported_windows=len(unsupported),
                scoring_support_hash=support_hash, same_run_resumed=True, real_prediction_reused=False,
                model_inference_executed=False,
                original_scoring_error=previous.get('original_scoring_error', previous.get('reason')))
            if unsupported:
                scored['status'] = 'PARTIAL'
                scored['reason'] = '; '.join(filter(None, (scored.get('reason'),
                    f'incomplete_ego_time_support: {len(unsupported)} requested windows remain unscored')))
            by_scene = {r['scene_id']:r for r in scored['per_scene']}
            states = deepcopy(previous.get('scene_status', [s for s in manifest.get('scene_status', []) if s['agent']==name]))
            for sid, frames in requested.items():
                counts = dict(requested_windows=len(frames), inferred_windows=sum((sid,f) in inferred for f in frames),
                              scored_windows=len(supported[sid]), unsupported_windows=len(frames)-len(supported[sid]))
                if sid in by_scene:
                    by_scene[sid].update(counts)
                for state in (s for s in states if s['scene']==sid):
                    missing_score = by_scene.get(sid, {}).get('PDMS_open_loop') is None
                    state.update(counts, inference_status='completed',
                        evaluation_status='partial' if counts['unsupported_windows'] or missing_score else 'completed',
                        original_scoring_error=state.get('original_scoring_error', state.get('failure_reason')),
                        failure_reason='incomplete_ego_time_support' if counts['unsupported_windows'] else
                            'insufficient_drivable_area_map_support' if not state.get('available_modalities', {}).get('drivable_area', True) else None)
            scored['scene_status'] = states
            result['agents'][name] = scored
        except Exception as exc:
            previous['scoring_recovery_error'] = f'{type(exc).__name__}: {exc}'
    result['comparison'] = []
    for name, model in result['agents'].items():
        row = dict(agent=name, status=model['status'], common_windows=model.get('valid_windows'), reason=model.get('reason'))
        for key in ('NC','DAC','TTC_score','Comfort','EP','EP_current_speed_proxy','PDMS_open_loop',
                    'RC_open_loop','Driving_Score_open_loop','ADE_3s','FDE_3s'):
            values = [s.get(key) for s in model.get('per_scene', [])]
            row[key] = float(np.mean(values)) if values and all(v is not None for v in values) else None
        result['comparison'].append(row)
    models = result['agents'].values()
    result['status'] = ('COMPLETED' if models and all(r['status'].upper()=='COMPLETED' for r in models) else
                        'PARTIAL' if any(r.get('per_scene') for r in models) else 'FAILED')
    result['same_run_resumed'] = any(r.get('same_run_resumed', False) for r in models)
    result['model_inference_executed'] = any(r.get('model_inference_executed', False) for r in models)
    return result


def run_formal50(config, root, dry_run=False):
    """One frozen population, asymmetric branches, using the existing evaluators."""
    import csv
    import fcntl
    import hashlib
    import json
    import os
    import time
    from copy import deepcopy
    from driveaudit_eval.utils.io import file_sha256, write_csv
    from driveaudit_eval.metrics.testing_effectiveness import content_hash
    from driveaudit_eval.evaluation.profiles import prepare_config
    root = Path(root).resolve()
    cfg = prepare_config(deepcopy(config))
    options = cfg['formal50']
    output = root / cfg['output']['root'] / cfg['output']['run_name']
    if output != root/'outputs/formal50' or output.is_symlink():
        raise ValueError('formal50 requires the explicit local outputs/formal50 directory')
    manifest_path = output/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selection = root/options['selected_scene_manifest']
    if file_sha256(selection) != manifest['selection']['selected_manifest_sha256']:
        raise ValueError('Frozen scene manifest changed')
    rows = list(csv.DictReader(selection.open()))
    from collections import Counter
    if dict(Counter(row['dataset'] for row in rows)) != options['counts'] or len({r['scene_id'] for r in rows}) != 50:
        raise ValueError('formal50 requires exactly the preregistered 50 paired scenes')
    if options['catk_tasks'] != ['trajectory'] or not options['fresh_gt_inference']:
        raise ValueError('formal50 branch/inference contract changed')
    if cfg['evaluation']['agent'].get('reuse_predictions_from'):
        raise ValueError('Historical Agent predictions are forbidden')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(options['resources']['gpu']),
                      OMP_NUM_THREADS=str(options['resources']['cpu_threads']),
                      OPENBLAS_NUM_THREADS=str(options['resources']['cpu_threads']))
    paths = [str(root/options['gt_root']/r['relative']) for r in rows]
    for section in SECTIONS:
        cfg['evaluation'][section]['scene_paths'] = paths
    cfg['evaluation']['video']['reference_scene_paths'] = paths
    from driveaudit_eval.agents.input_gt import prediction_windows
    cfg['evaluation']['agent']['score_frame_ids'] = None
    windows = prediction_windows(cfg['evaluation']['agent'], {'root':root})
    frozen_windows = {r['scene_id']: [w['frame_id'] for w in windows if w['scene_id']==r['scene_id']] for r in rows}
    if any(not f for f in frozen_windows.values()):
        raise ValueError('Selected scene has no legal native-time windows; replacement forbidden')
    cfg['evaluation']['agent']['score_frame_ids'] = frozen_windows
    protocol = json.loads((root/options['trs_protocol']).read_text())
    from driveaudit_eval.metrics.registry import TRS_POLICY
    if protocol['q'] != TRS_POLICY['q'] or protocol['anchor_excess_fraction'] != TRS_POLICY['anchor_excess_fraction']:
        raise ValueError('Migrated protocol differs from authoritative policy')
    if dry_run:
        print('formal50: 50 paired scenes;',len(windows),'native-time windows per agent; q=',protocol['q'])
        return {'status':'DRY_RUN'}
    lock = (output/'evidence/run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_identity = _formal50_source_identity(rows, root, options)
    code_identity = {str(p.relative_to(root)):file_sha256(p) for p in sorted((root/'driveaudit_eval').rglob('*.py')) if 'research' not in p.parts}
    binding = dict(inputs=content_hash(source_identity), code=content_hash(code_identity))
    if manifest.get('execution_binding') and manifest['execution_binding'] != binding:
        lock.close()
        raise ValueError('Same-run source or implementation identity changed; prior results retained')
    manifest['execution_binding'] = binding
    write_json(output/'evidence/selected_input_hashes.json', source_identity)
    logger = logging.getLogger('formal50'); logger.setLevel(logging.INFO); logger.propagate=False
    handlers = [logging.StreamHandler(), logging.FileHandler(output/'logs/evaluation.log',encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); logger.addHandler(handler)
    context = dict(root=root, output_dir=output/'gt', dry_run=False, logger=logger)
    write_json(output/'evidence/resolved_config.json',cfg)
    write_csv(output/'evidence/decision_windows.csv',windows)
    manifest.update(status='RUNNING', requested_windows_per_agent=len(windows), requested_scene_agent_pairs=300,
                    command=f'{os.sys.executable} tools/evaluate.py --config tools/configs/formal50.yaml',
                    protocol_sha256=file_sha256(root/options['trs_protocol']),
                    implementation_sha256=code_identity)
    write_json(manifest_path,manifest)

    def stage(name, settings, call):
        destination=output/'evidence'/(name+'.json')
        identity=content_hash({'settings':settings,'selection':manifest['selection']['selected_manifest_sha256'],
                               'run_id':manifest['run_id']})
        previous=manifest['stages'].get(name,{})
        if previous.get('identity')==identity and previous.get('execution_status')=='COMPLETED':
            if not destination.is_file() or file_sha256(destination)!=previous['result_sha256']:
                raise ValueError('Same-run result changed: '+name)
            for rel,digest in previous.get('artifacts',{}).items():
                if not (output/rel).is_file() or file_sha256(output/rel)!=digest:
                    raise ValueError('Same-run artifact changed: '+rel)
            logger.info('[RESUME] %s: same-run hash-verified result',name)
            previous['reuse']='same_run_resumed';write_json(manifest_path,manifest)
            return json.loads(destination.read_text())
        logger.info('[STAGE] %s START',name);started=time.monotonic()
        before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in output.rglob('*')
                if p.is_file() and not p.is_symlink()}
        manifest['stages'][name]={'identity':identity,'execution_status':'RUNNING','reuse':'freshly_computed'}
        write_json(manifest_path,manifest)
        try:
            result=call()
            execution='FAILED' if str(result.get('status','')).lower() in ('failed','blocked','error') else 'COMPLETED'
        except Exception as exc:
            logger.exception('[STAGE] %s failed',name)
            result={'status':'FAILED','reason':f'{type(exc).__name__}: {exc}'};execution='FAILED'
        write_json(destination,result)
        import gc
        gc.collect()
        if 'torch' in os.sys.modules:
            os.sys.modules['torch'].cuda.empty_cache()
        # Bind rewritten retry artifacts as well as new files. Agent recovery
        # also consumes immutable predictions created earlier in this same run.
        artifacts={str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*')
                   if p.is_file() and not p.is_symlink()
                   and (before.get(p)!=(p.stat().st_size,p.stat().st_mtime_ns)
                        or (name=='gt_agents' and output/'gt/real_agents' in p.parents))
                   and p not in (destination,manifest_path) and not {'logs','pytest_cache'} & set(p.relative_to(output).parts)
                   and p.suffix not in ('.log',)}
        manifest['stages'][name]={'identity':identity,'execution_status':execution,'metric_status':result.get('status'),
            'reuse':'freshly_computed','elapsed_s':time.monotonic()-started,'result_sha256':file_sha256(destination),'artifacts':artifacts}
        write_json(manifest_path,manifest)
        logger.info('[STAGE] %s %s %.1fs',name,execution,time.monotonic()-started)
        return result

    try:
        def validate():
            from driveaudit_eval.data.scene_input import load_scene_input
            from driveaudit_eval.trajectory.catk.export import verify_output
            result=[]
            for row,path in zip(rows,paths):
                scene=load_scene_input(path,agent_only=True)
                verify_output(root/options['catk_root']/row['relative'])
                result.append(dict(scene_id=row['scene_id'],dataset=scene.dataset_name,capabilities=scene.capabilities,
                                   requested_windows=len(frozen_windows[row['scene_id']])))
            return dict(status='COMPLETED',scenes=result,paired_scene_ids_identical=True)
        checked=stage('input_validation',{'gt_root':options['gt_root'],'catk_root':options['catk_root']},validate)
        if checked['status']!='COMPLETED':
            raise ValueError('Frozen input validation failed; scenes cannot be replaced')
        from driveaudit_eval.video.utils import audit_inputs
        decoded=stage('decoded_inputs',{'paths':paths},lambda:audit_inputs(paths,output/'evidence/decoded_inputs.csv'))
        if decoded.get('status') not in ('PASS','COMPLETED','completed'):
            raise ValueError('Current decoder self-consistency failed')
        def worker_decode():
            import subprocess
            worker=cfg['evaluation']['video']['visual_quality']['dino_temporal']['python']
            script=("import csv,json,sys; from pathlib import Path; from driveaudit_eval.video.utils import audit_inputs; "
                    "rows=list(csv.DictReader(open(sys.argv[1]))); "
                    "paths=[Path(sys.argv[2])/r['relative'] for r in rows]; "
                    "print(json.dumps(audit_inputs(paths,Path(sys.argv[3]))))")
            result=json.loads(subprocess.check_output([worker,'-c',script,str(selection),
                str(root/options['gt_root']),str(output/'evidence/decoded_inputs_dino.csv')],cwd=root,text=True))
            if result['decoded_pixels_sha256']!=decoded['decoded_pixels_sha256']:
                raise ValueError('Current worker/main decoded pixels differ; inspect both per-image manifests')
            return result
        worker=stage('decoded_inputs_dino',{'python':cfg['evaluation']['video']['visual_quality']['dino_temporal']['python']},worker_decode)
        if worker.get('status')!='PASS':raise ValueError('Worker decoder verification failed')
        summary={}
        for name in ('video','consistency','trajectory'):
            settings=cfg['evaluation'][name]
            evaluator=importlib.import_module('driveaudit_eval.'+name+'.evaluator').evaluate
            summary[name]=stage('gt_'+name,settings,lambda e=evaluator,s=settings:e(s,context))
        from driveaudit_eval.trajectory.population import prepare_gt, run as population
        gt_population=stage('gt_population',protocol,lambda:prepare_gt(rows,output,protocol,root=root,resume=True))
        from driveaudit_eval.agents.real_agents import evaluate as agents
        def gt_agents():
            result=agents(cfg['evaluation']['agent'],context)
            result=recover_agent_scoring(cfg['evaluation']['agent'],context,result)
            destination=output/'gt/real_agents'
            states=[state for model in result['agents'].values() for state in model.get('scene_status',[])]
            write_json(destination/'results.json',result['agents'])
            write_csv(destination/'comparison.csv',result['comparison'])
            write_json(destination/'scene_status.json',states)
            write_json(destination/'completion.json',dict(status=result['status'],agents=list(result['agents']),
                common_windows=len(windows),scoring_supported_windows=result['scoring_time_support']['supported_windows']))
            from driveaudit_eval.agents.registry import write_support_matrix
            from driveaudit_eval.metrics.leaderboard import build_cross_dataset_leaderboards
            write_support_matrix(states,destination)
            build_cross_dataset_leaderboards(result['agents'],destination,expected_agents=cfg['evaluation']['agent']['agents'])
            return result
        summary['agent']=stage('gt_agents',cfg['evaluation']['agent'],gt_agents)
        from driveaudit_eval.traffic_evaluator import evaluate as traffic
        summary['traffic_and_responsibility']=stage('gt_responsibility',{'traffic':cfg['traffic_compliance'],
            'responsibility':cfg['responsibility_v2'],'attribution':cfg['collision_responsibility_v3']},lambda:traffic(cfg,context,summary))
        from driveaudit_eval.episode_rollout import evaluate as episodes
        summary['episode_rollout']=stage('gt_episodes',cfg['episode_rollout'],lambda:episodes(cfg,context,summary))
        from driveaudit_eval.metrics.core import build_core
        summary['core_metrics']=stage('gt_core',cfg['evaluation']['core_metrics'],lambda:build_core(summary,output/'gt',cfg))
        trajectory=stage('catk_population',protocol,lambda:population(rows,output,protocol,root=root,resume=True))
        self_check=stage('gt_self_check',cfg['evaluation']['testing_effectiveness'],lambda:_formal50_self_check(summary,output,cfg,context))
        result=_formal50_report(summary,gt_population,trajectory,self_check,rows,output,manifest)
        manifest['status']=result['status'];write_json(manifest_path,manifest)
        print((output/'summary.md').read_text())
        return result
    finally:
        for handler in handlers:logger.removeHandler(handler);handler.close()
        lock.close()


def _formal50_source_identity(rows, root, options):
    """Bind bytes and link targets, including resolved RGB; no old result caches."""
    import os
    from driveaudit_eval.utils.io import file_sha256
    records={}; content={}
    for branch in ('gt_root','catk_root'):
        for row in rows:
            scene=root/options[branch]/row['relative']
            for base,dirs,files in os.walk(scene,followlinks=False):
                for name in sorted(dirs+files):
                    path=Path(base)/name;relative=str(path.relative_to(root))
                    if path.is_symlink():
                        records[relative]={'symlink':os.readlink(path),'resolved':str(path.resolve())}
                    if path.is_file():
                        target=path.resolve()
                        if target not in content:content[target]=file_sha256(path)
                        records.setdefault(relative,{})['sha256']=content[target]
    return records


def _formal50_self_check(summary, output, config, context):
    """Pair the same measured Agent records; never execute a second GT inference."""
    import json
    from driveaudit_eval.testing import scene_evidence, record_identity, measured_record, scoring_hash
    from driveaudit_eval.agents.provenance import model_identity
    from driveaudit_eval.agents.registry import FORMAL_AGENTS
    from driveaudit_eval.metrics.testing_effectiveness import paired_comparison
    directory=output/'gt/real_agents'
    manifest=json.loads((directory/'input_manifest.json').read_text())
    records=[];expected=[];scorer=scoring_hash(config)
    identities={a:model_identity(a) for a in FORMAL_AGENTS}
    for scene in manifest['scenes']:
        evidence=scene_evidence(scene)
        expected.append((scene['dataset'],scene['scene_id'].replace('__','/')))
        for agent in FORMAL_AGENTS:
            identity=record_identity(agent,scene,manifest['common_windows'],evidence,evidence,identities[agent],scorer)
            try:
                record=measured_record(identity,summary,directory,scene,identities[agent])
            except (OSError, ValueError, KeyError) as exc:
                record=dict(identity,rpdms=None,measurement_error=f'{type(exc).__name__}: {exc}')
            records.append(record)
    comparison=paired_comparison(records,records,expected)
    # Formal quality retention requires per-scene TRS, which the frozen population
    # metric deliberately does not invent. The unapproved population API stays off.
    return dict(status='PARTIAL',role='reference_self_check',comparison=comparison,
                TE=comparison['statistics']['mean'],Q_rel=None,QTES=None,
                reason='formal_gt_relative_v1_requires_per_scene_TRS; current_TRS_is_population_only',
                second_gt_inference_executed=False)


def _formal50_report(summary, gt_population, trajectory, self_check, rows, output, manifest):
    """Report measured child results and support without filling missing scores."""
    import csv
    import json
    import math
    from collections import Counter
    from driveaudit_eval.agents.registry import AGENTS
    from driveaudit_eval.metrics.core import _tvc, _valid_source
    from driveaudit_eval.utils.io import write_csv
    output = Path(output)
    (output/'gt').mkdir(parents=True, exist_ok=True)

    def read(path, default):
        return json.loads(path.read_text()) if path.is_file() else default

    def number(value):
        try:
            return float(value) if not isinstance(value, bool) and math.isfinite(float(value)) else None
        except (TypeError, ValueError):
            return None

    def mean(values):
        values = [number(value) for value in values]
        values = [value for value in values if value is not None]
        return sum(values)/len(values) if values else None

    selected = {row['scene_id']: row for row in rows}
    def sid(value):
        value = str(value)
        return next((key for key, row in selected.items() if value in (key, key.replace('__', '/'), row['relative'])
                     or value.endswith('/'+row['relative'])), value)

    def recordings(ids):
        return len({(selected[key]['dataset'], selected[key]['cluster_id']) for key in ids if key in selected})

    def show(value):
        if value is None:
            return 'UNAVAILABLE'
        if isinstance(value, float):
            return f'{value:.8g}'
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

    def table(headers, data):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
            ['| '+' | '.join(show(row.get(key)).replace('|', '/').replace('\n', ' ') for key in headers)+' |' for row in data])

    core = summary.get('core_metrics', {})
    video = summary.get('video', {})
    vq = video.get('visual_quality', {})
    frames = vq.get('frame_metrics', [])
    dino = vq.get('metrics', {}).get('dino_temporal', {})
    pairs = dino.get('pairs', [])
    geometry_path = output/'gt/video_details/camera_pair_metrics.csv'
    with geometry_path.open() if geometry_path.is_file() else __import__('io').StringIO('') as stream:
        geometry = list(csv.DictReader(stream))
    validation = read(output/'gt/input_gt_video/generated_video_validation_gate.json', {})
    expected_epi = {sid(row['scene_id']): 80*sum(bool(edge.get('valid')) for edge in row.get('overlap_graph', []))
                    for row in validation.get('scenes', [])}
    resolved = read(output/'evidence/resolved_config.json', {})
    tvc_cfg = resolved.get('evaluation', {}).get('core_metrics', {}).get('tvc', {})
    tvc = {sid(row['scene_id']): dict(row, scene_id=sid(row['scene_id'])) for row in core.get('tvc', [])}
    for scene in summary.get('consistency', {}).get('scenes', []):
        key = sid(scene.get('scene', scene.get('scene_id')))
        measured = _tvc(scene, tvc_cfg)
        raw = scene.get('backends', {}).get('mtpc', {}).get('result', {})
        support = raw.get('primary_metrics', {}).get('presence', {})
        tvc[key] = dict(measured, scene_id=key,
            expected_visible_agent_camera_frames=support.get('expected_visible_agent_camera_frames'),
            supported_agent_camera_frames=support.get('supported_agent_camera_frames'),
            observer_status=raw.get('status'),
            hard_gates=raw.get('hard_gates', {}), reason=scene.get('reason'),
            source_status=scene.get('status'))
    tvc = {key: value for key, value in tvc.items() if key in selected}
    write_csv(output/'gt/consistency_metrics.csv', list(tvc.values()), headers=['scene_id', 'tvc', 'tvc_status'])
    sources = {'fvd': video.get('fvd', {}), 'cmmd': vq.get('distribution', {}).get('cmmd', {}),
               'clip_iqa': vq.get('metrics', {}).get('clipiqa', {}), 'dino_t': dino,
               'epi_3px': video.get('geometry_consistency', {}), 'tvc': {}}
    stats = video.get('dataset_statistics', {})
    distribution_scenes = {sid(value) for value in stats.get('scene_ids', [])} & set(selected)
    scores, visual, scene_visual = [], [], []
    for key, scene in selected.items():
        fs = [row for row in frames if sid(row.get('scene_id')) == key and number(row.get('clipiqa')) is not None]
        ps = [row for row in pairs if sid(row.get('scene_id')) == key and number(row.get('cosine_similarity')) is not None]
        es = [row for row in geometry if sid(row.get('scene_id')) == key and number(row.get('epipolar_inlier_ratio_3px')) is not None]
        spans = [mean(row['cosine_similarity'] for row in ps if row.get('interval') == span)
                 for span in ('short', 'medium', 'long')]
        values = {'clip_iqa': mean(row['clipiqa'] for row in fs),
                  'dino_t': mean(spans) if all(value is not None for value in spans) else None,
                  'epi_3px': mean(row['epipolar_inlier_ratio_3px'] for row in es), 'tvc': tvc.get(key, {}).get('tvc')}
        for metric, value in values.items():
            source = sources[metric]
            valid = (metric == 'tvc' or bool(_valid_source(source))) and not str(source.get('metric_validity', '')).startswith('invalid')
            entry = dict(scope='scene', branch='gt', dataset=scene['dataset'], scene_id=key, metric=metric,
                value=value if valid else None, raw_value=value,
                status=tvc.get(key, {}).get('tvc_status', 'unavailable') if metric == 'tvc'
                       else source.get('status', 'unavailable') if value is not None and valid else 'unavailable',
                child_status=source.get('status', source.get('metric_validity')), validity=source.get('metric_validity'))
            scores.append(entry); scene_visual.append(entry)
    scopes = [('', set(selected))]+[(dataset, {key for key, row in selected.items() if row['dataset'] == dataset})
                                   for dataset in sorted({row['dataset'] for row in rows})]
    for dataset, ids in scopes:
        for metric, source in sources.items():
            fs = [row for row in frames if sid(row.get('scene_id')) in ids and number(row.get('clipiqa')) is not None]
            ps = [row for row in pairs if sid(row.get('scene_id')) in ids and number(row.get('cosine_similarity')) is not None]
            es = [row for row in geometry if sid(row.get('scene_id')) in ids and number(row.get('epipolar_inlier_ratio_3px')) is not None]
            entries = [row for row in scene_visual if row['scene_id'] in ids and row['metric'] == metric and row['value'] is not None]
            supported = {row['scene_id'] for row in entries}
            coverage, sample_count, frame_count, camera_count, clips, matches = None, 0, 0, 0, None, None
            raw_value, reason = None, source.get('reason')
            if metric in ('fvd', 'cmmd'):
                raw_value = number(source.get(metric)) if not dataset else None
                supported = distribution_scenes & ids if raw_value is not None else set()
                sample_count = source.get('num_gen_clips' if metric == 'fvd' else 'generated_sample_count', 0) if not dataset else 0
                clips = sample_count if metric == 'fvd' else None
                frame_count = sum(count for key, count in stats.get('future_frame_counts', {}).items()
                                  if sid(key.rsplit('/', 1)[0]) in supported)
                camera_count = len(stats.get('cameras', [])) if supported else 0
                coverage = len(supported)/len(ids) if ids else 0
                if dataset:
                    reason = 'dataset_distribution_features_not_separately_scored; population score is not copied'
            elif metric == 'clip_iqa':
                raw_value = mean(row['clipiqa'] for row in fs)
                sample_count = frame_count = len({(sid(r['scene_id']), r['camera'], r['frame_id']) for r in fs})
                camera_count = len({r['camera'] for r in fs})
                coverage = frame_count/(len(ids)*480) if ids else 0
            elif metric == 'dino_t':
                spans = [mean(row['cosine_similarity'] for row in ps if row.get('interval') == span)
                         for span in ('short', 'medium', 'long')]
                raw_value = mean(spans) if all(value is not None for value in spans) else None
                sample_count = len(ps)
                frame_count = len({(sid(r['scene_id']), r['camera'], r[f]) for r in ps for f in ('frame_id', 'next_frame_id')})
                camera_count = len({r['camera'] for r in ps})
                coverage = len({(sid(r['scene_id']), r['camera'], r['interval']) for r in ps})/(len(ids)*18) if ids else 0
            elif metric == 'epi_3px':
                raw_value = mean(row['epipolar_inlier_ratio_3px'] for row in es)
                sample_count = len(es)
                frame_count = len({(sid(r['scene_id']), r['frame_id']) for r in es})
                camera_count = len({camera for r in es for camera in r['camera_pair'].split('__')})
                matches = sum(int(number(r.get('epipolar_valid_match_count')) or 0) for r in es)
                denominator = sum(expected_epi.get(key, 0) for key in ids)
                coverage = len(es)/denominator if denominator else None
                if str(source.get('metric_validity', '')).startswith('invalid'):
                    reason = source['metric_validity']
            else:
                raw_value = mean(row['value'] for row in entries)
                coverage = mean(tvc[key].get('tvc_coverage') for key in supported)
                sample_count = sum(int(tvc[key].get('supported_agent_camera_frames') or 0) for key in supported)
                frame_count = camera_count = None  # Actor-camera observations do not identify unique image frames.
            valid = metric == 'tvc' or bool(_valid_source(source))
            if str(source.get('metric_validity', '')).startswith('invalid'):
                valid = False
            value = raw_value if valid else None
            state = 'UNAVAILABLE' if value is None else 'AVAILABLE'
            if value is not None and (len(supported) < len(ids) or coverage is None or coverage < (1 if metric in ('fvd', 'cmmd', 'clip_iqa') else .8)):
                state = 'PARTIAL_SUPPORT'
            entry = dict(scope='dataset' if dataset else 'population', branch='gt', dataset=dataset,
                metric=metric, value=value, raw_value=raw_value, status=state,
                child_status=source.get('status', source.get('metric_validity', 'see_scene_tvc' if metric == 'tvc' else 'unavailable')),
                validity=source.get('metric_validity'), confidence=source.get('metric_confidence'),
                valid_scenes=len(supported) if valid else 0, requested_scenes=len(ids),
                independent_recordings=recordings(supported) if valid else 0, requested_recordings=recordings(ids),
                samples=sample_count, frames=frame_count, cameras=camera_count, clips=clips, matches=matches,
                coverage=coverage, coverage_unit='scene_camera_interval' if metric == 'dino_t' else
                'available_component_weight' if metric == 'tvc' else 'image' if metric == 'clip_iqa' else
                'camera_pair_frame' if metric == 'epi_3px' else 'scene', reason=reason)
            scores.append(entry)
            if not dataset:
                visual.append(entry)
    write_csv(output/'gt/video_metrics.csv', [r for r in scores if r['metric'] in sources and r['metric'] != 'tvc'])
    agent_results = summary.get('agent', {}).get('agents', {})
    episodes = summary.get('episode_rollout', {}).get('episodes', [])
    agents, termination_rows = [], []
    requested_windows = manifest.get('requested_windows_per_agent', 0)
    core_agents = {row['agent']: row for row in core.get('rpdms_agents', [])}
    core_scenes = {(row['agent'], sid(row['scene_id'])): row for row in core.get('rpdms_scenes', [])}
    input_manifest = read(output/'gt/real_agents/input_manifest.json', {})
    native_datasets={sid(row['scene']):row['dataset'] for row in input_manifest.get('scene_status', [])}
    required_windows = {(sid(row['scene_id']), int(row['frame_id'])) for row in input_manifest.get('common_windows', [])}
    driving_metrics = ('ADE_3s', 'FDE_3s', 'PDMS_open_loop', 'rpdms', 'route_completion_ratio',
                       'collision_rate', 'NC', 'DAC', 'TTC_score', 'Comfort', 'EP')
    for name, label, _, _ in AGENTS:
        result = agent_results.get(name, {})
        scenes = result.get('per_scene', [])
        inference = result.get('inference', {})
        rpdms_datasets=core_agents.get(name, {}).get('rpdms_datasets')
        available_rpdms=[(key, scene) for (agent,key),scene in core_scenes.items()
                        if agent==name and number(scene.get('rpdms_mean')) is not None]
        official_rpdms=[scene for key,scene in available_rpdms
                       if rpdms_datasets is None or native_datasets.get(key) in rpdms_datasets]
        metric_path=output/'gt/real_agents'/name/'agent_details'/name/'per_frame_metrics.csv'
        with metric_path.open() if metric_path.is_file() else __import__('io').StringIO('') as stream:
            metric_windows=list(csv.DictReader(stream))
        own_episodes = [episode for episode in episodes if episode.get('agent_name', episode.get('agent')) == name]
        terminations = dict(Counter(episode.get('termination_reason', 'unavailable') for episode in own_episodes))
        partial_predictions = []
        prediction_path = output/'gt/real_agents'/name/'predictions.jsonl'
        if not inference and prediction_path.is_file():
            with prediction_path.open() as stream:
                for line in stream:
                    try:
                        prediction = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # An interrupted final line is not a completed inference.
                    if prediction.get('inference_success') and sid(prediction.get('scene_id')) in selected:
                        partial_predictions.append(prediction)
        partial_windows = {(sid(row['scene_id']), int(row['frame_id'])) for row in partial_predictions} & required_windows
        successful = int(inference.get('common_predictions', result.get('valid_windows', len(partial_windows))) or 0)
        successful_scenes = sum(int(scene.get('valid_windows', 0) or 0) > 0 for scene in scenes)
        row = dict(agent=name, label=label, status=result.get('status', 'NOT_RUN'),
            successful_scenes=successful_scenes, requested_scenes=len(rows), successful_windows=successful,
            requested_windows=requested_windows, failed_windows=max(0, requested_windows-successful),
            metric_valid_windows=result.get('valid_windows', 0), episodes=len(own_episodes),
            unsupported_scoring_windows=result.get('unsupported_windows', 0),
            valid_episodes=sum(episode.get('status') == 'valid' for episode in own_episodes),
            terminations=terminations, inference_seconds=inference.get('elapsed_s'),
            actual_inference_decisions=inference.get('total_inference_frames', len(partial_predictions)),
            partial_inference_scenes=len({key for key, _ in partial_windows}),
            partial_inference_latency_seconds=sum(number(row.get('latency_s')) or 0 for row in partial_predictions) if partial_predictions else None,
            same_run_resumed=result.get('same_run_resumed', False) or manifest.get('stages', {}).get('gt_agents', {}).get('reuse') == 'same_run_resumed',
            model_inference_executed=result.get('model_inference_executed', False),
            legacy_predictions_reused=result.get('real_prediction_reused', False), reason=result.get('reason'))
        for metric in driving_metrics:
            row[metric] = core_agents.get(name, {}).get(metric) if metric in ('rpdms', 'route_completion_ratio') else mean(s.get(metric) for s in scenes)
            row[metric+'_support_scenes'] = (sum(number(s.get(metric)) is not None for s in scenes)
                if metric not in ('rpdms', 'route_completion_ratio') else
                len(official_rpdms)
                if metric == 'rpdms' else len({sid(e.get('scene_id')) for e in own_episodes if number(e.get(metric)) is not None}))
            row[metric+'_support_windows']=(sum(s['common_window_count'] for s in official_rpdms)
                if all('common_window_count' in s for s in official_rpdms) else None) if metric=='rpdms' else (
                None if metric=='route_completion_ratio' else
                sum(number(window.get(metric)) is not None for window in metric_windows))
            scores.append(dict(scope='agent', branch='gt', agent=name, metric=metric, value=row[metric],
                status='AVAILABLE' if row[metric] is not None else 'UNAVAILABLE',
                successful_scenes=successful_scenes, successful_windows=successful,
                support_scenes=row[metric+'_support_scenes'], support_windows=row[metric+'_support_windows'],
                requested_windows=requested_windows, reason=result.get('reason')))
        for scene in scenes:
            key = sid(scene.get('scene_id'))
            for metric in driving_metrics:
                value = number(core_scenes.get((name, key), {}).get('rpdms_mean')) if metric == 'rpdms' else (
                    mean(e.get(metric) for e in own_episodes if sid(e.get('scene_id')) == key and e.get('status') == 'valid')
                    if metric == 'route_completion_ratio' else number(scene.get(metric)))
                scores.append(dict(scope='scene', branch='gt', agent=name, scene_id=key,
                    dataset=selected.get(key, {}).get('dataset'), metric=metric, value=value,
                    status='AVAILABLE' if value is not None else 'UNAVAILABLE', windows=scene.get('valid_windows')))
        row.update(responsibility_coverage=core_agents.get(name, {}).get('responsibility_coverage'),
            responsibility_confidence=core_agents.get(name, {}).get('responsibility_confidence'),
            rpdms_datasets=core_agents.get(name, {}).get('rpdms_datasets'),
            rpdms_available_scene_measurements=len(available_rpdms),
            route_datasets=core_agents.get(name, {}).get('route_datasets'),
            rpdms_common_window_coverage=core_agents.get(name, {}).get('common_window_coverage'))
        agents.append(row)
        termination_rows.extend(dict(agent=name, reason=reason, episodes=count) for reason, count in terminations.items())
    write_csv(output/'gt/agent_scores.csv', agents)
    write_csv(output/'gt/episode_terminations.csv', termination_rows, headers=['agent', 'reason', 'episodes'])
    populations, traffic = {'gt': gt_population, 'catk': trajectory.get('catk', {})}, []
    for branch, payload in populations.items():
        for dataset, _ in scopes:
            entry = payload.get('datasets', {}).get(dataset, {}) if dataset else payload.get('overall', {})
            boot = entry.get('bootstrap', {})
            line = dict(branch=branch, dataset=dataset or 'overall', status=entry.get('status', 'UNAVAILABLE'),
                valid_scenes=entry.get('valid_scenes'), requested_scenes=entry.get('requested_scenes'),
                independent_recordings=entry.get('independent_clusters'),
                ci95=(boot.get('ci95') or {}).get('TRS'), interval_status=boot.get('status', 'NOT_COMPUTED'),
                interval_limitations=boot.get('limitations'), trajectory_support=entry.get('coverage'),
                **{key: entry.get(key) for key in ('D', 'actor_exposure_rate', 'E_plus', 'E_minus', 'net_delta', 'G', 'TRS')})
            line['E_plus_pp'] = 100*line['E_plus'] if number(line['E_plus']) is not None else None
            traffic.append(line)
            for metric in ('D', 'actor_exposure_rate', 'E_plus', 'E_minus', 'net_delta', 'G', 'TRS'):
                scores.append(dict(scope='dataset' if dataset else 'population', branch=branch, dataset=dataset,
                    metric=metric, value=entry.get(metric), status=entry.get('status', 'UNAVAILABLE'),
                    ci95=(boot.get('ci95') or {}).get(metric), interval_status=line['interval_status'],
                    valid_scenes=line['valid_scenes'], requested_scenes=line['requested_scenes'],
                    independent_recordings=line['independent_recordings']))
    for branch, payload in [('gt', gt_population), ('catk', trajectory)]:
        for row in payload.get('scenes', []):
            for metric, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    scores.append(dict(scope='scene', branch=branch, scene_id=row.get('scene_id'),
                        dataset=row.get('dataset'), metric=metric, value=number(value), status=row.get('status', 'UNAVAILABLE')))
    self_check=dict(self_check)
    invalid_pairs=sum(not row.get('validity') for row in self_check.get('comparison', {}).get('rows', []))
    te_reason=f'incomplete_measured_RPDMS_pairing: {invalid_pairs} invalid scene-Agent pairs' if invalid_pairs else None
    if te_reason:
        self_check['reason']='TE: '+te_reason+'; Q_rel/QTES: '+str(self_check.get('reason'))
    for metric in ('TE', 'Q_rel', 'QTES'):
        scores.append(dict(scope='population', branch='gt', metric=metric, value=number(self_check.get(metric)),
            status='AVAILABLE' if number(self_check.get(metric)) is not None else 'UNAVAILABLE',
            role='reference_self_check', reason=te_reason if metric=='TE' and te_reason else self_check.get('reason')))
    for metric in ('fvd', 'cmmd', 'clip_iqa', 'dino_t', 'epi_3px', 'tvc', 'agent', 'rpdms', 'TE', 'Q_rel', 'QTES'):
        scores.append(dict(scope='population', branch='catk', metric=metric, value=None, status='NOT_RUN_BY_DESIGN'))
    write_csv(output/'scores.csv', scores)
    write_csv(output/'trajectory_scores.csv', traffic)
    limits = [f"{row['metric']}: {row['status']}; {row.get('reason') or row.get('validity') or row['child_status']}"
              for row in visual if row['status'] != 'AVAILABLE']
    limits += [f"{row['metric']}: measured coverage={row['coverage']:.6g} ({row['coverage_unit']}); see per-observation evidence"
               for row in visual if number(row['coverage']) is not None and row['coverage'] < 1]
    observer_statuses = Counter(row.get('observer_status') for row in tvc.values() if row.get('observer_status'))
    if observer_statuses:
        limits.append('TVC observer statuses: '+show(dict(observer_statuses))+'; component availability does not imply perfect visual tracking.')
    rpdms_scopes={(tuple(row['rpdms_datasets']),row['rpdms_support_scenes'],row['rpdms_support_windows'])
                  for row in agents if row.get('rpdms_datasets') and row['rpdms'] is not None}
    for datasets,scenes,windows in sorted(rpdms_scopes,key=str):
        excluded=sorted(set(native_datasets.values())-set(datasets))
        limits.append(f'Official RPDMS: equal dataset mean over {list(datasets)}; {scenes} scenes and {windows} common windows per Agent; '
            f'excluded datasets={excluded}. Available per-scene measurements outside this aggregate remain in scores.csv.')
    if te_reason:
        limits.append('GT reference self-check TE: '+te_reason+'; missing scores are not replaced by 0.5.')
    limits += [f"{row['agent']}: {row.get('reason') or row['status']}; failed/missing windows={row['failed_windows']}"
               for row in agents if row['failed_windows'] or str(row['status']).lower() != 'completed']
    scoring_support=summary.get('agent', {}).get('scoring_time_support', {})
    if scoring_support.get('unsupported'):
        limits.append(f"GT time support: {scoring_support['supported_windows']}/{scoring_support['requested_windows']} "
            f"windows per Agent can be scored under the unchanged {scoring_support['threshold_s']} s interpolation limit; "
            f"{len(scoring_support['unsupported'])} unsupported windows retain their predictions and missing scores.")
    protection=read(output/'evidence/protected_sources_verification.json', {})
    if protection.get('reason'):
        limits.append('Protected source inventory: '+protection['reason'])
    limits += ['PandaSet native lane/map diagnostics: N/A. Agent observed_common_v1 DAC uses existing labelled observed surfaces only; unknown cells and coverage below 0.8 remain unavailable.',
               'FVD/CMMD are measured GT self comparisons; camera views, frames and clips are not independent recordings.',
               'Small dataset recording counts limit confidence; bootstrap replicates do not add independent samples.',
               'Rounded-footprint geometric overlap is a geometric proxy, not confirmed physical collision.',
               'Current decoder consistency does not resolve historical decoded-pixel differences.']
    gt_complete = (all(row['status'] == 'AVAILABLE' for row in visual)
        and all(row['successful_scenes'] == len(rows) and row['failed_windows'] == 0 for row in agents)
        and all(str(summary.get(key, {}).get('status', '')).lower() == 'completed'
                for key in ('trajectory', 'agent', 'traffic_and_responsibility', 'episode_rollout'))
        and gt_population.get('status') == 'COMPLETE')
    catk_complete = trajectory.get('status') == 'COMPLETE' and all(
        row['status'] == 'available' for row in traffic if row['branch'] == 'catk')
    result = dict(status='COMPLETE' if gt_complete and catk_complete else 'PARTIAL',
        gt_status='COMPLETE' if gt_complete else 'PARTIAL', catk_status='COMPLETE' if catk_complete else 'PARTIAL',
        failed_scenes_by_agent={row['agent']: len(rows)-row['successful_scenes'] for row in agents},
        failed_windows=sum(row['failed_windows'] for row in agents), metric_support_limitations=limits,
        unsupported_scoring_windows=sum(row['unsupported_scoring_windows'] for row in agents),
        gt_module_status={key: summary.get(key, {}).get('status', 'UNAVAILABLE')
                          for key in ('video', 'consistency', 'trajectory', 'agent', 'traffic_and_responsibility', 'episode_rollout')},
        catk_population_status=populations['catk'].get('overall', {}).get('status', 'UNAVAILABLE'),
        catk_missing_scene_ids=populations['catk'].get('overall', {}).get('missing_scene_ids', []),
        self_check_status=self_check.get('status', 'UNAVAILABLE'))
    reset = manifest.get('reset_receipt', {})
    protected = manifest.get('protected_source_data_unchanged',
        protection.get('status', 'NOT_VERIFIED'))
    protected = ('PASS' if protected else 'FAIL') if isinstance(protected, bool) else protected
    checked = read(output/'evidence/input_validation.json', {})
    paired = checked.get('paired_scene_ids_identical')
    counts = Counter(row['dataset'] for row in rows)
    actual_decisions = sum(int(row['actual_inference_decisions'] or 0) for row in agents)
    interrupted=read(output/'evidence/recovery1/receipt.json', {})
    interrupted_decisions=int(interrupted.get('completed_native_forwards', 0))
    resumed = sum(row['successful_windows'] for row in agents if row['same_run_resumed'])
    fresh = 'YES' if all(agent_results.get(row['agent'], {}).get('inference', {}).get('status') == 'COMPLETED'
                         and not row['legacy_predictions_reused'] for row in agents) else '部分及原因: see agent support/status'
    lidar_modes = {result.get('inference', {}).get('sensor_protocol') for result in agent_results.values() if result.get('inference')}
    lidar = 'NO' if lidar_modes == {'camera_only'} else 'YES' if 'original_multimodal' in lidar_modes else 'NOT_VERIFIED'
    values = {row['metric']: row['value'] for row in visual}
    terminal = f"""================ ASTRA DRIVE FORMAL 50 ================

OUTPUT RESET
Historical outputs cleared:
{'YES' if reset.get('historical_outputs_cleared') is True else 'NO'}

Archive of old outputs created:
{'NO' if reset.get('archive_created') is False else 'YES' if reset.get('archive_created') else 'NOT_VERIFIED'}

Protected source data unchanged:
{protected}

Current output:
outputs/formal50/

--------------------------------------------------
SCENES
--------------------------------------------------
Total:
{len(rows)}
nuScenes300:
{counts.get('nuscenes300', 0)}
nuPlan:
{counts.get('nuplan', 0)}
AV2:
{counts.get('av2', 0)}
PandaSet:
{counts.get('pandaset', 0)}
GT–CAT-K scene IDs identical:
{'PASS' if paired is True else 'FAIL' if paired is False else 'NOT_VERIFIED'}
Selected manifest hash:
{manifest.get('selection', {}).get('selected_manifest_sha256', 'NOT_VERIFIED')}
Independent recordings:
{recordings(selected)}

--------------------------------------------------
GT VIDEO / CONSISTENCY
--------------------------------------------------
FVD self:
{show(values['fvd'])}
CMMD self:
{show(values['cmmd'])}
CLIP-IQA+:
{show(values['clip_iqa'])}
DINO-T:
{show(values['dino_t'])}
Epi@3:
{show(values['epi_3px'])}
TVC:
{show(values['tvc'])}
Unavailable metrics/support limitations:
{'; '.join(limits)}

--------------------------------------------------
GT DRIVING AGENTS
--------------------------------------------------
{table(['label','successful_scenes','successful_windows','ADE_3s','FDE_3s','rpdms','route_completion_ratio'], agents)}
Fresh GT model inference:
{fresh}
Actual inference decisions:
{actual_decisions+interrupted_decisions}
Accepted run forwards / interrupted attempt forwards:
{actual_decisions} / {interrupted_decisions}
Real LiDAR used:
{lidar}
Legacy prediction results reused:
{'YES' if any(row['legacy_predictions_reused'] for row in agents) else 'NO'}
Same-run resumed predictions:
{resumed}

--------------------------------------------------
TRAJECTORY SCORES
--------------------------------------------------
{table(['branch','D','actor_exposure_rate','E_plus_pp','G','TRS'], [row for row in traffic if row['dataset']=='overall'])}
CAT-K TRS 95% CI:
{show((populations['catk'].get('overall', {}).get('bootstrap', {}).get('ci95') or {}).get('TRS'))}
q:
0.55
500-scene score copied into this run:
NO
CAT-K inference rerun:
NO
CAT-K video/Agent/RPDMS/TE/QTES:
NOT_RUN_BY_DESIGN

--------------------------------------------------
GT SELF-CHECK
--------------------------------------------------
TE:
{show(self_check.get('TE'))}
Q_rel:
{show(self_check.get('Q_rel'))}
QTES:
{show(self_check.get('QTES'))}
Role:
REFERENCE SELF-CHECK ONLY

--------------------------------------------------
COMPLETION
--------------------------------------------------
GT evaluation:
{result['gt_status']}
CAT-K trajectory evaluation:
{result['catk_status']}
Failed scenes/windows:
{show(result['failed_scenes_by_agent'])}; windows={result['failed_windows']}
Unscored windows due to GT support (summed over Agents):
{result['unsupported_scoring_windows']}
Metric support limitations:
{'; '.join(limits)}
Official metric definitions changed:
NO
Reproduction command:
{manifest.get('command', 'UNAVAILABLE')}
Main result:
outputs/formal50/scores.csv
======================================================
"""
    (output/'final_terminal.txt').write_text(terminal, encoding='utf-8')
    lines = ['# AstraDrive formal 50', '',
        'GT observations; log-conditioned offline driving evaluation. This selected existing pool is not a new blind holdout.', '',
        '## A. GT video and consistency', '',
        table(['metric','value','status','valid_scenes','requested_scenes','independent_recordings','samples','frames','cameras','clips','matches','coverage'], visual), '',
        'Per-dataset and per-scene measurements and child statuses are preserved in scores.csv. FVD/CMMD subgroup scores are not copied from the population.', '',
        '## B. GT driving agents', '',
        table(['label','successful_scenes','successful_windows','failed_windows','metric_valid_windows','unsupported_scoring_windows','ADE_3s','FDE_3s','PDMS_open_loop','rpdms','route_completion_ratio','episodes','terminations'], agents), '',
        'ADE/FDE horizon: 3.0 s, 0.5 s output steps. Route completion is a distance ratio. Episode task success is a separate outcome.', '',
        table(['label','collision_rate','NC','DAC','TTC_score','Comfort','EP','responsibility_coverage','inference_seconds','status'], agents), '',
        'Metric support counts, common-window coverage, scored dataset subsets and partial inference counts are retained in gt/agent_scores.csv. Diagnostic averages use their available scene support; official RPDMS and Route Completion retain the frozen common-support aggregation.', '',
        '## C. GT and CAT-K trajectories', '',
        table(['branch','dataset','D','actor_exposure_rate','E_plus_pp','E_minus','net_delta','G','TRS','ci95','valid_scenes','independent_recordings','interval_status'], traffic), '',
        'Geometry: rounded-footprint geometric overlap. E_plus_pp is percentage points; E_minus/net_delta remain fractions. q=0.55.', '',
        '## Reference self-check', '',
        table(['TE','Q_rel','QTES','status','role','reason'], [self_check]), '',
        '## Support and completion', '',
        *['- '+item for item in limits], '',
        f"GT evaluation: {result['gt_status']}; CAT-K trajectory evaluation: {result['catk_status']}.", '',
        'CAT-K video/TVC/Agents/RPDMS/TE/QTES: NOT_RUN_BY_DESIGN. Full terminal receipt: final_terminal.txt.', '',
        f'Actual model forwards: {actual_decisions+interrupted_decisions} = {actual_decisions} accepted-run forwards + {interrupted_decisions} interrupted-attempt forwards. '
        'Recovery evidence is retained under evidence/recovery1 and evidence/recovery2; interrupted predictions are not used for scores.', '',
        'Reproduction command: '+manifest.get('command', 'UNAVAILABLE'), '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output/'evidence/report_status.json', result)
    return result

"""Resumable, gated 4 -> 40 -> 500 scene inference and evidence collection."""
import csv
import hashlib
import json
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from .adapter import prepare_scene, build_graph, sha256
from .export import export_scene, guard_output, verify_output, write_json
from .model import CATKPredictor, scene_seed

QUOTAS={'nuscenes':250,'nuplan':100,'av2':75,'pandaset':75}
MANIFEST_FIELDS=['dataset','scene_id','status','n_agents_input','n_agents_predicted','n_future_only_removed',
                 'history_padding_count','runtime_s','gpu_peak_memory_mb','output_path','output_hash','error']


def implementation_hash():
    paths=sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def verify_source_snapshot(source,snapshot):
    before=json.loads(Path(snapshot).read_text())
    actual={str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()}
    changed=list(set(before)^set(actual))
    for name,p in actual.items():
        if name in before and (p.stat().st_size!=before[name]['size'] or sha256(p)!=before[name]['sha256']):
            changed.append(name)
    result=dict(status='PASS' if not changed else 'FAIL',files=len(actual),changed_files=changed,
                verification='full SHA256 of every source file including all RGB')
    if changed: raise ValueError('input-gt differs from frozen baseline: '+str(changed[:10]))
    return result


def leakage_check(predictor,scene,seed,work):
    """Modify the actual future CSV in a linked scene, then run the real model."""
    baseline=predictor.rollout(scene,seed)
    results={}
    with tempfile.TemporaryDirectory(prefix='leakage-',dir=work) as directory:
        mirror=Path(directory)/scene.path.name;mirror.mkdir()
        for source in scene.path.iterdir():
            if source.name!='future': (mirror/source.name).symlink_to(source)
        (mirror/'future').mkdir()
        with (scene.path/'future/future_trajectories.csv').open() as f:
            reader=csv.DictReader(f); rows=list(reader); fields=reader.fieldnames
        rng=np.random.default_rng(seed)
        for mode in ('randomized_future','zero_invalid_future'):
            with (mirror/'future/future_trajectories.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                order=rng.permutation(len(rows)) if mode=='randomized_future' else np.arange(len(rows))
                for index in order:
                    row=dict(rows[index])
                    for key in ('x','y','z','yaw','width','length','height','vx','vy'):
                        if key in row: row[key]=float(rng.normal()*10000) if mode=='randomized_future' else 0.
                    row['valid']=bool(rng.integers(2)) if mode=='randomized_future' else False
                    writer.writerow(row)
            changed=prepare_scene(mirror)
            value=predictor.rollout(changed,seed)
            difference=max(float(np.max(np.abs(value[k].astype(float)-baseline[k].astype(float)))) for k in baseline)
            equivalent=all(np.array_equal(value[k],baseline[k]) for k in baseline)
            results[mode]=dict(bitwise_equivalent=equivalent,max_abs_difference=difference)
            if not equivalent: raise ValueError(f'Future leakage or nondeterminism: {mode}, difference={difference}')
    # Also verify that direct tensor callers cannot inject future state/validity.
    import torch
    graph=build_graph(scene)
    original=predictor.infer_graphs([graph],seed)
    for key in ('position','heading','velocity'):
        graph['agent'][key][:,11:]=torch.randn_like(graph['agent'][key][:,11:])*1000
    graph['agent']['valid_mask'][:,11:]=True
    changed=predictor.infer_graphs([graph],seed)
    if not all(np.array_equal(original[k],changed[k]) for k in original): raise ValueError('Tensor future leakage')
    results['tensor_future_guard']='PASS'
    return results


def batch_probe(predictor,scene,seed):
    import torch
    graph=build_graph(scene);reference=None; rows=[]
    constrained=predictor.config['sampling']['criterium']=='greedy_map'
    for size in ((1,) if constrained else (1,4,8,16)):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        try:
            result=predictor.infer_graphs([graph]*size,seed)
            torch.cuda.synchronize()
            first={k:v[:len(scene.ids)] for k,v in result.items()}
            if reference is None: reference=first
            equal=all(np.array_equal(reference[k],first[k]) for k in reference)
            copies_equal=[all(np.array_equal(reference[k],result[k][i*len(scene.ids):(i+1)*len(scene.ids)]) for k in reference) for i in range(size)]
            rows.append(dict(batch_size=size,status='PASS',same_scene_same_seed_bitwise=equal,all_batch_positions_same_seed_bitwise=copies_equal,
                             runtime_s=time.perf_counter()-started,gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/2**20))
            del result
        except torch.cuda.OutOfMemoryError as exc:
            rows.append(dict(batch_size=size,status='OOM',error=str(exc)));torch.cuda.empty_cache()
    # Official multinomial sampling advances one batch-wide RNG, not per-scene streams.
    # ponytail: use batch=1 for reproducible scene seeds; batch only after per-scene RNG parity exists upstream.
    return dict(probes=rows,selected_batch_size=1,
                reason=('Per-scene map constraints require batch=1' if constrained else
                        'Official sampling has batch-wide RNG; batch=1 preserves stable independent scene seeds'),
                largest_non_oom_batch=max(r['batch_size'] for r in rows if r['status']=='PASS'))


def downstream_check(scene,output,work,evaluator=False):
    from driveaudit_eval.data.scene_input import load_scene_input
    from driveaudit_eval.trajectory.input_gt import load_scene
    before,after=load_scene_input(scene.path),load_scene_input(output)
    if before.timeline!=after.timeline or not np.array_equal(before.ego_to_anchor,after.ego_to_anchor):
        raise ValueError('Timeline/ego geometry changed')
    tracks,_=load_scene(output)
    if not tracks or any(t.validate() for t in tracks): raise ValueError('Trajectory loader rejected output')
    result=dict(scene_loader='PASS',trajectory_loader='PASS',ego_and_timeline_unchanged=True)
    if evaluator:
        from driveaudit_eval.config import load_config
        from driveaudit_eval.trajectory.evaluator import evaluate
        config=load_config({'defaults':'leaderboard'})['evaluation']['trajectory']
        config.update(mode='input_gt',scene_paths=[str(output)],resume=False)
        # This is a reader smoke: evaluate integrity/kinematics, not unrelated paired quality scores.
        config['metrics']={name:dict(enabled=name in ('integrity','kinematics')) for name in
                           ('integrity','kinematics','map_compliance','collision','interaction','reference','distribution')}
        config.pop('protocol',None)
        context=dict(root=str(Path.cwd()),output_dir=str(work/'downstream_evaluator'),logger=logging.getLogger('catk'),dry_run=False)
        outcome=evaluate(config,context)
        write_json(work/'downstream_evaluator_result.json',outcome)
        if outcome['status']!='completed': raise ValueError('Trajectory evaluator smoke failed: '+str(outcome))
        result['trajectory_evaluator']='PASS'
    return result


def manifest_csv(output,scenes,source,failures):
    target=output/'CATK_INFERENCE_MANIFEST.csv'
    temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=MANIFEST_FIELDS);writer.writeheader()
        for path in scenes:
            out=output/path.relative_to(source);manifest=out/'catk_manifest.json'
            dataset=json.loads((path/'scene_context.json').read_text())['dataset']
            row=dict(dataset=dataset,scene_id=path.name,status='pending',output_path=str(out),error='')
            if manifest.is_file():
                m=json.loads(manifest.read_text());q=m['qc']
                row.update(status=m['status'],n_agents_input=q['n_agents_input'],n_agents_predicted=q['n_agents_predicted'],
                           n_future_only_removed=m['future_only_gt_agents_removed'],history_padding_count=m['history']['padded_points'],
                           runtime_s=m.get('runtime_seconds',0),gpu_peak_memory_mb=m.get('gpu_peak_memory_mb',0),output_hash=m['output_hash'])
            if str(path) in failures: row.update(status='failed',error=failures[str(path)][-1]['error'])
            writer.writerow(row)
    temporary.replace(target)


def run(config,stage='all',resume=False):
    import torch
    source=Path(config['input_root']).resolve();output=Path(config['output_root']).resolve()
    guard_output(source,output)
    if source.parent!=output.parent: raise ValueError('input-catk must be a sibling of input-gt')
    groups={key:[] for key in QUOTAS}
    for path in sorted(source.glob('*/scenes/*')):
        if path.is_dir(): groups[json.loads((path/'scene_context.json').read_text())['dataset']].append(path)
    if {key:len(value) for key,value in groups.items()}!=QUOTAS: raise ValueError('Formal input quotas mismatch')
    scenes=[p for values in groups.values() for p in values]
    output.mkdir(exist_ok=True)
    work=Path(config.get('work_root','outputs/catk_inference')).resolve();work.mkdir(parents=True,exist_ok=True)
    snapshot=work/'input_gt_before.json'
    if not snapshot.is_file(): raise ValueError('Freeze input-gt hashes before starting')
    if (work/'resolved_config.json').exists() and not resume: raise FileExistsError('Existing CAT-K run; use --resume')
    write_json(work/'resolved_config.json',config)
    write_json(work/'source_preflight.json',verify_source_snapshot(source,snapshot))
    print('SOURCE PREFLIGHT: all frozen SHA256 hashes verified',flush=True)
    code_hash=implementation_hash()
    start=time.perf_counter();predictor=CATKPredictor(config)
    write_json(work/'model_load.json',dict(predictor.load_info,repo_sha=predictor.repo_sha,checkpoint_sha256=predictor.checkpoint_sha,
                                        vocabulary_hashes=predictor.vocabulary_hashes,model_config=predictor.model_config))
    failures={};retried=[];status={};completed_this_run=0;skipped=0;processed=set();expected_by_path={}
    def process(path):
        nonlocal completed_this_run,skipped
        out=output/path.relative_to(source)
        for attempt in range(3):
            try:
                scene=prepare_scene(path); seed=scene_seed(scene.dataset,path.name,config['global_seed'])
                if config['sampling']['criterium']=='greedy_map':
                    for name in ('map/semantic_map.npz','map/map_meta.json'):
                        scene.input_hashes[name]=sha256(path/name)
                expected=dict(repo_sha=predictor.repo_sha,checkpoint_sha256=predictor.checkpoint_sha,implementation_sha256=code_hash,
                              seed=seed,model_input_files=scene.input_hashes,
                              sampling=dict(criterium=config['sampling']['criterium'],K=config['sampling']['num_k'],
                                            temperature=config['sampling']['temperature'],seed=seed,n_rollout=1),
                              decoding_policy=config.get('decoding_policy',{}))
                expected_by_path[str(path)]=expected
                if (resume or str(path) in processed) and verify_output(out,expected):
                    skipped+=1;failures.pop(str(path),None)
                    return scene,out,json.loads((out/'catk_manifest.json').read_text())
                if out.exists():
                    if not (out/'catk_manifest.json').is_file(): raise ValueError('Refusing unmanaged existing output '+str(out))
                    quarantine=work/'quarantine'/out.parent.parent.name
                    quarantine.mkdir(parents=True,exist_ok=True)
                    out.rename(quarantine/(out.name+f'-{time.time_ns()}'))
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                pred=predictor.rollout(scene,seed);torch.cuda.synchronize()
                runtime=time.perf_counter()-started
                manifest=dict(expected,status='success',checkpoint_path=config['selected_checkpoint'],token_vocabulary_sha256=predictor.vocabulary_hashes,
                              runtime_seconds=runtime,decoding_diagnostics=predictor.last_rollout_diagnostics,
                              gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,batch_size=1,attempt=attempt+1,
                              source_future_csv_sha256_diagnostic_only=sha256(path/'future/future_trajectories.csv'))
                manifest=export_scene(scene,pred,out,manifest)
                processed.add(str(path))
                completed_this_run+=1;failures.pop(str(path),None)
                print(f"SUCCESS {scene.dataset}/{path.name} agents={len(scene.ids)-1} segments={pred['position'].shape[1]//80} runtime={runtime:.2f}s",flush=True)
                return scene,out,manifest
            except Exception as exc:
                record=dict(attempt=attempt+1,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                failures.setdefault(str(path),[]).append(record);retried.append(dict(scene=str(path),**record))
                write_json(work/'retries.json',retried)
                print(f"RETRY {path.name} attempt={attempt+1} {record['error']}",flush=True)
                torch.cuda.empty_cache()
        return None
    def gate(name,selected,leakage=False):
        rows=[]
        for path in selected:
            result=process(path)
            if result is None: continue
            scene,out,manifest=result
            checks=downstream_check(scene,out,work,evaluator=name=='smoke' and len(rows)==0)
            q=manifest['qc']
            if any(q[key]!=1 for key in ('position_finite_rate','heading_finite_rate','velocity_finite_rate','valid_trajectory_rate')):
                raise ValueError('Prediction finite/validity gate failed')
            if leakage:
                checks['no_future_gt']=leakage_check(predictor,scene,manifest['seed'],work)
            checks.update(status='PASS',dataset=scene.dataset,scene_id=path.name,
                          continuity_checked='native-time derivatives and boundary diagnostics recorded; no quality-based selection',
                          boundary_position_error=q['boundary_position_error'],boundary_heading_error=q['boundary_heading_error'],boundary_speed_error=q['boundary_speed_error'])
            rows.append(checks);write_json(work/(name+'.json'),rows)
        manifest_csv(output,scenes,source,failures)
        if len(rows)!=len(selected): raise ValueError(f'{name} incomplete: {len(rows)}/{len(selected)}')
        status[name]=dict(status='PASS',count=len(rows));print(f'GATE {name} {len(rows)}/{len(selected)} PASS',flush=True)
    smoke=[values[0] for values in groups.values()]
    gate('smoke',smoke,leakage=True)
    probe_scene=prepare_scene(max(smoke,key=lambda p:(p/'history/agents.csv').stat().st_size))
    batch=batch_probe(predictor,probe_scene,scene_seed(probe_scene.dataset,probe_scene.path.name,config['global_seed']))
    write_json(work/'batch_probe.json',batch);print('BATCH',json.dumps(batch),flush=True)
    if stage=='smoke': return status
    validation=[p for values in groups.values() for p in values[:10]]
    gate('validation40',validation,leakage=True)
    if stage=='validation': return status
    for index,path in enumerate(scenes):
        process(path)
        if (index+1)%10==0: manifest_csv(output,scenes,source,failures)
    # Finish a separate failure pass after every scene has had its scheduled attempts.
    for name in list(failures): process(Path(name))
    manifest_csv(output,scenes,source,failures)
    verified={key:0 for key in groups};all_manifests=[]
    for path in scenes:
        out=output/path.relative_to(source)
        if str(path) in failures or str(path) not in expected_by_path or not verify_output(out,expected_by_path[str(path)]): continue
        m=json.loads((out/'catk_manifest.json').read_text());verified[m['dataset']]+=1;all_manifests.append(m)
    immutable=verify_source_snapshot(source,snapshot)
    write_json(work/'source_postflight.json',immutable)
    result=dict(status='SUCCESS' if verified==QUOTAS else 'INCOMPLETE',verified=verified,expected=QUOTAS,input_gt_immutable=immutable,
                runtime_seconds=time.perf_counter()-start,model_initializations=1,computed_this_run=completed_this_run,
                resume_skips=skipped,gates=status,batch=batch,failed_scenes=failures,retry_events=len(retried),
                output_root=str(output),repo_sha=predictor.repo_sha,checkpoint_sha256=predictor.checkpoint_sha,
                implementation_sha256=code_hash)
    write_json(work/'status.json',result)
    write_json(work/'all_scene_manifests.json',all_manifests)
    print(json.dumps(result,indent=2),flush=True)
    if verified!=QUOTAS: raise RuntimeError('Not all 500 scenes succeeded')
    return result

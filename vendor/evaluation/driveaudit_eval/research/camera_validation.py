"""Keyed, finite camera-only paired-subset validation; no experiment paths are implicit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from driveaudit_eval.agents.real_agents import model_input_signature
from driveaudit_eval.agents.adapters.camera_only_temporal import stable_request_seed
FRAMES=set(range(9,19))
FUSION=('diffusiondrivev2','goalflow')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def identity(value):return value.replace('__','/')

def require(value,message):
    if not value:raise ValueError(message)

def keyed(rows,scene):
    selected=[r for r in rows if identity(r['scene_id'])==scene]
    keys=[int(r['frame_id']) for r in selected]
    require(len(keys)==len(set(keys)),'Duplicate frame identity')
    return {int(r['frame_id']):r for r in selected}

def subset_manifest(manifest,scene):
    selected=[r for r in manifest['scenes'] if identity(r['scene_id'])==scene]
    require(len(selected)==1,'Expected one matching scene')
    return dict(scenes=selected)

def mean(values):
    values=np.asarray(list(values),float)
    require(values.size>0 and np.isfinite(values).all(),'Nonfinite or missing metric')
    return float(values.mean())

def subset_metrics(directory,agent,scene):
    """Never substitute the full10-scene aggregate for a one-scene comparison."""
    with (directory/'real_agents'/agent/'agent_details'/agent/'per_frame_metrics.csv').open() as stream:
        rows=keyed(list(csv.DictReader(stream)),scene)
    require(set(rows)==FRAMES,'Expected exactly10 scored metric rows in subset')
    result={field:mean(float(rows[f][field]) for f in sorted(FRAMES)) for field in
        ('PDMS_open_loop','collision_rate','ADE_1s','ADE_2s','ADE_3s','FDE_1s','FDE_2s','FDE_3s')}
    summary=read(directory/'summary.json')
    windows=[r for r in summary['core_metrics']['rpdms_windows'] if r['agent']==agent and identity(r['scene_id'])==scene]
    require({r['frame_id'] for r in windows}==FRAMES and len(windows)==10,'Incomplete subset RPDMS windows')
    require(all(r['rpdms_status']=='valid' for r in windows),'Invalid subset RPDMS')
    result['rpdms']=mean(r['rpdms'] for r in windows)
    scenes=[r for r in summary['core_metrics']['rpdms_scenes'] if r['agent']==agent and identity(r['scene_id'])==scene]
    require(len(scenes)==1 and scenes[0]['common_window_count']==10,'Missing subset RPDMS scene support')
    require(np.isclose(result['rpdms'],scenes[0]['rpdms_mean'],rtol=0,atol=1e-12),'RPDMS scene/window mismatch')
    episodes=[r for r in summary['episode_rollout']['episodes'] if (r.get('agent') or r.get('agent_name'))==agent and identity(r['scene_id'])==scene]
    require(len(episodes)==1 and episodes[0]['status']=='valid' and episodes[0]['route_available'],'Missing valid subset route episode')
    result['route_completion_ratio']=mean([episodes[0]['route_completion_ratio']])
    return result

def compare_subset(dataset,formal,reference,agent,scene):
    row=dict(dataset=dataset,agent=agent,scene=scene,status='NOT_READY',scope='one_scene10_scored_decisions')
    try:
        manifests=[read(p/'real_agents/input_manifest.json') for p in (formal,reference)]
        require(manifests[0]['sensor_protocol']=='camera_only' and manifests[1]['sensor_protocol']=='original_multimodal','Sensor protocol mismatch')
        require(model_input_signature(subset_manifest(manifests[0],scene))==model_input_signature(subset_manifest(manifests[1],scene)),'Allowed model input mismatch')
        subset=subset_manifest(manifests[0],scene)['scenes'][0]
        images={p for frame in subset['frames'] for p in frame['model_input']['camera_images'].values()}
        require(all(manifests[0]['image_sha256'][p]==manifests[1]['image_sha256'][p] for p in images),'RGB hash mismatch')
        if agent not in ('uniad','vad','diffusiondrivev2'):
            require(all(identity(m['scenes'][0]['scene_id'])==scene for m in manifests),'Service RNG prefix differs before selected scene')
        predictions=[];raws=[]
        for directory in (formal,reference):
            source=directory/'real_agents'/agent;status=read(source/'inference_status.json')
            require(status['status']=='COMPLETED' and status['predictions_sha256']==digest(source/'predictions.jsonl'),'Incomplete/changed prediction file')
            predictions.append(keyed(lines(source/'predictions.jsonl'),scene));raws.append(keyed(lines(source/'raw_outputs.jsonl'),scene))
        expected=set(range(19)) if agent in ('uniad','vad') else FRAMES
        require(all(set(p)==expected for p in predictions+raws),'Missing expected forward frames')
        require(list(predictions[0])==list(predictions[1]),'Request-order mismatch')
        differences=[]
        for frame in predictions[0]:
            values=[np.asarray(p[frame]['planning_traj'],float) for p in predictions]
            require(all(v.shape==(6,2) and np.isfinite(v).all() for v in values),'Invalid/nonfinite trajectory')
            if agent in ('uniad','vad'):
                require(all(r[frame]['output'].get('request_seed')==stable_request_seed(subset['scene_id'],frame) for r in raws),'Request-seed mismatch')
            if agent=='diffusiondrivev2':require(raws[0][frame]['output']['seed']==raws[1][frame]['output']['seed'],'Diffusion seed mismatch')
            if agent in FUSION:
                require(raws[0][frame]['output']['sensor_mode']=='zero_lidar' and raws[0][frame]['output']['lidar_feature_nonzero']==0,'Formal zero feature evidence missing')
                require(raws[1][frame]['output']['sensor_mode']=='original_multimodal' and raws[1][frame]['output']['uses_lidar'] is True,'Reference modality mismatch')
            differences.append(values[0]-values[1])
        difference=np.stack(differences)
        scored_difference=np.stack([delta for frame,delta in zip(predictions[0],differences) if frame in FRAMES])
        row.update(actual_forward_pairs=len(differences),scored_pairs=10,prediction_exact=bool(np.array_equal(difference,np.zeros_like(difference))),prediction_maxabs=float(np.abs(difference).max()),prediction_L2_difference=float(np.linalg.norm(scored_difference,axis=-1).mean()))
        metrics=[subset_metrics(p,agent,scene) for p in (formal,reference)]
        for field in metrics[0]:
            row[field+'_formal']=metrics[0][field];row[field+'_reference']=metrics[1][field];row['delta_'+field]=metrics[0][field]-metrics[1][field]
        if agent in FUSION:
            degraded=any(row['delta_'+f]<0 for f in ('PDMS_open_loop','rpdms','route_completion_ratio')) or any(row['delta_'+f]>0 for f in ('collision_rate','ADE_3s','FDE_3s'))
            row['status']='DEGRADED' if degraded else 'PASS'
        else:row['status']='PASS' if row['prediction_exact'] else 'REGRESSION_MISMATCH'
    except (OSError,KeyError,ValueError) as exc:row['reason']=str(exc)
    return row

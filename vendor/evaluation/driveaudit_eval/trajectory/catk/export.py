"""Native-time output with relative links and atomic scene publication."""
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .adapter import sha256
from driveaudit_eval.utils.io import write_json


def guard_output(input_root,output_root):
    source=Path(input_root).resolve(); output=Path(output_root).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from input-gt')


def native_prediction(scene,pred):
    target=scene.timestamps[10:]-scene.timestamps[9]
    times=np.arange(pred['position'].shape[1]+1)/10
    if target[-1]>times[-1]+1e-7: raise ValueError('Prediction does not cover the original timeline')
    n=len(scene.ids)
    position=np.empty((n,len(target),3));heading=np.empty((n,len(target)))
    initial=scene.agent['position'][:,10].astype(float)
    initial_heading=scene.agent['heading'][:,10].astype(float)
    for i in range(n):
        joined=np.vstack((initial[i],pred['position'][i]))
        for axis in range(3): position[i,:,axis]=np.interp(target,times,joined[:,axis])
        headings=np.unwrap(np.r_[initial_heading[i],pred['heading'][i]])
        heading[i]=(np.interp(target,times,headings)+np.pi)%(2*np.pi)-np.pi
    dt=np.diff(scene.timestamps[9:])
    velocity=np.diff(np.concatenate((initial[:,None,:2],position[:,:,:2]),axis=1),axis=1)/dt[None,:,None]
    acceleration=np.diff(np.concatenate((scene.agent['velocity'][:,10:11].astype(float),velocity),axis=1),axis=1)/dt[None,:,None]
    dh=np.diff(np.concatenate((initial_heading[:,None],heading),axis=1),axis=1)
    yaw_rate=((dh+np.pi)%(2*np.pi)-np.pi)/dt[None,:]
    # A query between model steps requires both endpoint states to be valid.
    valid=np.concatenate((scene.agent['valid_mask'][:,10:11],pred['valid']),axis=1)
    right=np.clip(np.searchsorted(times,target),1,len(times)-1)
    active=valid[:,right]&valid[:,right-1]
    return dict(position=position,heading=heading,velocity=velocity,acceleration=acceleration,yaw_rate=yaw_rate,valid=active)


def qc(scene,out):
    traffic=np.array([i for i,aid in enumerate(scene.ids) if aid!='__ego__'],int)
    pos=out['position'][traffic];vel=out['velocity'][traffic];head=out['heading'][traffic]
    active=out['valid'][traffic];shape=scene.agent['shape'][traffic]
    def maximum(x): return float(np.max(x)) if np.size(x) else 0.
    def rate(x): return float(np.isfinite(x).mean()) if np.size(x) else 1.
    current=scene.agent['position'][traffic,10]
    delta=pos[:,0]-current
    hdiff=(head[:,0]-scene.agent['heading'][traffic,10]+np.pi)%(2*np.pi)-np.pi
    speed=np.linalg.norm(vel,axis=-1)
    boundary=dict(position=np.linalg.norm(delta,axis=-1),heading=np.abs(hdiff),
                  speed=np.abs(speed[:,0]-np.linalg.norm(scene.agent['velocity'][traffic,10],axis=-1)))
    # Vectorized exact planar oriented-rectangle SAT, diagnostic only; no prediction edits.
    pairs=np.triu_indices(len(traffic),1); collision_count=0;tested=0
    for t in range(pos.shape[1]):
        a,b=pairs; mask=active[a,t]&active[b,t]
        a=a[mask];b=b[mask];tested+=len(a)
        if not len(a): continue
        u=np.column_stack((np.cos(head[:,t]),np.sin(head[:,t])))
        v=np.column_stack((-u[:,1],u[:,0]));axes=np.stack((u[a],v[a],u[b],v[b]),axis=1)
        displacement=pos[b,t,:2]-pos[a,t,:2]
        distance=np.abs(np.einsum('pi,pki->pk',displacement,axes))
        radius=np.zeros_like(distance)
        for ix in (a,b):
            radius+=np.abs(np.einsum('pi,pki->pk',u[ix],axes))*shape[ix,0,None]/2
            radius+=np.abs(np.einsum('pi,pki->pk',v[ix],axes))*shape[ix,1,None]/2
        collision_count+=int((distance<radius).all(axis=1).sum())
    return dict(position_finite_rate=rate(pos),heading_finite_rate=rate(head),velocity_finite_rate=rate(vel),
                valid_trajectory_rate=float(active.mean()) if active.size else 1.,
                boundary_position_error=maximum(boundary['position']),boundary_heading_error=maximum(boundary['heading']),
                boundary_speed_error=maximum(boundary['speed']),boundary_per_agent={scene.ids[i]:{k:float(v[j]) for k,v in boundary.items()} for j,i in enumerate(traffic)},
                boundary_definition='absolute first-future minus last-history displacement, wrapped heading difference, speed difference; no GT correction',
                max_speed_mps=maximum(speed),max_acceleration_mps2=maximum(np.linalg.norm(out['acceleration'][traffic],axis=-1)),
                future_frames=80,future_duration_s=float(scene.timestamps[-1]-scene.timestamps[9]),
                trajectory_length_m=float(np.linalg.norm(np.diff(np.concatenate((current[:,None],pos),axis=1),axis=1),axis=-1).sum()),
                collision_diagnostic=dict(method='planar_oriented_box_SAT_non_ego_traffic_pairs',overlapping_pair_frames=collision_count,tested_pair_frames=tested),
                n_agents_input=len(traffic),n_agents_predicted=int(active.all(axis=1).sum()),
                agent_count_retention=float(active.all(axis=1).mean()) if len(traffic) else 1.)


FIELDS=['frame_index','agent_id','category','x','y','z','width','length','height','yaw','valid',
        'timestamp','vx','vy','ax','ay','acceleration','yaw_rate']


def output_digest(hashes):
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_output(path,expected=None):
    path=Path(path)
    try:
        manifest=json.loads((path/'catk_manifest.json').read_text())
        if manifest['status']!='success': return False
        if expected and any(manifest.get(k)!=v for k,v in expected.items()): return False
        hashes=manifest['output_files']
        if not hashes or output_digest(hashes)!=manifest['output_hash']: return False
        if any(sha256(path/name)!=digest for name,digest in hashes.items()): return False
        if any(sha256(path/name)!=digest for name,digest in manifest['linked_file_hashes'].items()): return False
        for name,target in manifest['symlinks'].items():
            link=path/name
            if not link.is_symlink() or os.readlink(link)!=target or not link.exists(): return False
        return True
    except (OSError,ValueError,KeyError): return False


def export_scene(scene,pred,output,manifest):
    output=Path(output).absolute()
    guard_output(scene.path,output)
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    out=native_prediction(scene,pred)
    if not all(np.isfinite(out[k]).all() for k in ('position','heading','velocity','acceleration','yaw_rate')):
        raise ValueError('Nonfinite native-time prediction')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.catk-',dir=output.parent))
    try:
        links={}; linked_hashes={}
        def link(source,target):
            relative=os.path.relpath(source,target.parent)
            target.symlink_to(relative,target_is_directory=source.is_dir())
            links[str(target.relative_to(stage))]=relative
            if source.is_file() and source.suffix not in ('.jpg','.png'):
                linked_hashes[str(target.relative_to(stage))]=sha256(source)
        for source in scene.path.iterdir():
            if source.name=='future': continue
            if source.name=='generation_input.json': continue
            link(source,stage/source.name)
        (stage/'future').mkdir()
        for source in (scene.path/'future').iterdir():
            if source.name!='future_trajectories.csv': link(source,stage/'future'/source.name)
        # Hash all linked non-RGB files, including the observed history and calibration.
        for directory in ('history','calibration','map'):
            for source in (scene.path/directory).rglob('*'):
                if source.is_file() and source.suffix not in ('.jpg','.png'):
                    linked_hashes[str(source.relative_to(scene.path))]=sha256(source)
        with (stage/'future/future_trajectories.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for t,frame in enumerate(range(10,90)):
                for i,aid in enumerate(scene.ids):
                    if aid=='__ego__': continue
                    p=out['position'][i,t];shape=scene.agent['shape'][i];v=out['velocity'][i,t];a=out['acceleration'][i,t]
                    row=dict(frame_index=frame,agent_id=aid,category=scene.actor_metadata[i]['category'],
                             x=p[0],y=p[1],z=p[2],length=shape[0],width=shape[1],height=shape[2],yaw=out['heading'][i,t],
                             valid=bool(out['valid'][i,t]),timestamp=scene.timestamps[frame],vx=v[0],vy=v[1],ax=a[0],ay=a[1],
                             acceleration=np.linalg.norm(a),yaw_rate=out['yaw_rate'][i,t])
                    writer.writerow(row)
                for original in scene.static_current:
                    row={k:original[k] for k in FIELDS if k in original}
                    row.update(frame_index=frame,timestamp=scene.timestamps[frame],vx=0,vy=0,ax=0,ay=0,acceleration=0,yaw_rate=0)
                    writer.writerow(row)
        np.savez(stage/'catk_prediction_10hz.npz',**pred,agent_ids=np.array(scene.ids),shape=scene.agent['shape'])
        ego=scene.ids.index('__ego__')
        write_json(stage/'catk_ego_prediction.json',dict(coordinate_frame='history_frame_9_ego',
                   timestamps=scene.timestamps[10:].tolist(),position=out['position'][ego].tolist(),heading=out['heading'][ego].tolist(),
                   velocity=out['velocity'][ego].tolist(),valid=out['valid'][ego].tolist(),canonical_ego_replaced=False))
        generation=json.loads((scene.path/'generation_input.json').read_text())
        generation.update(trajectory_source='CAT-K',traffic_future='future/future_trajectories.csv',
                          ego_output_policy='preserve_source_ego_and_camera_geometry',catk_manifest='catk_manifest.json',
                          future_rgb=False,rgb_semantics='source reference RGB; not generated from CAT-K trajectories')
        write_json(stage/'generation_input.json',generation)
        # GT IDs are read only after inference for removal accounting, never as model input.
        with (scene.path/'future/future_trajectories.csv').open() as f: future_ids={r['agent_id'] for r in csv.DictReader(f)}
        with (scene.path/'history/agents.csv').open() as f: history_ids={r['agent_id'] for r in csv.DictReader(f)}
        metrics=qc(scene,out)
        manifest=dict(manifest,dataset=scene.dataset,scene_id=scene.path.name,source_input_gt=str(scene.path),
                      history=scene.history,actor_metadata=scene.actor_metadata,map_coverage=scene.map_coverage,
                      model_input_files=scene.input_hashes,model_input_policy='history CSV + history ego/calibration + canonical static map; future states zero/invalid',
                      prediction=dict(agent_count=len(scene.ids)-1,future_steps=80,frequency=10,segments=pred['position'].shape[1]//80,
                                      internal_steps_per_segment=91,exported_future_frames=80,continuation_history='predicted_states_only'),
                      no_future_gt_used=True,ego_output_policy='preserve_source_ego_and_camera_geometry',
                      future_only_gt_agents_removed=len(future_ids-history_ids),
                      future_gt_agents_not_active_at_current_removed=len(future_ids-set(scene.ids)-{r['agent_id'] for r in scene.static_current}),
                      static_actor_policy='hold observed current static objects; omit unobserved future objects',
                      rgb_policy='linked original reference RGB; CAT-K-conditioned video generation not run',qc=metrics,
                      symlinks=links,linked_file_hashes=linked_hashes)
        files=['future/future_trajectories.csv','catk_prediction_10hz.npz','catk_ego_prediction.json','generation_input.json']
        manifest['output_files']={name:sha256(stage/name) for name in files}
        manifest['output_hash']=output_digest(manifest['output_files'])
        write_json(stage/'catk_manifest.json',manifest)
        if not verify_output(stage): raise ValueError('Staged output integrity failed')
        from driveaudit_eval.data.scene_input import load_scene_input
        loaded=load_scene_input(stage)
        if len(loaded.timeline)!=90: raise ValueError('Downstream reload failed')
        stage.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise

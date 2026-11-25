"""Copied native-time interpolation and diagnostic functions; no legacy export/GT reads."""
import numpy as np

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

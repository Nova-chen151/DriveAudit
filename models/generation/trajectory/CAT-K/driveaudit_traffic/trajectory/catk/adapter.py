"""Causal input boundary: this module never opens future actor trajectories."""
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TYPES = {'vehicle':0, 'car':0, 'truck':0, 'bus':0, 'trailer':0,
         'motorcycle':0, 'construction_vehicle':0, 'pedestrian':1, 'bicycle':2, 'cyclist':2}
DEFAULT_SHAPES = {0:(4.8,2.0,1.6), 1:(0.8,0.8,1.7), 2:(2.0,0.7,1.7)}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def actor_type(category):
    return TYPES.get(category)


def is_valid(value):
    return str(value).lower() in ('true','1','yes')


@dataclass
class PreparedScene:
    path: Path
    dataset: str
    ids: list
    agent: dict
    actor_metadata: list
    static_current: list
    timestamps: np.ndarray
    history_times: np.ndarray
    history: dict
    map_data: dict
    map_coverage: dict
    input_hashes: dict


def prepare_scene(path):
    path=Path(path).resolve()
    hashes={}
    def read(name):
        hashes[name]=sha256(path/name)
        return json.loads((path/name).read_text())
    timeline=read('timeline.json')
    frames=timeline['frames']
    if timeline.get('unit')!='microseconds_relative_to_frame_0' or [r['frame_index'] for r in frames]!=list(range(90)):
        raise ValueError('Expected native ordered 90-frame microsecond timeline')
    ts=np.array([r['reference_time_us'] for r in frames],dtype=float)/1e6
    if not np.isfinite(ts).all() or not (np.diff(ts)>0).all(): raise ValueError('Invalid timestamps')
    if read('task.json')['reference_frame']!=9: raise ValueError('Expected current frame 9')
    context=read('task.json');context=dict(context,dataset=context['source_dataset'])
    if context['protocol']!='nuscenes_ego_conditioned_v1':raise ValueError('Unexpected participant protocol')
    queries=ts[9]+np.arange(-10,1)/10
    grouped=defaultdict(list)
    hashes['history/agents.csv']=sha256(path/'history/agents.csv')
    with (path/'history/agents.csv').open() as f:
        for row in csv.DictReader(f):
            frame=int(row['frame_index'])
            if not 0<=frame<=9: raise ValueError('Future frame in history file')
            if is_valid(row['valid']): grouped[row['agent_id']].append(row)
    records=[]; static=[]
    for aid,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:int(r['frame_index']))
        if len({r['frame_index'] for r in rows})!=len(rows): raise ValueError('Duplicate history frame')
        if int(rows[-1]['frame_index'])!=9: continue
        if actor_type(rows[-1]['category']) is None:
            if rows[-1]['category'] in ('barrier','traffic_cone','static_object'):
                static.append(rows[-1])
            continue
        if any(r['category']!=rows[-1]['category'] for r in rows): raise ValueError('Changing history category')
        records.append((aid,rows))
    # The ego CSV has no dimensions/z; read only historical calibrated poses for z.
    poses=read('calibration/history_ego_poses.json')
    if poses['coordinate_frame'] not in ('history_frame_9_ego','history_last_ego'):
        raise ValueError('Unsupported anchor coordinates')
    ego_z=np.array(poses['ego_to_anchor'][:10],float)[:,2,3]
    hashes['history/ego.csv']=sha256(path/'history/ego.csv')
    with (path/'history/ego.csv').open() as f: ego=list(csv.DictReader(f))
    if [int(r['frame_index']) for r in ego]!=list(range(10)): raise ValueError('Invalid ego history')
    ego_pos=np.array([[float(r['x']),float(r['y'])] for r in ego])
    ego_vel=np.vstack(((ego_pos[1]-ego_pos[0])/(ts[1]-ts[0]),np.diff(ego_pos,axis=0)/np.diff(ts[:10])[:,None]))
    for i,row in enumerate(ego):
        row.update(z=ego_z[i],vx=ego_vel[i,0],vy=ego_vel[i,1],category='vehicle')
    if '__ego__' in grouped: raise ValueError('Reserved ego ID collision')
    records.append(('__ego__',ego))
    n=len(records)
    agent=dict(position=np.zeros((n,91,3),np.float32),heading=np.zeros((n,91),np.float32),
               velocity=np.zeros((n,91,2),np.float32),valid_mask=np.zeros((n,91),bool),
               shape=np.zeros((n,3),np.float32),type=np.zeros(n,np.uint8),
               role=np.zeros((n,3),bool),id=np.arange(n,dtype=np.int64))
    metadata=[]
    for i,(aid,rows) in enumerate(records):
        times=ts[[int(r['frame_index']) for r in rows]]
        pos=np.array([[float(r[k]) for k in ('x','y','z')] for r in rows])
        head=np.unwrap([float(r['yaw']) for r in rows])
        # Exported native annotation velocities can use a future annotation. Derive
        # velocities only from positions inside this already-observed history.
        vel=np.zeros((len(rows),2))
        if len(rows)>1:
            vel[1:]=np.diff(pos[:,:2],axis=0)/np.diff(times)[:,None]
            vel[0]=vel[1]
        if not all(np.isfinite(x).all() for x in (pos,head,vel)): raise ValueError('Nonfinite history')
        interp=np.column_stack([np.interp(queries,times,pos[:,k]) for k in range(3)])
        before=queries<times[0]-1e-8
        interp[before,:2]=pos[0,:2]+(queries[before]-times[0])[:,None]*vel[0]
        agent['position'][i,:11]=interp
        agent['heading'][i,:11]=(np.interp(queries,times,head)+np.pi)%(2*np.pi)-np.pi
        agent['velocity'][i,:11]=np.column_stack([np.interp(queries,times,vel[:,k]) for k in range(2)])
        agent['valid_mask'][i,:11]=True
        kind=actor_type(rows[-1]['category']); agent['type'][i]=kind
        shape=[float(rows[-1].get(k,0)) for k in ('length','width','height')]
        default_fields=[k for k,v in zip(('length','width','height'),shape) if not np.isfinite(v) or v<=0]
        agent['shape'][i]=[v if np.isfinite(v) and v>0 else DEFAULT_SHAPES[kind][j] for j,v in enumerate(shape)]
        agent['role'][i,0]=aid=='__ego__'; agent['role'][i,2]=True
        metadata.append(dict(agent_id=aid,category=rows[-1]['category'],shape_source='documented_class_default' if default_fields else 'canonical_current_dimensions',
                             velocity_source='observed_history_position_differences' if len(rows)>1 else 'zero_no_observed_motion_baseline',
                             default_shape_fields=default_fields,padded_points=int(before.sum()),observed_points=len(rows),
                             earliest_observation_s=float(times[0]),history_padding_method='backward_constant_velocity' if before.any() else 'timestamp_resampling'))
    vectors=read('map/generation_vectors.json'); meta=read('map/map_meta.json')
    if not vectors['coordinate_frame'].startswith('history_frame_9_ego'): raise ValueError('Map frame mismatch')
    lines=list(vectors['lines']); raster_count=0
    if not lines and 'lane_divider' in meta.get('available_channels',[]):
        import cv2
        hashes['map/semantic_map.npz']=sha256(path/'map/semantic_map.npz')
        with np.load(path/'map/semantic_map.npz',allow_pickle=False) as data:
            mask=data['lane_marking_observed'] if 'lane_marking_observed' in data else data['mask'][meta['channels'].index('lane_divider')]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # Measured lane-marking patch outlines, never unknown/drivable-area borders.
        for contour in contours:
            ij=contour[:,0,::-1]
            if len(ij)<2: continue
            xy=(ij+.5)*meta['resolution_m']+np.array(meta['extent_m'])[::2]
            lines.append(dict(points=np.column_stack((xy,np.zeros(len(xy)))).tolist(),label=0,source='observed_lane_marking_patch_outline'))
        raster_count=len(lines)
    coverage=dict(source=vectors.get('source','canonical_generation_vectors'),static_map_causality=meta.get('causality','preexisting_static_map'),
                  source_lines=len(lines),raster_marking_outlines=raster_count,available_channels=meta.get('available_channels',[]),
                  unavailable_channels=meta.get('unavailable_channels',[]),traffic_light_coverage=0,traffic_light_policy='NO_LANE_STATE',
                  lane_centerlines='unavailable; canonical divider, boundary, crosswalk geometry only')
    return PreparedScene(path,context['dataset'],[r[0] for r in records],agent,metadata,static,ts,queries,
                         dict(observed_points=10,padded_points=int((queries<ts[0]-1e-8).sum()),
                              padding_method='backward_constant_velocity_then_timestamp_resampling',
                              agent_padding_total=sum(m['padded_points'] for m in metadata),
                              earlier_canonical_actor_observations_available=False),
                         dict(lines=lines),coverage,hashes)


def build_graph(scene):
    import torch
    from torch_geometric.data import HeteroData
    from src.smart.utils.preprocess import _interplating_polyline
    positions=[]; headings=[]; types=[]; polygons=[]
    for line in scene.map_data['lines']:
        xy=np.asarray(line['points'],float)[:,:2]
        if not np.isfinite(xy).all(): raise ValueError('Nonfinite map geometry')
        xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-6]]
        if len(xy)<2: continue
        arclength=np.r_[0,np.linalg.norm(np.diff(xy,axis=0),axis=1).cumsum()]
        distances=np.r_[np.arange(0,arclength[-1],1.),arclength[-1]]
        dense=np.column_stack([np.interp(distances,arclength,xy[:,k]) for k in range(2)])
        chunks=_interplating_polyline(dense)
        if chunks is None: continue
        kind=line.get('kind')
        if kind=='lane': point_type,polygon_type=1,0
        elif int(line['label'])==1: point_type,polygon_type=9,3
        elif int(line['label'])==2: point_type,polygon_type=4,1
        else:
            mark=str(line.get('attributes',{}).get('mark_type','')).upper()
            point_type={'DOUBLE_DASH_YELLOW':6,'DASH_SOLID_YELLOW':8,'SOLID_DASH_YELLOW':8}.get(mark)
            if point_type is None:
                point_type=8 if 'DOUBLE' in mark else 6 if ('DASH' in mark or 'BROKEN' in mark) else 7
            polygon_type=2
        positions.append(chunks[:,:,:2]);headings.append(chunks[:,0,2])
        types.extend([point_type]*len(chunks));polygons.extend([polygon_type]*len(chunks))
    if positions:
        pos=torch.cat(positions);head=torch.cat(headings)
    else:
        pos=torch.full((1,3,2),60000.,dtype=torch.float32);head=torch.zeros(1);types=[0];polygons=[0]
    graph=HeteroData()
    graph.map_scene_path=str(scene.path)
    for key,value in scene.agent.items(): graph['agent'][key]=torch.from_numpy(value.copy())
    graph['agent']['road_start_position']=torch.from_numpy(scene.agent['position'][:,10,:2].copy())
    graph['agent']['road_start_heading']=torch.from_numpy(scene.agent['heading'][:,10].copy())
    graph['agent']['num_nodes']=len(scene.ids)
    graph['map_save']['traj_pos']=pos
    graph['map_save']['traj_theta']=head
    graph['map_save']['num_nodes']=len(pos)
    graph['pt_token']['type']=torch.tensor(types,dtype=torch.uint8)
    graph['pt_token']['pl_type']=torch.tensor(polygons,dtype=torch.uint8)
    graph['pt_token']['light_type']=torch.zeros(len(pos),dtype=torch.uint8)
    graph['pt_token']['num_nodes']=len(pos)
    return graph

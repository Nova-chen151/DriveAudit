"""Preserved numeric helpers for timestamp replay and supplemental plan metrics."""
import numpy as np
from driveaudit_eval.agents.core import EGO_LENGTH, EGO_WIDTH, COMFORT_LIMITS, trajectory_kinematics
from driveaudit_eval.agents.open_loop_trajectory_evaluator import box_world
from driveaudit_eval.utils.map_provider.geometry import wrap_angle


def axes(yaw):
    return np.stack((np.stack((np.cos(yaw), np.sin(yaw)), axis=-1),
                     np.stack((-np.sin(yaw), np.cos(yaw)), axis=-1)), axis=-2)


def overlap(ego, eyaw, other, oyaw, length, width):
    """Broadcast SAT for oriented rectangles; touching counts as contact."""
    ea, oa = axes(eyaw), axes(oyaw)
    directions = np.concatenate((ea, oa), axis=-2)
    d = np.abs(np.einsum('...ki,...i->...k', directions, other-ego))
    er = np.sum(np.abs(directions @ np.swapaxes(ea, -1, -2)) * [EGO_LENGTH/2, EGO_WIDTH/2], axis=-1)
    sizes = np.stack((length/2, width/2), axis=-1)
    ar = np.sum(np.abs(directions @ np.swapaxes(oa, -1, -2)) * sizes[..., None, :], axis=-1)
    return np.all(d <= er+ar+1e-9, axis=-1)


def cv_ttc(ego, eyaw, velocity, other, oyaw, other_velocity, length, width, horizon=3.):
    """Continuous SAT time of contact for fixed-heading constant-velocity OBBs."""
    ea, oa = axes(np.asarray(eyaw)), axes(np.asarray(oyaw))
    directions = np.concatenate((ea, oa))
    radius = np.abs(directions @ ea.T) @ np.array([EGO_LENGTH/2, EGO_WIDTH/2])
    radius += np.abs(directions @ oa.T) @ np.array([length/2, width/2])
    distance = directions @ (other-ego); speed = directions @ (other_velocity-velocity)
    low, high = 0., horizon
    for d, v, r in zip(distance, speed, radius):
        if abs(v) < 1e-10:
            if abs(d) > r: return np.inf
        else:
            a, b = sorted(((-r-d)/v, (r-d)/v))
            low, high = max(low, a), min(high, b)
    return low if low <= high else np.inf


def interpolate(points, times, query):
    return np.column_stack([np.interp(query, times, points[:, j]) for j in range(points.shape[1])])


def scene_tracks(scene):
    tracks = {}
    for f in scene['frames']:
        for row in f['evaluation_only']['background_boxes']:
            b = box_world(row, f)
            tracks.setdefault(b['id'], []).append([float(f['timestamp'])/1e6, b['x'], b['y'], b['yaw'], b['length'], b['width']])
    return {k: np.asarray(v) for k,v in tracks.items()}


def replay_boxes(tracks, query):
    result=[]
    for ident, track in tracks.items():
        valid=(query >= track[0,0]) & (query <= track[-1,0])
        if not valid.any(): continue
        # Do not bridge a track disappearance longer than 0.3 seconds.
        ix=np.clip(np.searchsorted(track[:,0],query,side='right'),1,max(1,len(track)-1))
        if len(track)>1:
            exact=np.min(abs(query[:,None]-track[None,:,0]),axis=1)<1e-6
            valid &= exact | ((track[ix,0]-track[ix-1,0])<=.3)
        data=track[:,1:].copy();data[:,2]=np.unwrap(data[:,2])
        result.append((ident,interpolate(data,track[:,0],query),valid))
    return result


def causal_boxes(scene, index):
    """Use only current boxes and backward position differences, never vx labels."""
    current=scene['frames'][index]; t=float(current['timestamp'])/1e6
    prev=scene['frames'][index-1]; dt=t-float(prev['timestamp'])/1e6
    old={b['id']:b for b in (box_world(r,prev) for r in prev['evaluation_only']['background_boxes'])}
    output=[]
    for r in current['evaluation_only']['background_boxes']:
        b=box_world(r,current); pos=np.array([b['x'],b['y']]); matched=b['id'] in old
        vel=(pos-np.array([old[b['id']]['x'],old[b['id']]['y']]))/dt if matched else np.zeros(2)
        output.append((b,pos,vel,matched))
    return output


def rates(hit, grid, prefix):
    return {prefix+str(h)+'s':float(np.any(hit[grid<=h+1e-8])) for h in (1,2,3)}


def plan_metrics(local, dt, current_velocity, current_yaw, bounds):
    points=np.vstack(([0.,0.],local));k=trajectory_kinematics(points,dt)
    active=k['speed'][:-1]>.5
    curvature=np.abs(k['curvature'][active])
    feasible=(np.all(k['speed']<=bounds['max_speed']) and np.all(k['lon_acc']<=bounds['max_accel'])
              and np.all(k['lon_acc']>=bounds['max_decel']) and (not len(curvature) or np.max(curvature)<=1/bounds['R']))
    return {'kinematic_feasible_proxy':float(feasible),
            'initial_speed_error_mps':float(abs(np.linalg.norm(local[0])/dt-np.linalg.norm(current_velocity))),
            'initial_heading_error_rad':float(abs(wrap_angle(np.arctan2(local[0,1],local[0,0])))),
            'longitudinal_acceleration_violation':float(np.any((k['lon_acc']>COMFORT_LIMITS['lon_acc_pos']) | (k['lon_acc']<COMFORT_LIMITS['lon_acc_neg']))),
            'lateral_acceleration_violation':float(np.any(abs(k['lat_acc'])>COMFORT_LIMITS['lat_acc'])),
            'jerk_violation':float(np.any(abs(k['jerk'])>COMFORT_LIMITS['jerk']))}


def ego_progress_proxy(progress_m, current_speed_mps, horizon_s=3., speed_floor_mps=1.):
    """Declared current-speed reference; no expert future or model-relative normalization."""
    values = np.asarray([progress_m, current_speed_mps, horizon_s, speed_floor_mps], dtype=float)
    if not np.isfinite(values).all() or current_speed_mps < 0 or horizon_s <= 0 or speed_floor_mps <= 0:
        raise ValueError('Invalid EP inputs')
    target_m = horizon_s * max(current_speed_mps, speed_floor_mps)
    return float(np.clip(progress_m / target_m, 0., 1.)), float(target_m)



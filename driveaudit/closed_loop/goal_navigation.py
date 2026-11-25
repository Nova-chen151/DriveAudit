"""Explicit goal-to-command adapter; the native DrivoR network is unchanged."""
import math
import numpy as np


def command_for_goal(pose, config):
    goal=np.asarray(config['goal_anchor_xy'],float);pose=np.asarray(pose,float)
    if goal.shape!=(2,) or pose.shape!=(4,) or not np.isfinite(np.r_[goal,pose]).all():
        raise ValueError('Goal/pose must be finite in the shared anchor frame')
    threshold=float(config.get('turn_threshold_deg',30.));radius=float(config.get('arrival_radius_m',.5))
    if not 0<threshold<90 or not 0<radius<=2:raise ValueError('Invalid goal-navigation thresholds')
    delta=goal-pose[:2];distance=float(np.linalg.norm(delta));yaw=float(pose[3])
    local=[math.cos(yaw)*delta[0]+math.sin(yaw)*delta[1],-math.sin(yaw)*delta[0]+math.cos(yaw)*delta[1]]
    error=math.atan2(local[1],local[0]) if distance>1e-10 else 0.
    code=1 if error>math.radians(threshold) else 0 if error<-math.radians(threshold) else 2
    command={'code':code,'name':{0:'RIGHT',1:'LEFT',2:'STRAIGHT'}[code]}
    return command,dict(goal_anchor_xy=goal.tolist(),goal_ego_xy=local,distance_m=distance,bearing_error_deg=math.degrees(error),
        command=command,arrival_radius_m=radius,inside_arrival_region=distance<=radius,
        network_input='native_discrete_navigation_command_only',goal_coordinates_consumed_by='external_goal_navigation_adapter')


def apply_goal_terminal(prefix, current_pose, config, already_reached=False):
    """Absorbing terminal state after first entry into the fixed arrival region.

    Does not change any pre-arrival predicted position or use a reference route.
    This is an episode terminal convention, not a learned braking controller.
    """
    prefix=np.asarray(prefix,float).copy();current=np.asarray(current_pose,float)
    if prefix.shape!=(5,4) or not np.isfinite(prefix).all():raise ValueError('Expected five finite executed substeps')
    goal=np.asarray(config['goal_anchor_xy'],float);radius=float(config.get('arrival_radius_m',.5))
    if already_reached or np.linalg.norm(current[:2]-goal)<=radius:
        prefix[:]=current;return prefix,dict(reached=True,first_reached_offset_s=0.,terminal_hold=True)
    hits=np.flatnonzero(np.linalg.norm(prefix[:,:2]-goal,axis=1)<=radius)
    if len(hits):
        k=int(hits[0]);prefix[k:]=prefix[k];return prefix,dict(reached=True,first_reached_offset_s=(k+1)*.1,terminal_hold=True)
    return prefix,dict(reached=False,first_reached_offset_s=None,terminal_hold=False)

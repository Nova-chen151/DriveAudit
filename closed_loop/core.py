"""Fixed history-anchor coordinates; ideal execution of a 0.5s policy prefix."""
import json
import math
from pathlib import Path
import numpy as np


def pose_matrix(pose):
    x,y,z,yaw = pose
    c,s = math.cos(yaw), math.sin(yaw)
    return np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,z],[0,0,0,1]],float)


def execute_prefix(pose, native, offsets):
    traj=np.asarray(native,float); times=np.asarray(offsets,float)
    if traj.shape!=(8,3) or not np.isfinite(traj).all() or not np.allclose(times,np.arange(1,9)*.5):
        raise ValueError('Expected native 8x3 ego trajectory at 0.5s intervals')
    local=np.arange(1,6)[:,None]/5*traj[0]
    # Interpolate the shortest heading arc from current ego heading.
    local[:,2]=np.arange(1,6)/5*math.atan2(math.sin(traj[0,2]),math.cos(traj[0,2]))
    T=pose_matrix(pose)
    xy=local[:,:2]@T[:2,:2].T+T[:2,3]
    return np.column_stack((xy,np.full(5,pose[2]),local[:,2]+pose[3]))


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False));tmp.replace(path)

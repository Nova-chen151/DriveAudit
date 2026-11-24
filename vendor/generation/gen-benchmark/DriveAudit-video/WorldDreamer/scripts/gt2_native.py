"""Recover the model's LiDAR rig and original-resolution camera tokens.

GT2 images stay in their public challenge geometry.  Camera tokens use a
1600x900 virtual image plane, whose direct resize recovers that geometry.
For nuScenes the rig is the retained physical top-LiDAR calibration. For
other datasets a nominal nuScenes-height virtual rig is explicit metadata.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from genbench_gt2.schema import CameraCalibration

NATIVE_SIZE = (1600, 900)
VIRTUAL_TRANSLATION = np.array([0.985793, 0.0, 1.84019])
VIRTUAL_ROTATION_WXYZ = np.array([2**-.5, 0., 0., -2**-.5])


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError('invalid sensor-to-ego quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@lru_cache(maxsize=512)
def _rig(scene_directory, dataset):
    data = json.loads((Path(scene_directory)/'calibration/cameras.json').read_text())
    if dataset == 'nuscenes' and 'renderer' in data:
        renderer = data['renderer']
        if renderer.get('coordinate_frame') != 'source_lidar_top_at_history_last_frame':
            raise ValueError('nuScenes source top-LiDAR calibration missing')
        q = renderer['sensor2ego_rotation_wxyz']
        t = renderer['sensor2ego_translation_m']
        source = 'retained_nuscenes_top_lidar_calibration'
    else:
        if dataset == 'nuscenes':
            marker = Path(scene_directory)/'participant_render_frame.json'
            if not marker.is_file():
                raise ValueError('missing source LiDAR calibration without explicit participant virtual-rig declaration')
        q, t = VIRTUAL_ROTATION_WXYZ, VIRTUAL_TRANSLATION
        source = 'explicit_virtual_nuscenes_height_lidar_rig'
    native_to_ego = np.eye(4)
    native_to_ego[:3,:3] = quaternion_matrix(q)
    native_to_ego[:3,3] = np.asarray(t, dtype=np.float64)
    return np.linalg.inv(native_to_ego), source


def ego_to_native(inputs):
    """Column-vector transform; includes translation, roll, pitch and yaw."""
    asset_root = inputs.scene.asset_root or inputs.scene.root
    return _rig(str(asset_root/inputs.scene.relative_path), inputs.scene.dataset)[0].copy()


def native_calibration(inputs, camera):
    """Camera-to-native extrinsics with 1600x900 camera-token intrinsics."""
    c = inputs.calibrations[camera]
    k = c.intrinsics.copy()
    k[0] *= 1600/400
    k[1] *= 900/224
    return CameraCalibration(camera, k, ego_to_native(inputs) @ c.camera_to_ego)


def transform_xyz(points, transform):
    p = np.asarray(points)
    return p @ transform[:3,:3].T + transform[:3,3]


def rig_metadata(inputs):
    asset_root = inputs.scene.asset_root or inputs.scene.root
    transform, source = _rig(str(asset_root/inputs.scene.relative_path), inputs.scene.dataset)
    return dict(source=source, ego_to_native=transform.tolist(),
                camera_token_image_size=list(NATIVE_SIZE),
                challenge_image_size=[400,224])


def clip_polyline_xy(points, bounds=(-50.,50.,-50.,50.)):
    """Clip segments to native XY rectangle, preserving Z and disconnected runs.

    This is the local-patch step performed before the released map renderer.
    Keeping only in-bounds vertices would drop lines crossing the whole patch.
    """
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2:
        return []
    if not np.isfinite(points).all():
        raise ValueError('map polyline contains nonfinite coordinates')
    xmin,xmax,ymin,ymax=bounds
    if xmin>=xmax or ymin>=ymax:
        raise ValueError('invalid local map rectangle')
    lo=points[:,:2].min(axis=0);hi=points[:,:2].max(axis=0)
    if hi[0]<xmin or lo[0]>xmax or hi[1]<ymin or lo[1]>ymax:
        return []
    if lo[0]>=xmin and hi[0]<=xmax and lo[1]>=ymin and hi[1]<=ymax:
        return [points.copy()]
    result=[];run=[]
    for a,b in zip(points[:-1],points[1:]):
        d=b-a;lower=0.;upper=1.;visible=True
        for p,q in ((-d[0],a[0]-xmin),(d[0],xmax-a[0]),(-d[1],a[1]-ymin),(d[1],ymax-a[1])):
            if abs(p)<1e-14:
                if q<0:visible=False;break
            elif p<0:lower=max(lower,q/p)
            else:upper=min(upper,q/p)
            if lower>upper:visible=False;break
        if not visible or lower>=upper:
            if len(run)>1:result.append(np.asarray(run))
            run=[];continue
        start=a+lower*d;end=a+upper*d
        if run and np.allclose(run[-1],start,atol=1e-9,rtol=0):
            run.append(end)
        else:
            if len(run)>1:result.append(np.asarray(run))
            run=[start,end]
        if upper<1.:
            result.append(np.asarray(run));run=[]
    if len(run)>1:result.append(np.asarray(run))
    return result


@lru_cache(maxsize=512)
def _anchor_to_world(scene_directory):
    scene_directory=Path(scene_directory)
    for filename in ('generation_provenance.json','provenance.json'):
        path=scene_directory/filename
        if path.exists():
            document=json.loads(path.read_text())
            if 'anchor_ego_to_world' in document:
                transform=np.asarray(document['anchor_ego_to_world'],dtype=np.float64)
                if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
                    raise ValueError('invalid retained global anchor pose')
                return transform
    marker = scene_directory/'participant_render_frame.json'
    if marker.is_file():
        declaration=json.loads(marker.read_text())
        if declaration.get('world_frame')=='history_frame_9_ego_local_origin':
            return np.eye(4)
    raise ValueError('GT2 scene lacks retained anchor_ego_to_world provenance')


def ego_to_world(inputs, frame):
    """Recover the original global pose needed by DreamForge can_bus."""
    asset_root = inputs.scene.asset_root or inputs.scene.root
    return _anchor_to_world(str(asset_root/inputs.scene.relative_path)) @ inputs.ego_to_anchor[frame]

"""Canonical input-gt scene contract. Core inputs never require native dataset tables."""
import csv
import json
import bisect
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from driveaudit_eval.agents.core import WORLD_DREAMER_CAMERA_ORDER, derive_command, derive_kinematics, quat_to_yaw
from driveaudit_eval.agents.agent_input_adapter import assert_no_evaluation_leakage

CANONICAL_CAMERA_NAMES = tuple(WORLD_DREAMER_CAMERA_ORDER)

def quaternion(transform):
    return Rotation.from_matrix(np.asarray(transform)[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()


def interpolate_poses(times, poses, query):
    times = np.asarray(times, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    poses = np.asarray(poses, dtype=float)
    if len(times) < 2 or poses.shape != (len(times), 4, 4) or np.any(np.diff(times) <= 0) or not np.isfinite(poses).all():
        raise ValueError('Invalid native pose samples')
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError('Native pose extrapolation is forbidden')
    ix = np.clip(np.searchsorted(times, query), 1, len(times)-1)
    exact = (times[ix] == query) | (times[ix-1] == query)
    if np.any((times[ix]-times[ix-1] > 250000) & ~exact):
        raise ValueError('Native pose interpolation gap exceeds 250 ms')
    t = (times-times[0])/1e6
    q = (query-times[0])/1e6
    out = np.repeat(np.eye(4)[None], len(query), axis=0)
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(poses[:, :3, :3]))(q).as_matrix()
    for axis in range(3): out[:, axis, 3] = np.interp(q, t, poses[:, axis, 3])
    return out


def canvas_intrinsics(camera):
    """Services restore the stored RGB to a 1600x900 canvas, independent of source rig."""
    width, height = camera['challenge_image_size']
    k = np.asarray(camera['challenge_intrinsics'], dtype=float).copy()
    if min(width, height) <= 0 or k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Invalid camera intrinsics or image dimensions')
    k[0] *= 1600/width
    k[1] *= 900/height
    return k


def causal_scan(scans, timestamp):
    index = bisect.bisect_right([r['timestamp'] for r in scans], timestamp)-1
    if index < 0 or timestamp-scans[index]['timestamp'] > 100000:
        raise ValueError('No measured causal LiDAR within 100 ms')
    return scans[index]


def validate_camera_mapping(calibrations):
    if set(calibrations)!=set(CANONICAL_CAMERA_NAMES): raise ValueError('Canonical camera mapping mismatch')
    if len({c['source_camera'] for c in calibrations.values()})!=6: raise ValueError('Source camera mapping must be one-to-one')
    target=dict(zip(CANONICAL_CAMERA_NAMES,[55.,0.,-55.,-110.,180.,110.]))
    for name,camera in calibrations.items():
        rotation=np.asarray(camera['camera_to_ego_4x4'])[:3,:3]
        yaw=np.degrees(np.arctan2(rotation[1,2],rotation[0,2]))
        error=(yaw-target[name]+180)%360-180
        if abs(error)>60: raise ValueError('Camera semantic direction mismatch: '+name)
        # Wide source-rig differences remain visible, even when a canonical slot is usable.
        camera['canonical_direction_error_deg']=float(error)


@dataclass
class SceneInput:
    dataset_name: str
    scene_id: str
    scene_dir: Path
    timeline: list
    cameras: dict
    calibrations: dict
    ego_states: list
    actors: list
    map_context: dict
    metadata: dict
    ego_to_anchor: np.ndarray
    capabilities: dict
    lidar: object = None
    extra_sensors: dict = field(default_factory=dict)

    def model_frame(self, frame_id):
        i = frame_id
        state = dict(self.ego_states[i], **derive_kinematics(self.ego_states,i))
        reference = self.ego_to_anchor[i]
        # Explicit model coordinate convention at ego origin, not an invented physical sensor.
        virtual = np.array([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        state.update(lidar2ego_rotation=quaternion(virtual),lidar2ego_translation=[0.,0.,0.])
        calibration = {}
        timing = self.timeline[i]['cameras']
        for name in CANONICAL_CAMERA_NAMES:
            camera = self.calibrations[name]
            exposure = int(timing[name]['exposure_time_us'])
            history_pose = self.metadata['history_camera_poses'].get((i,name))
            if history_pose is not None:
                camera_anchor = np.asarray(history_pose)
                policy = 'exported_measured_history_camera_pose'
            else:
                camera_anchor = interpolate_poses([r['reference_time_us'] for r in self.timeline],
                    self.ego_to_anchor,[exposure])[0] @ np.asarray(camera['camera_to_ego_4x4'])
                policy = 'bounded_reference_pose_SO3_interpolation_at_exposure'
            transform = np.linalg.inv(reference) @ camera_anchor
            model_transform = np.linalg.inv(virtual) @ transform
            calibration[name] = dict(sensor2ego_translation=transform[:3,3].tolist(),sensor2ego_rotation=quaternion(transform),
                sensor2lidar_rotation=model_transform[:3,:3].tolist(),sensor2lidar_translation=model_transform[:3,3].tolist(),
                camera_intrinsics=canvas_intrinsics(camera).tolist())
        name,code = derive_command(self.ego_states,i)
        value = dict(camera_images={c:self.cameras[c][i] for c in CANONICAL_CAMERA_NAMES},camera_calibration=calibration,
            ego_state=state,history_ego_states=[dict(self.ego_states[j],**derive_kinematics(self.ego_states,j)) for j in range(max(0,i-3),i+1)],
            command=dict(name=name,code=code,source='derived_reference_route'))
        assert_no_evaluation_leakage(value)
        return dict(frame_id=i,timestamp=state['timestamp'],model_input=value,
            observation_available_timestamp=max(int(timing[c]['exposure_time_us']) for c in CANONICAL_CAMERA_NAMES),
            pose_policy=policy)


def load_scene_input(scene_dir, agent_only=False):
    scene = Path(scene_dir).resolve()
    def read(name): return json.loads((scene/name).read_text())
    context = read('scene_context.json')
    provenance_name = next((name for name in ('provenance.json','generation_provenance.json') if (scene/name).is_file()),None)
    if provenance_name is None: raise ValueError('Missing source metadata: '+str(scene))
    provenance = read(provenance_name)
    timeline = read('timeline.json')
    if timeline['unit'] != 'microseconds_relative_to_frame_0': raise ValueError('Unsupported time units')
    frames = timeline['frames']
    for row in read('future/rgb_manifest.json')['images']:
        frames[row['frame_index']].setdefault('cameras',{})[row['camera']]=row
    if [r['frame_index'] for r in frames] != list(range(90)): raise ValueError('Expected ordered 90-frame scene')
    if np.any(np.diff([r['reference_time_us'] for r in frames])<=0): raise ValueError('Nonmonotonic timeline')
    calibrations = {c['camera']:c for c in read('calibration/cameras.json')['cameras']}
    if set(calibrations)!=set(CANONICAL_CAMERA_NAMES): raise ValueError('Canonical camera mapping mismatch')
    validate_camera_mapping(calibrations)
    poses = read('calibration/reference_ego_poses.json')
    if poses['coordinate_frame'] not in ('history_frame_9_ego','history_last_ego'): raise ValueError('Unsupported ego frame')
    matrices = np.asarray(poses['ego_to_anchor'],float)
    if matrices.shape!=(90,4,4) or not np.isfinite(matrices).all() or not np.allclose(matrices[:,3,:],[0,0,0,1]): raise ValueError('Invalid ego poses')
    if not np.allclose(matrices[:,:3,:3].transpose(0,2,1)@matrices[:,:3,:3],np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(matrices[:,:3,:3]),1,atol=1e-5): raise ValueError('Invalid rigid ego rotation')
    for c in calibrations.values():
        transform=np.asarray(c['camera_to_ego_4x4'])
        if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]) or not np.isclose(np.linalg.det(transform[:3,:3]),1,atol=1e-5) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('Invalid right-handed camera-to-ego calibration')
        canvas_intrinsics(c)
    states = [dict(timestamp=int(f['reference_time_us']),position=p[:3,3].tolist(),rotation=quaternion(p),yaw=quat_to_yaw(quaternion(p))) for f,p in zip(frames,matrices)]
    cameras = {c:[str(scene/('history' if i<10 else 'future')/'rgb'/c/f'{i:06d}.jpg') for i in range(90)] for c in CANONICAL_CAMERA_NAMES}
    missing = [p for paths in cameras.values() for p in paths if not Path(p).is_file()]
    if missing: raise FileNotFoundError(missing[0])
    actors=[]
    for name in (() if agent_only else ('history/agents.csv','future/future_trajectories.csv')):
        with (scene/name).open() as handle: actors.extend(csv.DictReader(handle))
    map_available=not agent_only and all((scene/name).is_file() for name in ('map/map_meta.json','map/semantic_map.npz'))
    map_context=read('map/map_meta.json') if map_available else {'status':'unavailable','reason':'insufficient_drivable_area_map_support'}
    partial=False
    if map_available:
        with np.load(scene/'map/semantic_map.npz',allow_pickle=False) as data:
            partial = 'observed_surface' in data
    channels=map_context.get('available_channels',[])
    capabilities=dict(rgb=True,video=True,reference_video=True,camera=True,timeline=True,reference_route_proxy=True,camera_count=6,camera_names=list(CANONICAL_CAMERA_NAMES),calibration=True,trajectory=True,
        ego=True,actors=True,map=bool(channels),drivable_area='drivable_area' in channels and not partial,
        directed_lanes=False,partial_map=partial,lidar=False)
    history = read('calibration/history_camera_poses.json')['rows']
    metadata=dict(provenance=provenance,original_scene_id=provenance['source_scene'],source_root=provenance.get('source_root'),
        source_provenance_path=str(scene/provenance_name),anchor_ego_to_world=provenance['anchor_ego_to_world'],
        history_camera_poses={(r['frame_index'],r['camera']):r['camera_to_anchor_4x4'] for r in history},
        camera_mapping=[{k:c.get(k) for k in ('camera','source_camera','actual_yaw_deg','direction_error_deg','canonical_direction_error_deg')} for c in calibrations.values()],
        reference_frame='history_frame_9_ego; x forward, y left, z up; meters; CCW yaw radians',
        command_policy='oracle-conditioned reference route; actor labels/map never model input')
    return SceneInput(context['dataset'],scene.parent.parent.name+'__'+scene.name,scene,frames,cameras,calibrations,states,actors,map_context,metadata,matrices,capabilities)

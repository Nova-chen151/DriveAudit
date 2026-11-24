"""Explicit real-LiDAR ablation preparation; never imported by camera-only runs."""
from pathlib import Path
import numpy as np
from driveaudit_eval.agents.core import write_json
from driveaudit_eval.agents.real_agents import digest


def prepare_scene_lidar(scene, required, config, output):
    from . import load_lidar_for_scene, read_points
    lidar_index={}
    try:
        source_config=dict(config.get('native_map',{}),**config.get('extra_sensors',{}),timestamps_relative=True)
        lidar=load_lidar_for_scene(scene.scene_dir,[scene.timeline[i]['reference_time_us'] for i in required],source_config)
        converted={}
        for fid,scan in zip(required,lidar['records']):
            if scan is None: continue
            from driveaudit_eval.research.sensors import compensate_lidar
            state=scene.ego_states[fid]
            target_pose=dict(rotation=state['rotation'],translation=state['position'])
            compensated=scan.get('alignment_protocol')=='motion_compensated_v1'
            path=Path(scan['path']); key=(str(path),state['timestamp'] if compensated else None)
            canonical=compensate_lidar(scan,state['timestamp'],target_pose) if compensated else None
            if key not in converted:
                xyz=canonical.points_xyz if canonical else np.asarray(read_points(scan),dtype=np.float32)
                if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all(): raise ValueError('Invalid measured XYZ: '+str(path))
                suffix=f'_{state["timestamp"]}' if compensated else ''
                target=output/scene.scene_id/'lidar'/f'{scan["timestamp"]}{suffix}.npy'
                target.parent.mkdir(parents=True,exist_ok=True); np.save(target,xyz,allow_pickle=False)
                converted[key]=(str(target),digest(target),digest(path))
            target,target_hash,native_hash=converted[key]
            lidar_index[scene.cameras['CAM_FRONT'][fid]]=dict(lidar_path=target,point_format='xyz_npy',sha256=target_hash,
                native_path=str(path),native_sha256=native_hash,lidar_timestamp=scan['timestamp'],reference_timestamp=state['timestamp'],
                availability_timestamp=scan['availability_timestamp'],age_us=scan['age_us'],exists=True,
                lidar_calibration=canonical.calibration if canonical else scan['calibration'],
                lidar_ego_pose=target_pose if canonical else scan['ego_pose'],
                lidar_alignment=canonical.metadata if canonical else {},
                camera_ego_pose=target_pose,camera_timestamp_delta_us=0,lidar_token=scan['token'])
    except Exception as exc:
        # The declared LiDAR requirement controls availability; a sensor failure never disables RGB-only agents.
        reason='required_lidar_not_available' if isinstance(exc,FileNotFoundError) else 'invalid_sensor_data'
        lidar=dict(status='NOT_AVAILABLE',records=[None]*len(required),source_paths=[],
            missing=[dict(reference_timestamp=scene.timeline[i]['reference_time_us'],reason=reason,
                detail=f'{type(exc).__name__}: {exc}') for i in required])
    return lidar,lidar_index


def write_lidar_indexes(output,lidar_index):
    converted={k:dict(path=v['lidar_path'],point_format=v['point_format'],timestamp=v['lidar_timestamp'],sample_data_token=v['lidar_token'],
        reference_timestamp=v['reference_timestamp'],sha256=v['sha256'],calibration=v['lidar_calibration'],ego_pose=v['lidar_ego_pose'],reference_pose=v['camera_ego_pose'],lidar_alignment=v.get('lidar_alignment')) for k,v in lidar_index.items()}
    write_json(output/'goalflow_lidar.json',lidar_index)
    write_json(output/'diffusiondrivev2_lidar.json',dict(source_manifest_sha256=digest(output/'input_manifest.json'),frames=converted))

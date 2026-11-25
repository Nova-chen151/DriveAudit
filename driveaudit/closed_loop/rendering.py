"""History-only input construction and calibrated six-camera output contract."""
import json
from pathlib import Path
import numpy as np

def initial_inputs(cfg):
    from genbench_gt2.schema import SceneRef,SceneInputs
    from genbench_gt2.dataset import _calibration,_map,_history_rgb
    from genbench_gt2.constants import CAMERAS
    from PIL import Image
    scene=Path(cfg['scene']);root=scene.parents[2]
    ref=SceneRef(0,str(scene.relative_to(root)),'nuscenes',scene.name,root)
    order,calib=_calibration(scene)
    context=json.loads((scene/'scene_context.json').read_text())
    inputs=SceneInputs(ref,order,_history_rgb(ref),calib,np.repeat(np.eye(4)[None],90,axis=0),np.zeros((80,3)),{},_map(scene),context['generation_prompt'],np.arange(90)*500000)
    images=[Image.open(scene/'history/rgb'/cam/'000009.jpg').convert('RGB') for cam in CAMERAS]
    return inputs,images

def set_condition(inputs,frame,state):
    from genbench_gt2.schema import AgentBox
    from driveaudit.closed_loop.core import pose_matrix
    inputs.ego_to_anchor[frame]=pose_matrix(state['pose'])
    inputs.agents_by_frame[frame]=tuple(AgentBox(frame,a['id'],a['category'],np.array(a['xyz']),np.array(a['wlh']),a['yaw']) for a in state['actors'])

def publish(inputs,images,pose,output):
    from gt2_native import ego_to_native
    images=list(images)
    if len(images)!=6:raise ValueError('renderer must return six cameras')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    paths={};calibs={};rig=ego_to_native(inputs)
    for name,im in zip(inputs.camera_order,images):
        if im.size!=(400,224):raise ValueError('Explicit intrinsics-aware image conversion required before publish')
        p=folder/(name+'.png');im.save(p);paths[name]=str(p)
        c=inputs.calibrations[name];ext=rig@c.camera_to_ego
        calibs[name]=dict(sensor2lidar_rotation=ext[:3,:3].tolist(),sensor2lidar_translation=ext[:3,3].tolist(),camera_intrinsics=c.intrinsics.tolist())
    return dict(camera_images=paths,camera_calibration=calibs,render_pose=pose)

"""Uniform adapter for the six isolated model services."""
class AgentAdapter:
    def __init__(self, specification, sensor_mode='camera_only'):
        self.specification = specification
        self.sensor_mode = sensor_mode

    def load(self, context):
        from .registry import model_directories
        directory, _ = model_directories(self.specification[1])
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        self.context = context
        return self

    def predict(self, manifest, output):
        from .real_agents import infer
        return infer(self.specification, manifest, output, self.context)

    @staticmethod
    def normalize_output(agent, body):
        import numpy as np
        from .agent_input_adapter import parse_output
        from .agent_input_adapter import planning_to_unified_ego, select_vad_plan
        traj, mode, score, raw = parse_output(agent, body)
        if agent == 'vad':
            selected, _, mode, score = select_vad_plan(raw['raw_planning']['per_step_offsets'], raw['raw_planning']['command_one_hot'])
            np.testing.assert_allclose(selected, traj, atol=1e-5)
        if agent in ('uniad', 'vad'):
            traj = planning_to_unified_ego(traj).tolist()
        elif raw.get('coordinate_system') != 'ego_x_forward_y_left':
            raise ValueError('Native output coordinate mismatch')
        if np.asarray(traj).shape != (6, 2) or not np.isfinite(traj).all():
            raise ValueError('Invalid planning trajectory')
        return traj, mode, score, raw

    @property
    def requirements(self):
        from .registry import requirements_for
        return requirements_for(self.specification[0],self.sensor_mode)

    @property
    def temporal(self):
        return self.specification[0] in ('uniad','vad')

    @property
    def service_name(self):
        return 'sparsedrivev2' if self.specification[0].startswith('sparsedrivev2_') else self.specification[0]

    @property
    def health_path(self):
        return '/driver-clean/' if self.temporal else '/health'

    @staticmethod
    def prepare_input(scene, frame_id):
        return scene.model_frame(frame_id)

    def payload(self, frame, scene_id):
        import copy
        import numpy as np
        from .agent_input_adapter import build_http_request
        inp = frame['model_input']
        if self.service_name == 'drivor':
            from PIL import Image
            inp = copy.deepcopy(inp)
            from .core import quat_rotation
            for camera,cal in inp['camera_calibration'].items():
                # Native NavSim world_2_cam anchors are ego-frame, unlike UniAD/VAD planning axes.
                cal['sensor2lidar_rotation']=quat_rotation(cal['sensor2ego_rotation']).tolist()
                cal['sensor2lidar_translation']=cal['sensor2ego_translation']
                with Image.open(inp['camera_images'][camera]) as image: width,height=image.size
                intrinsic=np.asarray(cal['camera_intrinsics'],float)
                intrinsic[0]*=width/1600.; intrinsic[1]*=height/900.
                cal['camera_intrinsics']=intrinsic.tolist()
        return build_http_request(inp,scene_id,frame['frame_id']) if self.temporal else {'model_input':inp}

    def launch(self, root, output, port):
        import os
        from pathlib import Path
        from driveaudit import __file__ as public_package
        from driveaudit_eval import __file__ as evaluation_package
        from .registry import dependencies_root, model_directories, service_path
        name,directory,environment,extra=self.specification
        agent_root, assets = model_directories(directory)
        py=Path(os.environ.get('DRIVEAUDIT_CONDA_ENVS', Path.home()/'miniconda3/envs')).expanduser()/environment/'bin/python'
        pythonpath = [agent_root, Path(public_package).resolve().parents[1],
                      Path(evaluation_package).resolve().parents[1], root, dependencies_root(),
                      os.environ.get('NUPLAN_DEVKIT_ROOT', agent_root/'nuplan-devkit')]
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, pythonpath)),
            DRIVEAUDIT_AGENT_SOURCES=str(agent_root.parent), DRIVEAUDIT_AGENT_ASSETS=str(assets.parent),
            HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',UNIAD_DISABLE_SR='0',UNIAD_SR_OOM_FALLBACK='0')
        # Native extensions must use their own environment's compiler runtime.
        library_path = str(py.parent.parent/'lib')
        if env.get('LD_LIBRARY_PATH'):
            library_path += os.pathsep + env['LD_LIBRARY_PATH']
        env['LD_LIBRARY_PATH'] = library_path
        service=[str(py),str(service_path(name)),'--port',str(port),*extra]
        if self.temporal:
            service+=['--no-visualization','--out_dir',str(output/name/'preview'),
                      '--out',str(output/name/'native_result.pkl')]
        if self.requirements.lidar_required: service+=['--lidar-index',str(output/(self.service_name+'_lidar.json'))]
        if name in ('goalflow','diffusiondrivev2'):
            service+=['--sensor-mode','zero_lidar' if self.sensor_mode=='camera_only' else 'original_multimodal']
        if self.sensor_mode=='camera_only':
            guard=Path(__file__).with_name('sensor_guard.py')
            service=[service[0],str(guard),'--audit',str(output/name/'sensor_access_audit.jsonl'),'--service',*service[1:]]
        cwd=agent_root if self.temporal else root
        return service,cwd,env

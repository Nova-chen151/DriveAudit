"""GoalFlow inference on DriveAudit RGB and time-aligned nuScenes LiDAR."""
import json
from dataclasses import asdict
from pathlib import Path
import time

import cv2
import numpy as np

from driveaudit_eval.agents.null_lidar import make_null_lidar_tensor
from driveaudit_eval.agents.registry import dependencies_root
ROOT = dependencies_root() / 'DrivingAgents' / 'GoalFlow'


SENSOR_MODES = ('original_multimodal', 'zero_lidar', 'camera_only_native')


def validate_sensor_mode(mode):
    if mode not in SENSOR_MODES:
        raise ValueError('Unknown sensor mode: ' + str(mode))
    if mode == 'camera_only_native':
        raise ValueError('camera_only_native is not supported by the frozen checkpoint; use zero_lidar')
    return mode


def camera_panorama(images):
    """Scale official 1920x1080 crop bounds to the actual image resolution."""
    crops = []
    for i, rgb in enumerate(images):
        h, w = rgb.shape[:2]
        y = round(h * 28 / 1080)
        x = round(w * 416 / 1920) if i != 1 else 0
        crop = rgb[y:h-y, x:w-x]
        if crop.size == 0:
            raise ValueError('Empty camera crop')
        crops.append(crop)
    return cv2.resize(np.concatenate(crops, axis=1), (2048, 512))


def scoring_prefix(trajectory):
    points = np.asarray(trajectory, dtype=float)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite native [8,3] absolute ego trajectory')
    return points[:6, :2].tolist()


class GoalFlowAdapter:
    def __init__(self, data_root=ROOT / 'data', lidar_index=None, sensor_mode='zero_lidar'):
        self.sensor_mode = validate_sensor_mode(sensor_mode)
        import torch
        from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
        from navsim.agents.goalflow.goalflow_model_navi import GoalFlowNaviModel
        from navsim.agents.goalflow.goalflow_model_traj import GoalFlowTrajModel, TrajectoryHead
        from navsim.agents.goalflow.goalflow_features import GoalFlowFeatureBuilder
        import navsim.agents.goalflow.goalflow_model_traj as implementation
        if not Path(implementation.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError('GoalFlow imported from another checkout')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.config = GoalFlowConfig(tf_d_model=1024, training=False, topk=15,
            fusion=True, beta=0., cond_threshold=1., cond_weight=1., has_navi=False,
            has_student_navi=True, start=True, cur_sampling=True, use_nearest=True,
            train_scale=.1, test_scale=.1, theta=4.5, ep_score_weight=.2,
            ep_point_weight=.5, infer_steps=5, anchor_size=384,
            voc_path=str(data_root / 'cluster_points_8192_.npy'))
        self.models = {}
        self.tensor_count = {}
        checkpoint_prefixes = {}
        for name, cls, filename in (
            ('navi', GoalFlowNaviModel, 'goalflow_navi_epoch_99-step_132500.ckpt'),
            ('traj', GoalFlowTrajModel, 'goalflow_traj_epoch_54-step_18260.ckpt')):
            model = cls(self.config)
            state = torch.load(data_root / filename, map_location='cpu')['state_dict']
            prefix = next((p for p in ('agent._transfuser_model.', 'agent._goalflow_model.')
                           if state and all(k.startswith(p) for k in state)), None)
            if prefix is None:
                raise ValueError('Unexpected checkpoint namespace: ' + name)
            checkpoint_prefixes[name] = prefix
            if name == 'traj' and prefix + '_trajectory_head._mlp.0.weight' in state:
                # The release checkpoint retains this official auxiliary head;
                # the released flow forward does not call it. Load it, never discard its tensors.
                model._trajectory_head = TrajectoryHead(self.config.trajectory_sampling.num_poses,
                                                        self.config.tf_d_ffn, self.config.tf_d_model)
            model.load_state_dict({k[len(prefix):]:v for k,v in state.items()}, strict=True)
            self.models[name] = model.eval().cuda()
            self.tensor_count[name] = len(state)
            del state
        self.builder = GoalFlowFeatureBuilder(self.config)
        self.provenance = {'runtime_config': asdict(self.config),
                           'sensor_mode': self.sensor_mode,
                           'checkpoint_namespace_prefixes': checkpoint_prefixes,
                           'weights': json.loads((data_root / 'weights_manifest.json').read_text())}
        if self.sensor_mode == 'original_multimodal':
            from driveaudit_eval.research.sensors.fusion import load_index
            self.lidar_index,index_hash=load_index(lidar_index or data_root/'nuscenes3_lidar_index.json')
            self.provenance['lidar_index_sha256']=index_hash
        print('STRICT_LOAD_PASS', self.tensor_count, flush=True)

    def predict(self, model_input):
        import torch
        from PIL import Image
        from driveaudit_eval.agents.adapters.drivor import validate_model_input, command_one_hot
        validate_model_input(model_input)
        from driveaudit_eval.agents.provenance import seed_camera_request
        request_seed = seed_camera_request(model_input['ego_state']['timestamp'])
        if self.sensor_mode == 'original_multimodal':
            from driveaudit_eval.research.sensors.fusion import goalflow_feature
            lidar,entry=goalflow_feature(model_input,self.lidar_index,self.builder)
        images = []
        for channel in ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'):
            with Image.open(model_input['camera_images'][channel]) as im:
                images.append(np.array(im.convert('RGB')))
        camera = torch.from_numpy(camera_panorama(images).transpose(2,0,1).copy()).float() / 255
        if self.sensor_mode != 'original_multimodal':
            lidar = make_null_lidar_tensor(self.config)
        ego = model_input['ego_state']
        status = torch.tensor(command_one_hot(model_input['command']) + ego['velocity'][:2] + ego['acceleration'][:2])
        features = {k:v[None].float().cuda() for k,v in
                    {'camera_feature':camera, 'lidar_feature':lidar, 'status_feature':status}.items()}
        if not all(torch.isfinite(v).all() for v in features.values()):
            raise ValueError('Nonfinite input features')
        features['token'] = [entry['lidar_token'] if self.sensor_mode == 'original_multimodal'
                             else str(model_input['camera_images']['CAM_FRONT'])]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            scores = self.models['navi'](features, {})
            features['online_goal_scores'] = (scores['im_scores'], scores['dac_scores'])
            output = self.models['traj'](features, {})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        trajectory = output['trajectory'][0].cpu().numpy()
        result = {'planning_traj':scoring_prefix(trajectory), 'coordinate_system':'ego_x_forward_y_left',
                'request_seed':request_seed, 'native_trajectory':trajectory.tolist(), 'native_time_offsets':[.5*(i+1) for i in range(8)],
                'command':model_input['command'], 'strict_state_dict_tensors':self.tensor_count,
                'sensor_mode':self.sensor_mode, 'uses_lidar':self.sensor_mode == 'original_multimodal',
                'lidar_feature_shape':list(features['lidar_feature'].shape),
                'lidar_feature_nonzero':int(torch.count_nonzero(features['lidar_feature']).item()),
                'goal_score_source':'online_goalflow_navi',
                'model_provenance':self.provenance,
                'profile':{'model_inference_s':elapsed, 'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20}}
        if self.sensor_mode == 'original_multimodal':
            result.update(lidar_token=entry['lidar_token'], lidar_age_us=entry['age_us'])
        return result


"""Local GoalFlow service with the existing DriveAudit endpoint lifecycle."""
import argparse
from driveaudit_eval.agents.adapters.drivor import serve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11006)
    parser.add_argument('--lidar-index')
    parser.add_argument('--sensor-mode', choices=SENSOR_MODES, default='zero_lidar')
    args = parser.parse_args()
    serve(GoalFlowAdapter(lidar_index=args.lidar_index, sensor_mode=args.sensor_mode), args.port)

if __name__ == '__main__': main()

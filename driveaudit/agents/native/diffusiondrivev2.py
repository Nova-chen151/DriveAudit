"""DiffusionDriveV2 native model with DriveAudit's local HTTP and scoring contract."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from driveaudit.agents.native.drivor import command_one_hot, scoring_prefix, validate_model_input, serve

from driveaudit.agents.null_lidar import make_null_lidar_tensor
from driveaudit.agents.runtime import model_root, asset_root
ROOT = model_root('DiffusionDriveV2')
CAMERAS = {'cam_f0': 'CAM_FRONT', 'cam_l0': 'CAM_FRONT_LEFT', 'cam_r0': 'CAM_FRONT_RIGHT'}


SENSOR_MODES = ('original_multimodal', 'zero_lidar', 'camera_only_native')


def validate_sensor_mode(mode):
    if mode not in SENSOR_MODES:
        raise ValueError('Unknown sensor mode: ' + str(mode))
    if mode == 'camera_only_native':
        raise ValueError('camera_only_native is not supported by the frozen checkpoint; use zero_lidar')
    return mode


class DiffusionDriveV2Adapter:
    def __init__(self, checkpoint, lidar_index=None, sensor_mode='zero_lidar'):
        self.sensor_mode = validate_sensor_mode(sensor_mode)
        import torch
        from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
        from navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel
        from navsim.agents.diffusiondrivev2.transfuser_features import TransfuserFeatureBuilder
        import navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel as implementation
        if not Path(implementation.__file__).resolve().is_relative_to(ROOT): raise RuntimeError('Wrong NAVSIM checkout')
        torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if self.sensor_mode == 'original_multimodal':
            from driveaudit.agents.fusion import load_index
            self.index,_=load_index(lidar_index)
        self.config = TransfuserConfig()
        self.config.bkb_path = str(asset_root('DiffusionDriveV2') / self.config.bkb_path)
        self.config.plan_anchor_path = str(asset_root('DiffusionDriveV2') / self.config.plan_anchor_path)
        self.model = V2TransfuserModel(self.config)
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        prefix = 'agent._transfuser_model.'
        if not all(k.startswith(prefix) for k in state): raise ValueError('Unexpected checkpoint namespace')
        # Training registers this frozen vocabulary lazily; retain it for a strict full load.
        vocab = state[prefix + '_trajectory_head.vocab']
        if vocab.shape != (16384, 40, 3): raise ValueError('Unexpected training vocabulary shape')
        self.model._trajectory_head.vocab = torch.nn.Parameter(torch.empty_like(vocab), requires_grad=False)
        self.model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
        self.model.eval().cuda()
        self.builder = TransfuserFeatureBuilder(self.config)
        self.tensor_count = len(state)
        print('STRICT_LOAD_PASS', self.tensor_count, implementation.__file__, flush=True)

    def predict(self, model_input):
        import torch
        from PIL import Image
        from navsim.common.dataclasses import Camera, Cameras
        validate_model_input(model_input)
        from driveaudit.agents.runtime import seed_camera_request
        request_seed = seed_camera_request(model_input['ego_state']['timestamp'])
        state = model_input['ego_state']
        if self.sensor_mode == 'original_multimodal':
            from driveaudit.agents.fusion import diffusion_points
            xyz,entry,age=diffusion_points(model_input,self.index)
        cameras = {name: Camera() for name in Cameras.__dataclass_fields__}
        for name, channel in CAMERAS.items():
            with Image.open(model_input['camera_images'][channel]) as im:
                # The frozen comparison supplies 400x224 images. Restore the official crop canvas.
                rgb = np.array(im.convert('RGB').resize((1920, 1080), Image.Resampling.BILINEAR))
            cameras[name] = Camera(image=rgb)
        if self.sensor_mode == 'original_multimodal':
            from navsim.common.dataclasses import AgentInput, EgoStatus, Lidar
            ego = EgoStatus(ego_pose=np.zeros(3), ego_velocity=np.asarray(state['velocity'][:2]),
                            ego_acceleration=np.asarray(state['acceleration'][:2]), driving_command=np.asarray(command_one_hot(model_input['command'])))
            features = self.builder.compute_features(AgentInput(ego_statuses=[ego], cameras=[Cameras(**cameras)], lidars=[Lidar(lidar_pc=xyz.T)]))
        else:
            from types import SimpleNamespace
            features = dict(camera_feature=self.builder._get_camera_feature(SimpleNamespace(cameras=[Cameras(**cameras)])),
                            lidar_feature=make_null_lidar_tensor(self.config),
                            status_feature=torch.tensor(command_one_hot(model_input['command']) + state['velocity'][:2] + state['acceleration'][:2]))
        features = {k: v[None].float().cuda() for k, v in features.items()}
        if not all(torch.isfinite(v).all() for v in features.values()): raise ValueError('Nonfinite features')
        seed_timestamp = entry['reference_timestamp'] if self.sensor_mode == 'original_multimodal' else state['timestamp']
        seed = int(hashlib.sha256(str(seed_timestamp).encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(): output = self.model(features)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        trajectory = output['trajectory'][0].cpu().numpy()
        result = dict(planning_traj=scoring_prefix(trajectory), coordinate_system='ego_x_forward_y_left',
                    native_trajectory=trajectory.tolist(), native_time_offsets=[.5*(i+1) for i in range(8)],
                    command=model_input['command'], seed=seed, request_seed=seed, strict_state_dict_tensors=self.tensor_count,
                    sensor_mode=self.sensor_mode, uses_lidar=self.sensor_mode == 'original_multimodal',
                    lidar_feature_shape=list(features['lidar_feature'].shape),
                    lidar_feature_nonzero=int(torch.count_nonzero(features['lidar_feature']).item()),
                    image_preprocessing='frozen image -> bilinear 1920x1080 -> official three-camera crop/stitch/1024x256',
                    profile=dict(model_inference_s=elapsed, peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20))

        if self.sensor_mode == 'original_multimodal':
            result.update(lidar_source={k:entry[k] for k in ('path','sha256','sample_data_token','timestamp','reference_timestamp')},
                          lidar_age_s=age, lidar_points=len(xyz))
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=11006)
    p.add_argument('--checkpoint', type=Path, default=asset_root('DiffusionDriveV2')/'ckpts/diffusiondrivev2_sel.ckpt')
    p.add_argument('--lidar-index', type=Path, default=asset_root('DiffusionDriveV2')/'driveaudit_eval_artifacts/lidar_index.json')
    p.add_argument('--sensor-mode', choices=SENSOR_MODES, default='zero_lidar')
    args = p.parse_args()
    serve(DiffusionDriveV2Adapter(args.checkpoint, args.lidar_index, sensor_mode=args.sensor_mode), args.port)


if __name__ == '__main__': main()

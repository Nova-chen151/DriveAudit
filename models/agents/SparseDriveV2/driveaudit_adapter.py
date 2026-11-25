"""SparseDriveV2 inference on DriveAudit model_input, with the shared driver API."""
import argparse
from pathlib import Path
import time
import numpy as np
from driveaudit.agents.navsim_service import command_one_hot, scoring_prefix, validate_model_input, serve
from driveaudit.agents.runtime import quat_rotation

from driveaudit.agents.runtime import model_root, asset_root
ROOT = model_root('SparseDriveV2')
CAMERAS = {'cam_l0': 'CAM_FRONT_LEFT', 'cam_f0': 'CAM_FRONT', 'cam_r0': 'CAM_FRONT_RIGHT'}


def camera_parameters(calibration, path, size):
    """nuScenes camera -> ego; scale original 1600x900 intrinsics to the actual image."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size')
    rotation = quat_rotation(calibration['sensor2ego_rotation'])
    translation = np.asarray(calibration['sensor2ego_translation'], dtype=float)
    intrinsic = np.asarray(calibration['camera_intrinsics'], dtype=float).copy()
    if rotation.shape != (3, 3) or translation.shape != (3,) or intrinsic.shape != (3, 3):
        raise ValueError('Invalid calibration shape')
    if not all(np.isfinite(x).all() for x in (rotation, translation, intrinsic)):
        raise ValueError('Nonfinite calibration')
    intrinsic[0] *= width / 1600.
    intrinsic[1] *= height / 900.
    # The model's geometric anchors live in ego coordinates; its field is named lidar.
    return dict(image_path=path, sensor2lidar_rotation=rotation,
                sensor2lidar_translation=translation, intrinsics=intrinsic, distortion=np.zeros(5))


class SparseDriveAdapter:
    def __init__(self, checkpoint, version):
        if version != 'v2': raise ValueError('Only SparseDriveV2 v2 is supported')
        import torch
        from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
        from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel
        from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder
        import navsim.agents.sparsedrive.sparsedrive_model as implementation
        if not Path(implementation.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError('Wrong navsim checkout imported')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.config = SparseDriveConfig()
        self.config.dataset_version = version
        for key in ('bkb_path', 'path_anchor', 'velocity_anchor', 'trajectory_anchor'):
            setattr(self.config, key, str(asset_root('SparseDriveV2') / getattr(self.config, key)))
        self.model = SparseDriveModel(self.config)
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        prefix = 'agent._sparsedrive_model.'
        if not all(k.startswith(prefix) for k in state): raise ValueError('Unexpected checkpoint namespace')
        self.model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
        self.model.eval().cuda()
        self.builder = SparseDriveFeatureBuilder(self.config)
        self.tensor_count = len(state)
        self.version = version
        print('STRICT_LOAD_PASS', len(state), implementation.__file__, flush=True)

    def predict(self, model_input):
        import torch
        from PIL import Image
        from torch.utils.data._utils.collate import default_collate
        validate_model_input(model_input)
        from driveaudit.agents.runtime import seed_camera_request
        request_seed = seed_camera_request(model_input['ego_state']['timestamp'])
        cameras = {}
        sizes = []
        for name, channel in CAMERAS.items():
            path = model_input['camera_images'][channel]
            with Image.open(path) as im: size = im.size
            sizes.append(size)
            cameras[name] = camera_parameters(model_input['camera_calibration'][channel], path, size)
        if len(set(sizes)) != 1: raise ValueError('Camera image dimensions differ')
        self.config.W, self.config.H = sizes[0]
        ego = model_input['ego_state']
        features = {'camera_feature': [cameras], 'status_feature': torch.tensor(
            command_one_hot(model_input['command']) + ego['velocity'][:2] + ego['acceleration'][:2], dtype=torch.float32)}
        # Official inference augmentation and normalization; no targets or metric cache.
        features, _, _ = self.builder.pipeline(features, None, None, test_mode=True)
        features = default_collate([features])
        def to_cuda(value):
            if isinstance(value, dict): return {k: to_cuda(v) for k, v in value.items()}
            if isinstance(value, list): return [to_cuda(v) for v in value]
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all(): raise ValueError('Nonfinite input features')
                return value.float().cuda()
            return value
        features = to_cuda(features)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(): output, _ = self.model(features, None)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        trajectory = output['trajectory'][0].cpu().numpy()
        return {'planning_traj': scoring_prefix(trajectory), 'coordinate_system': 'ego_x_forward_y_left',
                'request_seed': request_seed, 'native_trajectory': trajectory.tolist(), 'native_time_offsets': [.5 * (i + 1) for i in range(8)],
                'command': model_input['command'], 'checkpoint_version': self.version,
                'strict_state_dict_tensors': self.tensor_count,
                'profile': {'model_inference_s': elapsed, 'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 2**20}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11004)
    parser.add_argument('--version', choices=['v2'], default='v2')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    names = {'v2': 'sparsedrive_navsimv2_90p3.ckpt'}
    adapter = SparseDriveAdapter(args.checkpoint or asset_root('SparseDriveV2') / 'ckpt' / names[args.version], args.version)

    serve(adapter, args.port, version=args.version)


if __name__ == '__main__': main()

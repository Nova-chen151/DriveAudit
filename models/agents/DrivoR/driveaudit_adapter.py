"""Local DrivoR service using DriveAudit's model_input and driver-* endpoints."""
import argparse
from pathlib import Path
import time
import numpy as np
from driveaudit.agents.navsim_service import command_one_hot, scoring_prefix, validate_model_input, serve

from driveaudit.agents.runtime import model_root, asset_root
ROOT = model_root('DrivoR')
CAMERAS = {'cam_f0': 'CAM_FRONT', 'cam_b0': 'CAM_BACK', 'cam_l0': 'CAM_FRONT_LEFT', 'cam_r0': 'CAM_FRONT_RIGHT'}


class DrivoRAdapter:
    def __init__(self, checkpoint, backbone=None):
        import torch
        from omegaconf import OmegaConf
        from navsim.agents.drivoR.drivor_model import DrivoRModel
        from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
        import navsim.agents.drivoR.drivor_model as implementation
        if not Path(implementation.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError('Imported original checkout instead of independent DrivingAgents copy')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
        self.config = OmegaConf.load(ROOT / 'navsim/planning/script/config/common/agent/drivoR.yaml').config
        self.config.image_backbone.model_weights = str(backbone or (asset_root('DrivoR') / self.config.image_backbone.model_weights))
        self.model = DrivoRModel(self.config)
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        prefix = 'agent._drivor_model.'
        if not all(k.startswith(prefix) for k in state): raise ValueError('Unexpected checkpoint namespace')
        self.model.load_state_dict({k[len(prefix):]: v for k, v in state.items()}, strict=True)
        self.model.eval().cuda()
        self.builder = DrivoRFeatureBuilder(self.config)
        self.tensor_count = len(state)
        print('STRICT_LOAD_PASS', self.tensor_count, 'source', implementation.__file__, flush=True)

    def predict(self, model_input):
        import torch
        from PIL import Image
        from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
        validate_model_input(model_input)
        from driveaudit.agents.runtime import seed_camera_request
        request_seed = seed_camera_request(model_input['ego_state']['timestamp'])
        cameras = {name: Camera() for name in Cameras.__dataclass_fields__}
        for name, channel in CAMERAS.items():
            calib = model_input['camera_calibration'][channel]
            rotation = np.asarray(calib['sensor2lidar_rotation'], dtype=float)
            translation = np.asarray(calib['sensor2lidar_translation'], dtype=float)
            intrinsic = np.asarray(calib['camera_intrinsics'], dtype=float)
            if rotation.shape != (3, 3) or translation.shape != (3,) or intrinsic.shape != (3, 3):
                raise ValueError('Invalid camera calibration shape')
            if not all(np.isfinite(v).all() for v in (rotation, translation, intrinsic)):
                raise ValueError('Nonfinite camera calibration')
            with Image.open(model_input['camera_images'][channel]) as im:
                rgb = np.array(im.convert('RGB'))
            cameras[name] = Camera(image=rgb, sensor2lidar_rotation=rotation, sensor2lidar_translation=translation, intrinsics=intrinsic)
        state = model_input['ego_state']
        ego = EgoStatus(ego_pose=np.zeros(3), ego_velocity=np.asarray(state['velocity'][:2]),
                        ego_acceleration=np.asarray(state['acceleration'][:2]), driving_command=np.asarray(command_one_hot(model_input['command'])))
        features = self.builder.compute_features(AgentInput(ego_statuses=[ego], cameras=[Cameras(**cameras)], lidars=[]))
        features = {k: v[None].float().cuda() for k, v in features.items()}
        if not all(torch.isfinite(v).all() for v in features.values()): raise ValueError('Nonfinite features')
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(): output = self.model(features)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        trajectory = output['trajectory'][0].cpu().numpy()
        proposals = output['proposals'][0].cpu().numpy()
        scores = output['pdm_score'][0].cpu().numpy()
        if not np.isfinite(proposals).all() or not np.isfinite(scores).all(): raise ValueError('Nonfinite model output')
        index = int(np.argmax(scores))
        np.testing.assert_allclose(trajectory, proposals[index])
        return {'planning_traj': scoring_prefix(trajectory), 'coordinate_system': 'ego_x_forward_y_left',
                'request_seed': request_seed, 'native_trajectory': trajectory.tolist(), 'native_time_offsets': [.5 * (i + 1) for i in range(8)],
                'proposals': proposals.tolist(), 'predicted_log_scores': scores.tolist(),
                'mode_index': index, 'mode_score': float(scores[index]), 'command': model_input['command'],
                'strict_state_dict_tensors': self.tensor_count,
                'profile': {'model_inference_s': elapsed, 'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 2**20}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11003)
    parser.add_argument('--checkpoint', type=Path, default=asset_root('DrivoR') / 'weights/drivor_Nav1_25epochs.pth')
    args = parser.parse_args()
    adapter = DrivoRAdapter(args.checkpoint)

    serve(adapter, args.port)


if __name__ == '__main__': main()

"""Local DrivoR service using DriveAudit's model_input and driver-* endpoints."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time
import numpy as np
from driveaudit.agents.runtime import assert_no_evaluation_leakage

from driveaudit.agents.runtime import model_root, asset_root
ROOT = model_root('DrivoR')
CAMERAS = {'cam_f0': 'CAM_FRONT', 'cam_b0': 'CAM_BACK', 'cam_l0': 'CAM_FRONT_LEFT', 'cam_r0': 'CAM_FRONT_RIGHT'}


def command_one_hot(command):
    names = {0: 'RIGHT', 1: 'LEFT', 2: 'STRAIGHT'}
    code = command['code']
    if isinstance(code, bool) or code not in names or command['name'] != names[code]:
        raise ValueError('DriveAudit command code/name mismatch')
    return {'LEFT': [1, 0, 0, 0], 'STRAIGHT': [0, 1, 0, 0], 'RIGHT': [0, 0, 1, 0]}[names[code]]


def scoring_prefix(trajectory):
    points = np.asarray(trajectory, dtype=float)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite native [8,3] absolute ego trajectory')
    return points[:6, :2].tolist()


def validate_model_input(value):
    allowed = {'camera_images', 'camera_calibration', 'ego_state', 'history_ego_states', 'command'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Only model_input fields are accepted')
    assert_no_evaluation_leakage(value)
    for name in allowed - {'history_ego_states'}:
        if name not in value: raise ValueError('Missing model input: ' + name)
    command_one_hot(value['command'])
    for name in ('velocity', 'acceleration'):
        vec = np.asarray(value['ego_state'][name], dtype=float)
        if vec.shape != (3,) or not np.isfinite(vec).all(): raise ValueError('Invalid ego ' + name)


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


def serve(adapter, port):
    class Handler(BaseHTTPRequestHandler):
        result = None

        def respond(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/health': self.respond(200, {'ready': True, 'tensors': adapter.tensor_count})
            elif self.path == '/driver-clean/': Handler.result = None; self.respond(200, {'reset': True})
            elif self.path == '/driver-get/': self.respond(200 if Handler.result is not None else 409, Handler.result)
            else: self.respond(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if self.path != '/driver-api/': self.respond(404, {'error': 'unknown endpoint'}); return
            Handler.result = None
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024 * 1024: raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict) or set(value) != {'model_input'}: raise ValueError('Expected model_input envelope')
                Handler.result = adapter.predict(value['model_input'])
                self.respond(201, {'ready': True})
            except (ValueError, KeyError, OSError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception as exc:
                self.respond(500, {'error': type(exc).__name__ + ': ' + str(exc)})

    # ponytail: serial requests for one GPU; add a bounded worker queue if concurrent clients are needed.
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__': main()

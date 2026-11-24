"""Measured model identity and camera-only evidence for paired comparisons."""
import hashlib
import json
from pathlib import Path

from driveaudit_eval.agents.registry import AGENTS, dependencies_root
from driveaudit_eval.metrics.testing_effectiveness import content_hash

MODEL_FILES = {
    'uniad': ('ckpts/uniad_base_e2e.pth', 'ckpts/CAMixerSRx4_DF.pth'),
    'vad': ('ckpts/VAD_base.pth', 'ckpts/CAMixerSRx4_DF.pth'),
    'drivor': ('weights/drivor_Nav1_25epochs.pth', 'weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors'),
    'sparsedrivev2_v2': ('ckpt/sparsedrive_navsimv2_90p3.ckpt', 'ckpt/resnet34.bin',
                         'ckpt/kmeans/path_1024.npy', 'ckpt/kmeans/velocity_256.npy', 'ckpt/kmeans/trajectory_1024_256.npz'),
    'diffusiondrivev2': ('ckpts/diffusiondrivev2_sel.ckpt', 'ckpts/resnet34.a1_in1k/pytorch_model.bin', 'kmeans_navsim_traj_20.npy'),
    'goalflow': ('data/goalflow_navi_epoch_99-step_132500.ckpt', 'data/goalflow_traj_epoch_54-step_18260.ckpt',
                 'data/cluster_points_8192_.npy'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def timestamp_seed(timestamp):
    return int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)


def seed_camera_request(timestamp):
    """Same timestamp => same RNG in both RGB domains, independent of scene order."""
    import random
    import numpy as np
    import torch
    seed = timestamp_seed(timestamp)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def model_identity(agent):
    """Hash released weights, anchors, native configuration/source and local adapters.

    Formal launch exposes no checkpoint/config override; these are its actual
    defaults. Capture before and after each service run, never infer identity
    from a current checkpoint for an unrecorded historical prediction.
    """
    directory = next(item[1] for item in AGENTS if item[0] == agent)
    native = dependencies_root()/'DrivingAgents'/directory
    package = Path(__file__).resolve().parent
    resources = {name: digest(native/name) for name in MODEL_FILES[agent]}
    sources = {}
    for folder in ('projects', 'CAMixerSR', 'navsim', 'vovnet'):
        for path in sorted((native/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.so'):
                sources[str(path.relative_to(native))] = digest(path)
    adapters = {str(path.relative_to(package)): digest(path) for path in sorted(package.rglob('*.py'))
                if 'score_audit' not in path.parts}
    return dict(checkpoint_hash=content_hash(resources), agent_config_hash=content_hash(dict(
        native_sources=sources, adapters=adapters)), resources=resources,
        native_sources=sources, adapter_sources=adapters, agent=agent)


def protocol_identity(agent):
    from driveaudit_eval.agents.registry import requirements_for
    package = Path(__file__).resolve().parent
    protocol = dict(version='qtes_camera_only_v1', sensor_protocol='camera_only', real_lidar_allowed=False,
                    requirements=requirements_for(agent).as_dict(),
                    lidar_mode='zero' if agent in ('goalflow', 'diffusiondrivev2') else 'not_requested',
                    seed_policy='scene_frame_v1' if agent in ('uniad', 'vad') else 'timestamp_sha256_v1',
                    source_hashes={name:digest(package/name) for name in
                        ('sensor_guard.py','null_lidar.py','agent_input_adapter.py','adapters/camera_only_temporal.py','provenance.py')})
    return content_hash(protocol)


def request_seeds(agent, sid, frames):
    from driveaudit_eval.agents.adapters.camera_only_temporal import stable_request_seed
    return [stable_request_seed(sid, frame['frame_id']) if agent in ('uniad','vad')
            else timestamp_seed(frame['timestamp']) for frame in frames]


def audit_inference(directory, agent, scene, expected_identity):
    """Bind actual predictions, raw forwards, guards and RNG to the input manifest."""
    from driveaudit_eval.agents.registry import requirements_for
    from driveaudit_eval.agents.agent_input_adapter import assert_no_evaluation_leakage
    import numpy as np
    path = Path(directory)/agent
    def read(name):
        return json.loads((path/name).read_text())
    def lines(name):
        return [json.loads(s) for s in (path/name).read_text().splitlines() if s.strip()]
    def require(condition, reason):
        if not condition:
            raise ValueError(agent + ':' + reason)
    identity = read('model_identity.json')
    require(identity == expected_identity, 'model_identity_mismatch')
    status = read('inference_status.json')
    require(status['status'] == 'COMPLETED' and status.get('sensor_protocol') == 'camera_only', 'inference_not_completed')
    require(status.get('model_identity_hash') == content_hash(identity), 'identity_not_bound_to_inference')
    for filename in ('predictions.jsonl', 'raw_outputs.jsonl', 'agent_input_manifest.json', 'sensor_access_audit.jsonl'):
        require(status.get('evidence_sha256', {}).get(filename) == digest(path/filename), 'evidence_hash_mismatch:' + filename)
    source = json.loads((Path(directory)/'input_manifest.json').read_text())
    require(source['sensor_protocol'] == 'camera_only', 'camera_only_required')
    sid = scene['scene_id']
    expected = [frame for frame in source['scenes'] if frame['scene_id'] == sid]
    require(len(expected) == 1 and expected[0]['frames'] == scene['frames'], 'scene_input_mismatch')
    scored = [w['frame_id'] for w in source['common_windows'] if w['scene_id'] == sid]
    frames = [f for f in scene['frames'] if agent in ('uniad','vad') or f['frame_id'] in scored]
    frame_ids = [f['frame_id'] for f in frames]
    seeds = request_seeds(agent, sid, frames)
    manifest = read('agent_input_manifest.json')
    require(manifest['protocol'] == 'camera_only' and manifest['sensor_leakage_detected'] is False, 'sensor_leakage')
    guard = lines('sensor_access_audit.jsonl')
    require(any(r.get('event') == 'guard_installed' and r.get('resolved_paths') is True for r in guard)
            and not any(r.get('event') == 'blocked' for r in guard), 'sensor_guard_missing_or_blocked')
    decisions = [r for r in manifest['decisions'] if r['scene'] == sid]
    raw = [r for r in lines('raw_outputs.jsonl') if r['scene_id'] == sid]
    predictions = [r for r in lines('predictions.jsonl') if r['scene_id'] == sid]
    require([r['decision_frame'] for r in decisions] == frame_ids
            and [r['frame_id'] for r in raw] == frame_ids
            and [r['frame_id'] for r in predictions] == frame_ids, 'forward_support_mismatch')
    for frame, decision, output, prediction, seed in zip(frames, decisions, raw, predictions, seeds):
        assert_no_evaluation_leakage(frame['model_input'])
        require(decision['rgb_used'] is True and decision['camera_count'] == requirements_for(agent).required_camera_count,
                'rgb_protocol_mismatch')
        require(all(decision.get(k) is False for k in ('lidar_used','real_lidar_accessed','real_lidar_used','gt_future_used','gt_boxes_used')),
                'sensor_leakage_or_missing_evidence')
        require(decision['scored_decision'] == (frame['frame_id'] in scored), 'scored_decision_mismatch')
        value = output['output']
        require(value.get('request_seed', value.get('seed')) == seed, 'request_seed_mismatch')
        if agent in ('goalflow','diffusiondrivev2'):
            require(value.get('sensor_mode') == 'zero_lidar' and value.get('uses_lidar') is False
                    and value.get('lidar_feature_nonzero') == 0 and value.get('lidar_feature_shape') == [1,1,256,256],
                    'zero_lidar_mismatch')
        plan = np.asarray(prediction['planning_traj'])
        require(prediction['timestamp'] == frame['timestamp'] and prediction['inference_success'] is True
                and plan.shape == (6,2) and np.isfinite(plan).all(), 'invalid_prediction')
    return dict(real_lidar_used=False, sensor_leakage=False, inference_valid=True,
                sensor_protocol='camera_only', lidar_mode='zero' if agent in ('goalflow','diffusiondrivev2') else 'not_requested',
                inference_evidence_hash=content_hash(status['evidence_sha256']))

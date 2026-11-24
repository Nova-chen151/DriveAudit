"""CATK provenance and standard GenBench publication for MagicDriveDiT."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARED_REPO = REPO.parent / 'gen-benchmark'
DATASETS = {
    'av2_challenger_v1': 'av2',
    'nuplan_challenger_v1': 'nuplan',
    'nuscenes300_challenger_v1': 'nuscenes',
    'pandaset_challenger_v1': 'pandaset',
}
CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT')
CONFIG = REPO / 'rd/configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py'
CHECKPOINT = REPO / 'rd/ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt'


def standard_api():
    # Put the shared package before this script's similarly named module.
    if str(SHARED_REPO) not in sys.path[:1]:
        sys.path.insert(0, str(SHARED_REPO))
    from genbench_gt2 import output
    from genbench_gt2.schema import SceneRef
    return output, SceneRef


def standard_scene_ref(output, relative):
    _, SceneRef = standard_api()
    output, relative = Path(output).resolve(), Path(relative)
    if output.name != 'magicdrivev2' or output.parent.name != 'predictions':
        raise ValueError('Standard output must end in predictions/magicdrivedit')
    if len(relative.parts) != 3 or relative.parts[1] != 'scenes' or relative.parts[0] not in DATASETS:
        raise ValueError(f'Invalid scene ID: {relative}')
    return SceneRef(index=0, relative_path=relative.as_posix(), dataset=DATASETS[relative.parts[0]],
                    scene_id=relative.name, root=output.parent.parent)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return 'sha256:' + digest.hexdigest()


def digest_json(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def conditioning_evidence(scene, trajectory_scene):
    scene = Path(scene).resolve()
    trajectory_scene = Path(trajectory_scene).resolve() if trajectory_scene is not None else scene
    names = ['history/ego.csv', 'history/agents.csv', 'future/ego_trajectory.csv', 'timeline.json',
             'calibration/cameras.json', 'calibration/reference_ego_poses.json',
             'map/map_meta.json', 'map/semantic_map.npz']
    if (scene / 'scene_context.json').exists():
        names.append('scene_context.json')
    names.extend(f'history/rgb/{camera}/{frame:06d}.jpg' for camera in CAMERAS for frame in range(1, 10))
    sources = {str(scene / name): sha256_file(scene / name) for name in names}
    trajectory = trajectory_scene / 'future/future_trajectories.csv'
    sources[str(trajectory)] = sha256_file(trajectory)
    evidence = dict(context_source=str(scene), future_agent_trajectory_source=str(trajectory),
                    source_hashes=sources, location_prompt_override=os.getenv('CHALLENGE_LOCATION_PROMPT', ''))
    return dict(evidence, conditioning_digest=digest_json(evidence))


def checkpoint_files():
    """Check complete local weight shards/tokenizer before any expensive inference."""
    vae = REPO / 'rd/pretrained/CogVideoX-2b/vae'
    text = REPO / 'rd/pretrained/t5-v1_1-xxl'
    index_path = text / 'model.safetensors.index.json'
    fixed = [CHECKPOINT, CONFIG, vae / 'config.json', vae / 'diffusion_pytorch_model.safetensors',
             text / 'config.json', text / 'tokenizer_config.json', text / 'spiece.model', index_path]
    if not index_path.is_file():
        raise ValueError(f'Missing checkpoint index: {index_path}')
    shards = sorted(set(json.loads(index_path.read_text())['weight_map'].values()))
    files = fixed + [text / shard for shard in shards]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/empty checkpoint resource: {path}')
    return files


def inference_settings(gpus_per_worker=2, temporal_mode='legacy-ar17'):
    if gpus_per_worker not in (2, 4, 8) or temporal_mode not in ('legacy-ar17', 'full89'):
        raise ValueError('Expected SP size 2/4/8 and temporal mode legacy-ar17/full89')
    return dict(sampling_steps=30, scheduler='rflow', cfg_scale=2.0,
                image_size=[400, 224], sequence_parallel_size=gpus_per_worker, cpu_offload=True,
                force_pad_h_for_sp_size=4, dtype='bf16', autoregressive_reference=temporal_mode=='legacy-ar17',
                camera_embedding_intrinsics_policy={
                    'native_lidar_rig': 'native_intrinsics (nuScenes training convention)',
                    'ego_renderer_rig': 'challenge_intrinsics at 400x224 (cross-dataset compatibility scaling)',
                    'source_calibration_modified': False})


def temporal_strategy(temporal_mode='legacy-ar17'):
    inference_settings(2, temporal_mode)
    if temporal_mode == 'legacy-ar17':
        return dict(window_frames=17, overlap_frames=9, stride_frames=8,
                    generated_frames=list(range(10, 90)), nominal_contract_fps=12)
    return dict(mode='full89', window_frames=89, overlap_frames=0, stride_frames=80,
                input_frames=list(range(1,90)), history_frames=list(range(1,10)),
                generated_frames=list(range(10,90)), nominal_contract_fps=12)


def validate_full_audit(path, gpus_per_worker, fps):
    """Bind a full89 publication to actual completed native scheduler/decode inputs."""
    path=Path(path)
    if not path.is_file():raise ValueError('Missing native full89 conditioning audit: '+str(path))
    row=json.loads(path.read_text())
    expected=dict(rgb_frames=89,history_rgb_frames=9,latent_frames=23,reference_latent_frames=3,
        frozen_latents_per_sample=[3],sequence_parallel_size=gpus_per_worker,sample_seed=1024,
        box_latent_seed=1024,box_latent_mode='random',scheduler='rflow',sampling_steps=30,cfg_scale=2.0,
        decoded_shape=[1,6,3,89,224,400])
    mismatch=[k for k,v in expected.items() if row.get(k)!=v]
    flags=('completed','valid','all_finite','required_conditions_present','temporal_conditioning_aligned',
           'reference_prefix_and_mask_aligned','diffusion_output_finite','decoded_finite')
    mismatch += [k for k in flags if row.get(k) is not True]
    import torch
    # prepare() creates float32 FPS; the native model moves conditions to BF16.
    # Verify that exact representable value rather than relaxing the raw FPS.
    model_fps=torch.tensor(fps,dtype=torch.float32).to(torch.bfloat16).float().item()
    actual_fps=row.get('fps')
    if actual_fps != [model_fps] or row.get('conditions',{}).get('fps',{}).get('dtype') != 'torch.bfloat16':
        mismatch.append('fps')
    if Path(row.get('checkpoint','')).resolve()!=CHECKPOINT.resolve():mismatch.append('checkpoint')
    if mismatch:raise ValueError('Native full89 conditioning audit mismatch: '+','.join(mismatch))
    return dict(path=str(path.resolve()),sha256=sha256_file(path),boundary='completed_native_scheduler_and_decode')


def runtime_identity(gpus_per_worker=2, temporal_mode='legacy-ar17'):
    api, _ = standard_api()
    code = [Path(__file__), REPO / 'challenge_generation/genbench_gt2.py',
            REPO / 'challenge_generation/infer.py', REPO / 'challenge_generation/v2_contract.py',
            REPO / 'challenge_generation/conditioning_timing.py', REPO / 'challenge_generation/model_conditioning.py',
            REPO / 'rd/scripts/adv_inference_magicdrive.py', CONFIG]
    settings = inference_settings(gpus_per_worker, temporal_mode)
    settings['code_hashes'] = {str(path): sha256_file(path) for path in code}
    return dict(checkpoint=api.checkpoint_identity(checkpoint_files()), seed=1024, inference_settings=settings)


def metadata(scene, trajectory_scene, identity):
    if __package__:
        from .conditioning_timing import conditioning_timing
    else:
        from conditioning_timing import conditioning_timing
    map_meta = json.loads((scene / 'map/map_meta.json').read_text())
    strategy = temporal_strategy('legacy-ar17' if identity['inference_settings']['autoregressive_reference'] else 'full89')
    strategy.update(conditioning_timing(scene))
    return dict(identity, **conditioning_evidence(scene, trajectory_scene),
                history_frames_used=list(range(1, 10)), repeated_reference_slots=[],
                used_future_ego_trajectory=True, used_future_agent_trajectories=True,
                future_rgb_conditioning=False, trajectory_conditioning='catk' if trajectory_scene else 'ground_truth',
                ego_pose_conditioning='public participant calibration/reference_ego_poses.json (native full SE3)',
                available_map_channels=map_meta.get('available_channels', []),
                unavailable_map_channels=map_meta.get('unavailable_channels', []),
                missing_map_policy='zero placeholders; checkpoint has no unknown-channel mask',
                temporal_strategy=strategy,
                slurm_job_id=os.getenv('SLURM_JOB_ID'))


def can_resume(output, relative, expected):
    api, _ = standard_api()
    scene = standard_scene_ref(output, relative)
    final = Path(output) / scene.dataset / 'scenes' / scene.scene_id
    if not final.exists():
        return False
    result = api.validate_completed_scene(final, 'magicdrivev2', scene)
    if not result.valid:
        raise ValueError(f'Existing output is incomplete/incompatible; preserved at {final}: {result.reason}')
    actual = json.loads((final / 'status.json').read_text())
    for field in ('conditioning_digest', 'checkpoint', 'seed', 'inference_settings', 'temporal_strategy'):
        if actual.get(field) != expected.get(field):
            raise ValueError(f'Existing output {field} mismatch; preserved at {final}')
    return True


def publish_pixels(pixels, output, relative, evidence):
    api, _ = standard_api()
    scene = standard_scene_ref(output, relative)
    with api.ScenePublisher(Path(output).parent, 'magicdrivev2', scene,
                            os.getenv('SLURM_JOB_ID', 'local'), str(os.getpid())) as publisher:
        for c, camera in enumerate(CAMERAS):
            for t in range(80):
                publisher.write_frame(camera, t + 10, pixels[c, t])
        return publisher.publish(evidence)

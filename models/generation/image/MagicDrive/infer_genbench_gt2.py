"""Direct released MagicDrive-t inference on canonical GT2 controls."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import json
import sys
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genbench_gt2.constants import CAMERAS
from genbench_gt2.dataset import load_generation_set, load_scene_inputs
from genbench_gt2.geometry import box_corners, transform_agent_to_target, project_points
from genbench_gt2.raster import class_index, warp_semantic_map
from genbench_gt2.selection import select_scenes, shard_scenes
from genbench_gt2.temporal import magicdrive_plan, choose_magicdrive_splice
from genbench_gt2.output import ScenePublisher, checkpoint_identity, validate_completed_scene, write_failure

CONTROL_ADAPTER_VERSION = 'magicdrive-native-lidar-v2'


def model_lidar_to_ego(inputs):
    """Use nuScenes' calibrated LIDAR_TOP, or an explicit virtual training rig.

    Legacy camera_to_renderer includes a different exposure/anchor transform;
    only the renderer's rigid sensor calibration is reusable here.
    """
    asset_root = inputs.scene.asset_root or inputs.scene.root
    path = asset_root / inputs.scene.relative_path / 'calibration/cameras.json'
    renderer = json.loads(path.read_text()).get('renderer', {})
    result = np.eye(4)
    if renderer.get('coordinate_frame') == 'source_lidar_top_at_history_last_frame':
        q = np.asarray(renderer['sensor2ego_rotation_wxyz'], dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError('Invalid source LiDAR quaternion')
        w, x, y, z = q / np.linalg.norm(q)
        result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        result[:3, 3] = renderer['sensor2ego_translation_m']
    else:
        # No native nuScenes LiDAR in AV2/nuPlan/PandaSet: right/forward/up,
        # with the fixed reference nuScenes sensor offset, in metres.
        result[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        result[:3, 3] = [.985793, 0., 1.84019]
    if not np.isfinite(result).all():
        raise ValueError('Invalid source LiDAR calibration')
    return result


def control_metadata(inputs):
    return {'version': CONTROL_ADAPTER_VERSION,
            'camera_intrinsic_canvas_wh': [1600, 900],
            'lidar_to_ego': model_lidar_to_ego(inputs).tolist(),
            'box_corner_convention': 'nuscenes_wlh_clockwise_minus_yaw_minus_pi_over_2'}


def magicdrive_box_corners(box, ego_to_lidar):
    # nuScenes converter: (w,l,h,-yaw-pi/2), mmdet clockwise rotation.
    # Merely using mmdet's binary corner order on (l,w,h) changes token order.
    corners = box_corners(box)[[4, 5, 1, 0, 6, 7, 3, 2]]
    return corners @ ego_to_lidar[:3, :3].T + ego_to_lidar[:3, 3]

@dataclass(frozen=True)
class MagicDriveBatch:
    bev_map: object
    camera_param: object
    boxes: object
    classes: object
    masks: object
    prompts: list
    dropped_categories: dict


def build_magicdrive_window(inputs, frames):
    import torch
    maps, visible = [], []
    dropped = {}
    lidar_to_ego = model_lidar_to_ego(inputs)
    ego_to_lidar = np.linalg.inv(lidar_to_ego)
    # Training passes pre-ImageAug3D K in a 1600x900 sensor canvas. GT2 K
    # describes a direct 400x224 resize. Restore that convention for every rig.
    sensor_scale = np.diag([1600 / 400, 900 / 224, 1.])
    calibration = np.stack([np.concatenate([sensor_scale @ inputs.calibrations[c].intrinsics,
                         (ego_to_lidar @ inputs.calibrations[c].camera_to_ego)[:3, :4]], axis=-1) for c in CAMERAS])
    for frame in frames:
        m = inputs.map_condition
        maps.append(warp_semantic_map(m.mask, m.extent_m, m.resolution_m,
                    inputs.ego_to_anchor[frame] @ lidar_to_ego, (-50, 50, -50, 50), .5,
                    m.available_channels, (200, 200))[0])
        views = [[] for _ in CAMERAS]
        for agent in inputs.agents_by_frame.get(frame, ()):
            label = class_index(agent.category)
            if label is None:
                dropped[agent.category] = dropped.get(agent.category, 0) + 1
                continue
            target_box = transform_agent_to_target(agent, inputs.ego_to_anchor[frame])
            corners_ego = box_corners(target_box)
            corners = magicdrive_box_corners(target_box, ego_to_lidar)
            for vi, camera in enumerate(CAMERAS):
                if project_points(corners_ego, inputs.calibrations[camera])[1].any():
                    views[vi].append((corners, label))
        visible.append(views)
    count = max(1, max(len(view) for views in visible for view in views))
    boxes = np.zeros((len(frames), 6, count, 8, 3), np.float32)
    classes = np.full((len(frames), 6, count), -1, np.int64)
    masks = np.zeros(classes.shape, bool)
    for fi, views in enumerate(visible):
        for vi, entries in enumerate(views):
            for bi, (corners, label) in enumerate(entries):
                boxes[fi, vi, bi], classes[fi, vi, bi], masks[fi, vi, bi] = corners, label, True
    return MagicDriveBatch(torch.tensor(np.stack(maps), dtype=torch.float32),
        torch.tensor(np.repeat(calibration[None], len(frames), axis=0), dtype=torch.float32),
        torch.from_numpy(boxes), torch.from_numpy(classes), torch.from_numpy(masks),
        [inputs.prompt] * len(frames), dropped)


def generate_scene(pipe, cfg, inputs, seed, weight_dtype=None):
    import torch
    weight_dtype = weight_dtype or torch.float16
    windows = []
    dropped = {}
    for frame in range(10, 90):
        for agent in inputs.agents_by_frame.get(frame, ()):
            if class_index(agent.category) is None:
                dropped[agent.category] = dropped.get(agent.category, 0) + 1
    for wi, frames in enumerate(magicdrive_plan().condition_windows):
        batch = build_magicdrive_window(inputs, frames)
        print('Generating scene {} window {}'.format(inputs.scene.index, wi), flush=True)
        output = pipe(prompt=batch.prompts, image=batch.bev_map.to('cuda', dtype=weight_dtype),
            camera_param=batch.camera_param.to('cuda', dtype=weight_dtype), height=224, width=400,
            generator=torch.Generator(device='cuda').manual_seed(seed + inputs.scene.index * 1009 + wi),
            bev_controlnet_kwargs={'bboxes_3d_data': {
                'bboxes': batch.boxes.to('cuda'), 'classes': batch.classes.to('cuda'),
                'masks': batch.masks.to('cuda'), 'cam_params': batch.camera_param.to('cuda', dtype=weight_dtype)}},
            guidance_scale=2, num_inference_steps=20, eta=0.0, controlnet_conditioning_scale=1.0,
            guess_mode=False, use_zero_map_as_unconditional=False, init_noise='rand_all',
            view_order=list(CAMERAS), double_cfg_inference=True, decode_bs=1, keyframe_rate=6)
        assert len(output.images) == 61 and all(len(row) == 6 for row in output.images)
        windows.append({c: [row[vi] for row in output.images] for vi, c in enumerate(CAMERAS)})
    images, decision = merge_magicdrive_windows(*windows)
    return images, decision, dropped


def merge_magicdrive_windows(first, second):
    decision = choose_magicdrive_splice(first, second)
    images = {c: {f: first[c][f-10] if f < decision.frame_index else second[c][f-29]
                  for f in range(10, 90)} for c in CAMERAS}
    return images, decision


def identity_paths(checkpoint, sd15):
    return [checkpoint / p for p in ('hydra/config.yaml', 'hydra/overrides.yaml',
        'controlnet/config.json', 'controlnet/diffusion_pytorch_model.bin',
        'unet/config.json', 'unet/diffusion_pytorch_model.bin')] + [sd15 / p for p in (
        'text_encoder/model.safetensors', 'unet/diffusion_pytorch_model.safetensors',
        'vae/diffusion_pytorch_model.safetensors')]


def load_pipeline(args):
    # Python prepends this script's MagicDrive-main directory; explicitly prefer
    # the pinned temporal checkout over that older same-named package.
    sys.path.insert(0, str(args.video_root.resolve()))
    import hydra
    from omegaconf import OmegaConf
    from magicdrive.misc.test_utils import build_pipe
    overrides = list(OmegaConf.load(args.checkpoint / 'hydra/overrides.yaml'))
    overrides += ['resume_from_checkpoint=' + str(args.checkpoint),
                  'model.pretrained_model_name_or_path=' + str(args.sd15_root)]
    with hydra.initialize_config_dir(config_dir=str(args.video_root / 'configs'), version_base=None):
        cfg = hydra.compose(config_name='test_config', overrides=overrides)
    pipe, dtype = build_pipe(cfg, 'cuda')
    return pipe, cfg, dtype


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'prediction-root', 'checkpoint', 'sd15-root', 'video-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--generation-set', default='balanced_500')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--scene-indices')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('inference requires a Slurm allocation')
    import torch
    assert torch.cuda.device_count() == 1 and 'L40' in torch.cuda.get_device_name(0).upper()
    scenes = shard_scenes(select_scenes(load_generation_set(args.data_root, args.generation_set),
                          args.pilot, args.scene_indices, args.limit), args.rank, args.world_size)
    pipe, cfg, dtype = load_pipeline(args)
    identity = checkpoint_identity(identity_paths(args.checkpoint, args.sd15_root))
    failed = False
    for scene in scenes:
        final = args.prediction_root / 'magicdrive' / scene.dataset / 'scenes' / scene.scene_id
        if validate_completed_scene(final, 'magicdrive', scene).valid:
            if json.loads((final / 'status.json').read_text()).get('control_adapter', {}).get('version') != CONTROL_ADAPTER_VERSION:
                raise RuntimeError('Archive obsolete control-adapter output before rerunning: ' + str(final))
            print('SKIP {}'.format(scene.relative_path), flush=True)
            continue
        try:
            inputs = load_scene_inputs(scene)
            with ScenePublisher(args.prediction_root, 'magicdrive', scene, os.environ['SLURM_JOB_ID'], str(args.rank)) as publisher:
                with torch.inference_mode():
                    images, decision, dropped = generate_scene(pipe, cfg, inputs, args.seed, dtype)
                publisher.write_frames(images)
                publisher.publish({'checkpoint': identity, 'seed': args.seed,
                    'control_adapter': control_metadata(inputs),
                    'inference_settings': {'guidance_scale': 2, 'num_inference_steps': 20, 'decode_bs': 1, 'init_noise': 'rand_all'},
                    'history_frames_used': [], 'repeated_reference_slots': [], 'conditioning_mode': 'layout_only',
                    'used_future_ego_trajectory': True, 'used_future_agent_trajectories': True,
                    'available_map_channels': list(inputs.map_condition.available_channels),
                    'unavailable_map_channels': list(inputs.map_condition.unavailable_channels),
                    'temporal_strategy': {'windows': [list(w) for w in magicdrive_plan().condition_windows],
                        'splice_frame': decision.frame_index, 'splice_score': decision.score, 'splice_metric': decision.metric},
                    'dropped_agent_categories': dropped})
            print('COMPLETE {}'.format(scene.relative_path), flush=True)
        except Exception as exc:
            traceback.print_exc()
            failed = True
            try:
                write_failure(args.prediction_root, 'magicdrive', scene, 'inference', exc, args.rank, os.environ['SLURM_JOB_ID'])
            except Exception:
                # A broken failure-record filesystem must not discard the rest
                # of this worker's assigned scenes. Preserve both tracebacks.
                traceback.print_exc()
    return int(failed)

if __name__ == '__main__':
    raise SystemExit(main())

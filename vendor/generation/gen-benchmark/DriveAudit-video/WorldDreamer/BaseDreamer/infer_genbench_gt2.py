"""Released BaseDreamer single-reference inference on RGB-safe GenBench GT2."""
import argparse
import os
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np
from PIL import Image

MODEL_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = MODEL_ROOT.parents[2]
sys.path.insert(0, str(BENCH_ROOT))
from genbench_gt2.constants import CAMERAS
from genbench_gt2.dataset import load_generation_set, load_scene_inputs, history_rgb_path
from genbench_gt2.geometry import box_corners, transform_agent_to_target, transform_points, project_points, apply_linear
from genbench_gt2.raster import class_index, warp_semantic_map, _draw_polyline
from genbench_gt2.selection import select_scenes, shard_scenes
from genbench_gt2.output import ScenePublisher, checkpoint_identity, validate_completed_scene, write_failure

sys.path.insert(0, str(MODEL_ROOT.parent / 'scripts'))
from gt2_raster import bev_raster, camera_raster, clip_native_polyline


def _transform(points, matrix):
    return apply_linear(matrix, np.column_stack((points, np.ones(len(points)))))[:, :3]


def image_tensor(images):
    import torch
    return torch.from_numpy(np.stack([np.asarray(im.convert('RGB'), dtype=np.float32)
        .transpose(2, 0, 1) / 127.5 - 1 for im in images]))[None]


def build_conditions(inputs, frame, reference_frame):
    """Build native 4-channel BEV, 14-channel layout and visible box tokens."""
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from dataclasses import replace
    from gt2_native import ego_to_native, native_calibration
    if frame not in range(10, 90) or reference_frame not in (9, frame - 1):
        raise ValueError('reference must be history9 or previous generated frame')
    pose = inputs.ego_to_anchor[frame]
    ego_native = ego_to_native(inputs)
    native_to_anchor = pose @ np.linalg.inv(ego_native)
    m = inputs.map_condition
    semantic, _ = warp_semantic_map(m.mask, m.extent_m, m.resolution_m,
        native_to_anchor, (-50, 50, -50, 50), .5, m.available_channels, (200, 200))
    drivable = np.maximum(semantic[0], semantic[7])
    lines_native, lines_ego = [], []
    vector_classes = m.vectors.get('classes', [])
    for line in m.vectors.get('lines', []):
        label = line['label']
        label = vector_classes[label] if isinstance(label, int) else label
        if label not in ('divider', 'ped_crossing', 'boundary'):
            continue
        channel = ('divider', 'ped_crossing', 'boundary').index(label)
        points = transform_points(np.asarray(line['points']), pose)
        # Camera map geometry lives on the target ego ground plane. The native
        # lidar frame is only used for BEV/token coordinates, not its z=0 plane.
        points[:, 2] = 0
        for native_segment in clip_native_polyline(_transform(points, ego_native), 49.8 if channel == 2 else 50.):
            lines_native.append((channel, native_segment))
            lines_ego.append((channel, _transform(native_segment, np.linalg.inv(ego_native))))
    bev = bev_raster(lines_native, drivable)
    cells = np.argwhere(drivable > 0)
    ground_native = np.column_stack((cells * .5 - 50 + .25, np.zeros(len(cells))))
    ground_ego = _transform(ground_native, np.linalg.inv(ego_native))
    ground_ego[:, 2] = 0
    layouts, views, camera_params, dropped = [], [], [], {}
    agents, token_boxes = [], []
    for agent in inputs.agents_by_frame.get(frame, ()):
        label = class_index(agent.category)
        if label is None:
            dropped[agent.category] = dropped.get(agent.category, 0) + 1
            continue
        target_agent = transform_agent_to_target(agent, pose)
        center = _transform(target_agent.center_xyz[None], ego_native)[0]
        heading = ego_native[:3, :3] @ np.array([np.cos(target_agent.yaw), np.sin(target_agent.yaw), 0.])
        theta = np.arctan2(heading[1], heading[0])
        # This checkout's old mmdet3d uses [width,length,height,-yaw-pi/2].
        # Its training token path retains the raw-center-as-bottom convention;
        # the image renderer separately recenters to the physical box center.
        raw = torch.tensor([[*center, *target_agent.size_wlh, -theta - np.pi/2]], dtype=torch.float32)
        tokens = LiDARInstance3DBoxes(raw.clone(), origin=(.5, .5, 0.)).corners.numpy()[0]
        physical = LiDARInstance3DBoxes(raw.clone(), origin=(.5, .5, .5)).corners.numpy()[0]
        token_boxes.append(tokens)
        agents.append((_transform(physical, np.linalg.inv(ego_native)), label))
    for camera in CAMERAS:
        calibration = native_calibration(inputs, camera)
        camera_params.append(np.concatenate((calibration.intrinsics,
            calibration.camera_to_ego[:3]), axis=-1))
        # Camera tokens use the virtual 1600x900 plane. Native raster output
        # applies y_out=y_full/4-1; invert that distinct operation so projected
        # layouts remain aligned with the public 400x224 challenge images.
        raster_k = inputs.calibrations[camera].intrinsics.copy()
        raster_k[:2] *= 4
        raster_k[1, 2] += 4
        ego_camera = replace(calibration, intrinsics=raster_k,
                             camera_to_ego=inputs.calibrations[camera].camera_to_ego)
        layouts.append(camera_raster(lines_ego, ground_ego, agents, ego_camera))
        visible = []
        for (corners, label), tokens in zip(agents, token_boxes):
            if project_points(corners, inputs.calibrations[camera])[1].any():
                visible.append((tokens, label))
        views.append(visible)
    count = max(1, max(map(len, views)))
    boxes = np.zeros((1, 6, count, 8, 3), np.float32)
    classes = np.full((1, 6, count), -1, np.int64)
    masks = np.zeros((1, 6, count), bool)
    for vi, entries in enumerate(views):
        for bi, (corners, label) in enumerate(entries):
            boxes[0, vi, bi], classes[0, vi, bi], masks[0, vi, bi] = corners, label, True
    return {'bev_hdmap': torch.from_numpy(bev.astype(np.float32)[None]),
        'camera_param': torch.from_numpy(np.array(camera_params, dtype=np.float32)[None]),
        'rel_pose': torch.from_numpy((np.linalg.inv(pose) @ inputs.ego_to_anchor[reference_frame]).astype(np.float32)[None]),
        'layout_canvas': torch.from_numpy(np.stack(layouts).astype(np.float32)[None]),
        'bev_controlnet_kwargs': {'bboxes_3d_data': {'bboxes': torch.from_numpy(boxes),
            'classes': torch.from_numpy(classes), 'masks': torch.from_numpy(masks)}}}, dropped


def load_pipeline(args):
    import hydra
    from omegaconf import OmegaConf
    # Upstream loader resolves its fixed CLIP path relative to BaseDreamer.
    os.chdir(MODEL_ROOT)
    sys.path.insert(0, str(MODEL_ROOT))
    from projects.dreamer.utils.test_utils import build_pipe
    overrides = list(OmegaConf.load(args.checkpoint.parent / 'hydra/overrides.yaml'))
    overrides += ['resume_from_checkpoint=' + str(args.checkpoint),
                  'model.pretrained_model_name_or_path=' + str(args.sd15_root)]
    with hydra.initialize_config_dir(config_dir=str(MODEL_ROOT / 'configs'), version_base=None):
        cfg = hydra.compose(config_name='test_config_nus', overrides=overrides)
    pipe, dtype = build_pipe(cfg, 'cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe, cfg, dtype


def generate_scene(pipe, cfg, inputs, seed, dtype, num_inference_steps=None, reference_mode="autoregressive"):
    import torch
    with_images = []
    for camera in CAMERAS:
        with Image.open(history_rgb_path(inputs.scene, camera, 9)) as im:
            with_images.append(im.convert('RGB'))
    reference = image_tensor(with_images)
    images = {camera: {} for camera in CAMERAS}
    dropped = {}
    params = dict(cfg.runner.pipeline_param)
    if num_inference_steps is not None:
        params['num_inference_steps'] = num_inference_steps
    for frame in range(10, 90):
        kwargs, frame_dropped = build_conditions(inputs, frame, 9 if reference_mode == 'fixed' else frame - 1)
        for key, value in frame_dropped.items():
            dropped[key] = dropped.get(key, 0) + value
        for key in ('bev_hdmap', 'camera_param', 'rel_pose', 'layout_canvas'):
            kwargs[key] = kwargs[key].to(device='cuda', dtype=dtype)
        kwargs['ref_images'] = reference.to(device='cuda', dtype=dtype)
        with torch.inference_mode(), torch.autocast('cuda'):
            rows = pipe(prompt=[inputs.prompt], height=224, width=400,
                generator=torch.Generator(device='cuda').manual_seed(seed + inputs.scene.index * 1009),
                **kwargs, **params).images
        if len(rows) != 1 or len(rows[0]) != 6 or any(im.size != (400, 224) for im in rows[0]):
            raise RuntimeError('BaseDreamer returned incorrect image dimensions or view count')
        if reference_mode == 'autoregressive':
            reference = image_tensor(rows[0])
        for camera, im in zip(CAMERAS, rows[0]):
            images[camera][frame] = im
        print('FRAME scene={} frame={}'.format(inputs.scene.index, frame), flush=True)
    return images, dropped, params


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--prediction-root', type=Path, required=True)
    weights = BENCH_ROOT / 'checkpoints/base-pretrained'
    p.add_argument('--checkpoint', type=Path, default=weights / 'dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000')
    p.add_argument('--sd15-root', type=Path, default=weights / 'pretrained/stable-diffusion-v1-5')
    p.add_argument('--generation-set', default='balanced_500')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--scene-indices')
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--reference-mode', choices=('fixed', 'autoregressive'), default='autoregressive')
    p.add_argument('--num-inference-steps', type=int, default=None, help='Defaults to released config (50).')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('inference requires a Slurm allocation')
    for name in ('data_root', 'prediction_root', 'checkpoint', 'sd15_root'):
        setattr(args, name, getattr(args, name).resolve())
    import torch
    if torch.cuda.device_count() != 1 or 'L40' not in torch.cuda.get_device_name(0).upper():
        p.error('each worker requires one visible L40 GPU')
    scenes = shard_scenes(select_scenes(load_generation_set(args.data_root, args.generation_set),
        args.pilot, args.scene_indices, args.limit), args.rank, args.world_size)
    scenes = [s for s in scenes if not validate_completed_scene(args.prediction_root /
        'basedreamer' / s.dataset / 'scenes' / s.scene_id, 'basedreamer', s).valid]
    if not scenes:
        return 0
    pipe, cfg, dtype = load_pipeline(args)
    identity = checkpoint_identity([args.checkpoint / 'controlnet/diffusion_pytorch_model.bin',
        args.checkpoint / 'unet/diffusion_pytorch_model.bin',
        args.checkpoint / 'image_proj/image_proj_model.bin',
        args.checkpoint.parent / 'hydra/overrides.yaml'])
    failed = False
    consecutive_failures = 0
    for scene in scenes:
        try:
            inputs = load_scene_inputs(scene)
            from gt2_native import rig_metadata
            with ScenePublisher(args.prediction_root, 'basedreamer', scene,
                    os.environ['SLURM_JOB_ID'], str(args.rank)) as publisher:
                images, dropped, params = generate_scene(pipe, cfg, inputs, args.seed, dtype, args.num_inference_steps, args.reference_mode)
                publisher.write_frames(images)
                publisher.publish({'checkpoint': identity, 'seed': args.seed,
                    'inference_settings': params, 'history_frames_used': [9],
                    'conditioning_revision': 'native_rig_raster_v2', 'native_rig': rig_metadata(inputs),
                    'raster_intrinsics': 'inverse_stride4_and_crop_first_row_of_challenge_K',
                    'repeated_reference_slots': [], 'conditioning_mode': 'history_single_ref_' + args.reference_mode,
                    'used_future_ego_trajectory': True, 'used_future_agent_trajectories': True,
                    'available_map_channels': list(inputs.map_condition.available_channels),
                    'unavailable_map_channels': list(inputs.map_condition.unavailable_channels),
                    'temporal_strategy': {'type': args.reference_mode + '_single_frame', 'noise_policy': 'fixed_per_scene', 'initial_reference': 9,
                        'generated_reference_frames': list(range(10, 89)) if args.reference_mode == 'autoregressive' else [], 'future_frames': list(range(10, 90))},
                    'dropped_agent_categories': dropped})
            consecutive_failures = 0
            print('COMPLETE {}'.format(scene.relative_path), flush=True)
        except Exception as exc:
            failed = True
            consecutive_failures += 1
            traceback.print_exc()
            try:
                write_failure(args.prediction_root, 'basedreamer', scene, 'inference', exc, args.rank, os.environ['SLURM_JOB_ID'])
            except Exception:
                traceback.print_exc()
            if consecutive_failures >= 3:
                print('STOP: three consecutive scene failures', flush=True)
                return 1
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

"""Released DreamForge-t rolling inference with history-only RGB references."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from genbench_gt2.constants import CAMERAS, MAP_CHANNELS
from genbench_gt2.dataset import load_generation_set, load_scene_inputs
from genbench_gt2.geometry import transform_agent_to_target, transform_points, project_points
from genbench_gt2.raster import class_index, warp_semantic_map, _draw_polyline
from genbench_gt2.selection import select_scenes, shard_scenes
from genbench_gt2.output import ScenePublisher, checkpoint_identity, validate_completed_scene, write_failure
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gt2_native import ego_to_native, native_calibration, rig_metadata, clip_polyline_xy, ego_to_world



def clip_native_polyline(points, extent=50.0):
    """Use the native local patch; boundary contours trim a further0.2m."""
    return clip_polyline_xy(points, (-extent, extent, -extent, extent))



def build_can_bus(inputs, frames):
    """Native 9D bus: global delta XYZ, absolute quaternion/yaw, delta degrees."""
    import torch
    from pyquaternion import Quaternion
    motion, previous = [], None
    for frame in frames:
        pose = ego_to_world(inputs, frame)
        rotation = Quaternion(matrix=pose[:3, :3])
        angle = np.arctan2(rotation.rotation_matrix[1, 0], rotation.rotation_matrix[0, 0]) / np.pi * 180
        if angle < 0:
            angle += 360
        bus = np.zeros(9, np.float32)
        bus[3:7] = rotation.elements
        bus[7] = angle / 180 * np.pi
        if previous is not None:
            bus[:3] = pose[:3, 3] - previous[0]
            # Match native get_can_bus/prepare_train_data, including the
            # native 0/360 boundary convention; do not rotate deltas to ego.
            bus[8] = angle - previous[1]
        previous = (pose[:3, 3].copy(), angle)
        motion.append(bus)
    return torch.tensor(np.stack(motion))


def build_dreamforge_window(inputs, frames, references):
    """Build native four-plane maps, fourteen-plane layouts and 3D controls."""
    import torch
    maps, layouts, visible, raw_boxes = [], [], [], []
    dropped = {}
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from dreamforge.dataset.map_utils import visualize_bev_hdmap, project_box_to_image
    native_from_ego = ego_to_native(inputs)
    ego_from_native = np.linalg.inv(native_from_ego)
    calibrations = [native_calibration(inputs, c) for c in CAMERAS]
    camera = np.stack([np.concatenate([c.intrinsics, c.camera_to_ego[:3]], -1) for c in calibrations])
    projections = []
    for c in calibrations:
        k = np.eye(4)
        k[:3, :3] = c.intrinsics
        projections.append(k @ np.linalg.inv(c.camera_to_ego))
    for frame in frames:
        pose = inputs.ego_to_anchor[frame] @ ego_from_native
        m = inputs.map_condition
        semantic = warp_semantic_map(m.mask, m.extent_m, m.resolution_m, pose,
            (-50, 50, -50, 50), .5, m.available_channels, (200, 200))[0]
        drivable = semantic[MAP_CHANNELS.index('drivable_area')] | semantic[MAP_CHANNELS.index('road_block')]
        vector_lines = []
        labels = m.vectors['classes']
        for line in m.vectors['lines']:
            label = labels[line['label']] if isinstance(line['label'], int) else line['label']
            if label not in ('divider', 'ped_crossing', 'boundary'):
                continue
            channel = ('divider', 'ped_crossing', 'boundary').index(label)
            points = transform_points(np.asarray(line['points']), pose)
            for clipped in clip_native_polyline(points, 49.8 if channel == 2 else 50.0):
                vector_lines.append((channel, clipped))
        bev = visualize_bev_hdmap([p[:, :2] for _, p in vector_lines],
            [label for label, _ in vector_lines], (200, 200), drivable_mask=drivable)
        bev = bev.transpose(2, 0, 1).copy()
        box_rows, box_labels = [], []
        for agent in inputs.agents_by_frame.get(frame, ()):
            label = class_index(agent.category)
            if label is None:
                dropped[agent.category] = dropped.get(agent.category, 0) + 1
                continue
            agent = transform_agent_to_target(agent, pose)
            w, length, h = agent.size_wlh
            x, y, z = agent.center_xyz
            # This installed mmdet3d uses clockwise SECOND yaw and w,l,h.
            # Keep the native dataset's center-z tensor; its bbox embedding
            # and scene embedder consume this unshifted representation.
            box_rows.append([x, y, z, w, length, h, -agent.yaw-np.pi/2])
            box_labels.append(label)
        native_boxes = LiDARInstance3DBoxes(
            torch.tensor(np.asarray(box_rows, dtype=np.float32).reshape(-1, 7)),
            origin=(.5, .5, 0))
        raw_boxes.append(native_boxes.tensor)
        labels_tensor = torch.tensor(box_labels, dtype=torch.long)
        token_corners = native_boxes.corners.numpy() if len(native_boxes) else np.empty((0, 8, 3))
        # Native visibility and the 14-plane raster use gravity-centered boxes.
        physical_boxes = LiDARInstance3DBoxes(native_boxes.tensor.clone(), origin=(.5, .5, .5))
        physical_corners = physical_boxes.corners.numpy() if len(physical_boxes) else np.empty((0, 8, 3))
        view_entries, view_layouts = [], []
        cells = np.argwhere(drivable > 0)
        xy = cells * .5 - 49.75
        # BEV is in the elevated LiDAR frame. Ground is ego z=0, never native
        # z=0; otherwise a 1.84m-high plane is projected through the cameras.
        z = (-ego_from_native[2, 3] - xy @ ego_from_native[2, :2]) / ego_from_native[2, 2]
        ground = np.column_stack((xy, z))
        for vi, calibration in enumerate(calibrations):
            # Match native rasterization before its nearest-neighbor downsize.
            map_canvas = np.zeros((900, 1600, 4), np.uint8)
            for channel, points in vector_lines:
                rgb = np.zeros((900, 1600, 3), np.uint8)
                _draw_polyline(rgb, points, calibration, (1, 0, 0), 4)
                map_canvas[:, :, channel] |= rgb[..., 0]
            drivable_canvas = np.zeros((900, 1600), np.uint8)
            pixels, valid = project_points(ground, calibration)
            for uv in pixels[valid]:
                if -20 < uv[0] < 1620 and -20 < uv[1] < 920:
                    cv2.circle(drivable_canvas, tuple(uv.astype(int)), 20, 1, -1)
            map_canvas[:, :, 3] = drivable_canvas
            map_canvas = cv2.resize(map_canvas, (400, 224), interpolation=cv2.INTER_NEAREST)
            box_canvas = project_box_to_image(native_boxes, labels_tensor,
                torch.tensor(projections[vi], dtype=torch.float32),
                object_classes=list(range(10)), image_size=(224, 400))
            entries = []
            for corners, token, label in zip(physical_corners, token_corners, box_labels):
                if project_points(corners, calibration)[1].any():
                    entries.append((token, label))
            view_entries.append(entries)
            view_layouts.append(np.concatenate((map_canvas, box_canvas), -1).transpose(2, 0, 1))
        maps.append(bev)
        layouts.append(view_layouts)
        visible.append(view_entries)
    count = max(1, max(len(v) for row in visible for v in row))
    boxes = np.zeros((len(frames), 6, count, 8, 3), np.float32)
    classes = np.full((len(frames), 6, count), -1, np.int64)
    masks = np.zeros(classes.shape, bool)
    for fi, row in enumerate(visible):
        for vi, entries in enumerate(row):
            for bi, (corners, label) in enumerate(entries):
                boxes[fi, vi, bi], classes[fi, vi, bi], masks[fi, vi, bi] = corners, label, True
    return SimpleNamespace(bev=torch.tensor(np.stack(maps), dtype=torch.float32),
        layout=torch.tensor(np.asarray(layouts), dtype=torch.float32),
        camera=torch.tensor(np.repeat(camera[None], len(frames), 0), dtype=torch.float32),
        boxes=torch.from_numpy(boxes), classes=torch.from_numpy(classes), masks=torch.from_numpy(masks),
        can_bus=build_can_bus(inputs, list(references) + list(frames)), dropped=dropped,
        meta={'lidar2image': [torch.tensor(np.stack(projections), dtype=torch.float32) for _ in frames],
              'gt_bboxes_3d': [SimpleNamespace(data=SimpleNamespace(tensor=b)) for b in raw_boxes]})


def load_pipeline(args):
    import hydra
    from omegaconf import OmegaConf
    from dreamforge.misc.test_utils import build_pipe
    overrides = [x for x in OmegaConf.load(args.checkpoint / 'hydra/overrides.yaml')
                 if not x.lstrip('+').startswith('resume_from_checkpoint=')]
    overrides += ['model.pretrained_model_name_or_path=' + str(args.sd15_root),
                  '++model.pretrained_dreamforge=' + str(args.single_checkpoint)]
    with hydra.initialize_config_dir(config_dir=str(Path(__file__).parent / 'configs'), version_base=None):
        cfg = hydra.compose(config_name='test_config', overrides=overrides)
    cfg.resume_from_checkpoint = str(args.checkpoint)
    pipe, dtype = build_pipe(cfg, 'cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe, cfg, dtype


def normalize_views(views):
    import torch
    return torch.stack([torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy()).permute(2,0,1)/127.5-1 for image in views])


def generate_scene(pipe, cfg, dtype, inputs, seed, steps, reference_history=(9, 9)):
    import torch
    scene_seed = seed + inputs.scene.index * 1009
    torch.manual_seed(scene_seed)
    torch.cuda.manual_seed_all(scene_seed)
    references = list(reference_history)
    ref_views = []
    for frame in references:
        history = []
        for c in CAMERAS:
            with Image.open(inputs.history_rgb[c][frame]) as image:
                history.append(image.convert('RGB'))
        ref_views.append(normalize_views(history))
    ref_values = torch.stack(ref_views)
    images = {c: {} for c in CAMERAS}
    windows = []
    dropped = {}
    length = int(cfg.model.video_length)
    assert int(cfg.model.ref_length) == 2 and length >= 2
    for wi, start in enumerate(range(10, 90, length)):
        output_frames = list(range(start, min(start+length, 90)))
        frames = output_frames + [89] * (length-len(output_frames))
        batch = build_dreamforge_window(inputs, frames, references)
        print('DreamForge scene {} window {} frames {}'.format(inputs.scene.index, wi, output_frames), flush=True)
        params = dict(cfg.runner.pipeline_param)
        params.update(num_inference_steps=steps, decode_bs=1, init_noise='rand_all', double_cfg_inference=False)
        output = pipe(prompt=[inputs.prompt]*length,
            image=[batch.bev.to(dtype), batch.layout.to(dtype)],
            ref_image=ref_values.to(dtype), can_bus=batch.can_bus.to(dtype),
            camera_param=batch.camera.to(dtype), height=224, width=400,
            generator=torch.Generator(device='cuda').manual_seed(scene_seed+wi),
            bev_controlnet_kwargs={'bboxes_3d_data': {'bboxes': batch.boxes,
                'classes': batch.classes, 'masks': batch.masks}}, img_metas=batch.meta, **params)
        assert len(output.images) == length and all(len(row) == 6 for row in output.images)
        for f, row in zip(output_frames, output.images):
            for c, image in zip(CAMERAS, row):
                images[c][f] = image
        windows.append({'reference_frames': references, 'control_frames': frames, 'output_frames': output_frames})
        references = output_frames[-2:]
        if len(references) == 1:
            references = references * 2
        ref_values = torch.stack([normalize_views([images[c][f] for c in CAMERAS]) for f in references])
        for category, count in batch.dropped.items():
            dropped[category] = dropped.get(category, 0) + count
    return images, windows, dropped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--prediction-root', type=Path, required=True)
    weights = ROOT / 'checkpoints'
    p.add_argument('--checkpoint', type=Path, default=weights/'models--Jianbiao--dreamforge-t')
    p.add_argument('--single-checkpoint', type=Path, default=weights/'models--Jianbiao--dreamforge-s')
    p.add_argument('--sd15-root', type=Path, default=weights/'base-pretrained/pretrained/stable-diffusion-v1-5')
    p.add_argument('--generation-set', default='balanced_500')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--scene-indices')
    p.add_argument('--limit', type=int)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--num-inference-steps', type=int, default=20)
    p.add_argument('--reference-history', default='9,9', help='Two chronological history frame indices, e.g. 8,9; default repeats9.')
    args = p.parse_args()
    try:
        args.reference_history = tuple(int(v) for v in args.reference_history.split(','))
        if len(args.reference_history) != 2 or any(f not in range(10) for f in args.reference_history) or list(args.reference_history) != sorted(args.reference_history):
            raise ValueError
    except ValueError:
        p.error('--reference-history requires two chronological indices in0..9')
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('inference requires a Slurm allocation')
    import torch
    assert torch.cuda.device_count() == 1 and 'L40' in torch.cuda.get_device_name(0).upper()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    scenes = shard_scenes(select_scenes(load_generation_set(args.data_root, args.generation_set),
        args.pilot, args.scene_indices, args.limit), args.rank, args.world_size)
    pipe, cfg, dtype = load_pipeline(args)
    identity = checkpoint_identity([path for directory in (args.checkpoint, args.single_checkpoint)
        for path in sorted(directory.rglob('*')) if path.is_file() and path.suffix in ('.json', '.yaml', '.bin', '.safetensors')])
    failed = False
    consecutive_failures = 0
    for scene in scenes:
        final = args.prediction_root/'dreamforge'/scene.dataset/'scenes'/scene.scene_id
        if validate_completed_scene(final, 'dreamforge', scene).valid:
            print('SKIP {}'.format(scene.relative_path), flush=True)
            continue
        try:
            inputs = load_scene_inputs(scene)
            with ScenePublisher(args.prediction_root, 'dreamforge', scene, os.environ['SLURM_JOB_ID'], str(args.rank)) as publisher:
                with torch.inference_mode():
                    images, windows, dropped = generate_scene(pipe, cfg, dtype, inputs, args.seed, args.num_inference_steps, args.reference_history)
                publisher.write_frames(images)
                publisher.publish({'checkpoint': identity, 'seed': args.seed,
                    'inference_settings': {'num_inference_steps': args.num_inference_steps, 'decode_bs': 1, 'init_noise': 'rand_all'},
                    'history_frames_used': sorted(set(args.reference_history)), 'repeated_reference_slots': [args.reference_history[0]] if args.reference_history[0] == args.reference_history[1] else [], 'conditioning_mode': 'history_reference_and_layout',
                    'used_future_ego_trajectory': True, 'used_future_agent_trajectories': True,
                    'available_map_channels': list(inputs.map_condition.available_channels),
                    'unavailable_map_channels': list(inputs.map_condition.unavailable_channels),
                    'temporal_strategy': {'windows': windows, 'method': 'rolling_two_generated_references'},
                    'native_rig': rig_metadata(inputs),
                    'adapter_version': 'native_rig_raster_boxes_localmap_worldbus_v4',
                    'can_bus_coordinate_frame': ('history_frame_9_ego_local_origin' if (inputs.scene.root / inputs.scene.relative_path / 'participant_render_frame.json').exists() else 'world'),
                    'dropped_agent_categories': dropped})
            consecutive_failures = 0
            print('COMPLETE {}'.format(scene.relative_path), flush=True)
        except Exception as exc:
            failed = True
            traceback.print_exc()
            consecutive_failures += 1
            write_failure(args.prediction_root, 'dreamforge', scene, 'inference', exc, args.rank, os.environ['SLURM_JOB_ID'])
            if consecutive_failures >= 3:
                print('Stopping after three consecutive scene failures', flush=True)
                break
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

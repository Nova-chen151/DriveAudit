"""GT-control inference; future RGB is never opened by batch preparation."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np

if __package__:
    from . import infer as geom
    from . import catk_output
    from .v2_contract import MapMetadata
else:
    import infer as geom
    import catk_output
    from v2_contract import MapMetadata

REPO = Path(__file__).resolve().parents[1]
FRAMES = tuple(range(1, 90))


def load_map(scene):
    metadata = MapMetadata.from_dict(json.loads((scene / 'map/map_meta.json').read_text()))
    # GT2 adds ground_height_m / observability arrays to the v2 archive.
    # The pretrained map encoder consumes the eight semantic channels only.
    with np.load(scene / 'map/semantic_map.npz', allow_pickle=False) as archive:
        mask = archive['mask']
    if mask.shape != (8, 1024, 1024) or mask.dtype != np.uint8 or np.any(mask > 1):
        raise ValueError(f'Invalid semantic map: {scene}')
    return mask, metadata


def scene_paths(root, scene_list=None):
    values = (json.loads(Path(scene_list).read_text()) if scene_list else
              json.loads((root / 'metadata/generation_sets.json').read_text())['balanced_500'])
    if not isinstance(values, list) or not values or any(not isinstance(p, str) for p in values):
        raise ValueError('Scene list must be a nonempty JSON array of relative scene IDs')
    paths = [Path(p) for p in values]
    if len(set(paths)) != len(paths) or (scene_list is None and len(paths) != 500):
        raise ValueError('Scene IDs must be distinct; balanced_500 must contain exactly 500 scenes')
    for p in paths:
        if (p.is_absolute() or '..' in p.parts or len(p.parts) != 3 or
                p.parts[0] not in catk_output.DATASETS or p.parts[1] != 'scenes' or
                not re.fullmatch(r'case_[0-9]{6}', p.name) and not p.name.startswith('adv_case_')):
            raise ValueError(f'Invalid relative scene ID: {p}')
        (root / p).resolve().relative_to(root.resolve())
    return paths


def rows(path):
    with path.open(newline='') as h:
        return list(csv.DictReader(h))


def read_controls(scene, trajectory_scene=None):
    ego = {}
    agents = {}
    for folder, ego_file, agent_file, allowed in (
        ('history', 'ego.csv', 'agents.csv', range(1, 10)),
        ('future', 'ego_trajectory.csv', 'future_trajectories.csv', range(10, 90)),
    ):
        for row in rows(scene / folder / ego_file):
            f = int(row['frame_index'])
            if f in allowed:
                if f in ego or not geom._bool(row['valid']):
                    raise ValueError(f'Invalid/duplicate ego frame {f}: {scene}')
                ego[f] = row
        agent_path = scene / folder / agent_file
        if folder == 'future' and trajectory_scene is not None:
            agent_path = trajectory_scene / folder / agent_file
        for row in rows(agent_path):
            f = int(row['frame_index'])
            if f in allowed:
                key = (f, row['agent_id'])
                if key in agents:
                    raise ValueError(f'Duplicate agent {key}: {scene}')
                if geom._bool(row['valid']):
                    values = np.array([float(row[k]) for k in ('x','y','z','width','length','height','yaw')])
                    if not np.isfinite(values).all() or np.any(values[3:6] <= 0):
                        raise ValueError(f'Invalid box {key}: {scene}')
                agents[key] = row
    if set(ego) != set(FRAMES):
        raise ValueError(f'Missing ego frames {set(FRAMES) - set(ego)}: {scene}')
    if not np.isfinite([[float(ego[f][k]) for k in ('x','y','yaw')] for f in FRAMES]).all():
        raise ValueError(f'Nonfinite ego pose: {scene}')
    return ego, agents


def build_geometry(scene, ego, agents):
    if __package__:
        from .model_conditioning import training_calibration
    else:
        from model_conditioning import training_calibration
    calibration = training_calibration(json.loads((scene / 'calibration/cameras.json').read_text()))
    origin = geom._renderer_origin(calibration)
    rotation = geom._renderer_to_ego_rotation(calibration)
    cameras, to_cameras = geom._camera_matrices(calibration, origin)
    pose_data = json.loads((scene / 'calibration/reference_ego_poses.json').read_text())
    native_poses = np.asarray(pose_data['ego_to_anchor'], dtype=np.float64)
    if native_poses.shape != (90, 4, 4) or not np.isfinite(native_poses).all():
        raise ValueError(f'Invalid native ego poses: {scene}')
    renderer_to_ego = np.eye(4)
    renderer_to_ego[:3, :3] = rotation
    renderer_to_ego[:3, 3] = origin
    ids = sorted({aid for _, aid in agents})
    # One masked padding box supports genuinely empty scenes.
    n = max(len(ids), 1)
    boxes = np.zeros((89, 1, n, 8, 3), np.float32)
    classes = np.full((89, 1, n), -1, np.int32)
    masks = np.zeros((89, 6, n), np.float32)
    poses = []
    positions, yaws = [], []
    for t, f in enumerate(FRAMES):
        pos = np.array([float(ego[f]['x']), float(ego[f]['y'])])
        yaw = float(ego[f]['yaw'])
        positions.append(pos)
        yaws.append(yaw)
        # GT2 supplies the full native oracle pose. Keep roll, pitch and height
        # consistent with the exact camera extrinsics, not only the CSV XY/yaw.
        if not np.allclose(native_poses[f, :2, 3], pos, atol=1e-4):
            raise ValueError(f'CSV/native pose mismatch at frame {f}: {scene}')
        pose = native_poses[f] @ renderer_to_ego
        poses.append(pose)
        anchor_to_renderer = np.linalg.inv(pose)
        for j, aid in enumerate(ids):
            row = agents.get((f, aid))
            if row is None or not geom._bool(row['valid']):
                continue
            anchor_corners = geom._box_corners(
                np.array([float(row[k]) for k in ('x','y','z')]),
                float(row['width']), float(row['length']), float(row['height']),
                float(row['yaw']))
            corners = (np.column_stack([anchor_corners, np.ones(8)]) @ anchor_to_renderer.T)[:, :3]
            boxes[t, 0, j] = corners
            classes[t, 0, j] = geom.CATEGORY_TO_CLASS[row['category']]
            homogeneous = np.column_stack([corners, np.ones(8)])
            for c, transform in enumerate(to_cameras):
                masks[t, c, j] = np.any((homogeneous @ transform.T)[:, 2] > .05)
    frame_emb = np.stack([np.linalg.inv(p) @ poses[0] for p in poses]).astype(np.float32)
    return dict(bboxes=boxes, classes=classes, masks=masks, frame_emb=frame_emb,
                camera_param=cameras, positions=np.array(positions), yaws=np.array(yaws),
                origin=origin, rotation=rotation, agent_ids=ids)


def prepare(scene, destination, trajectory_scene=None):
    import torch
    from mmcv.parallel import DataContainer
    if __package__:
        from .conditioning_timing import conditioning_timing
    else:
        from conditioning_timing import conditioning_timing
    timing = conditioning_timing(scene)
    ego, agents = read_controls(scene, trajectory_scene)
    g = build_geometry(scene, ego, agents)
    mask, metadata = load_map(scene)
    maps = geom._warp_maps(mask, g['positions'], g['yaws'], g['origin'], metadata, g['rotation'])
    batch = {
        'pixel_values_shape': torch.tensor([[89, 6, 3, 224, 400]], dtype=torch.int32),
        'reference_images': torch.from_numpy(geom._load_reference_images(scene)),
        'captions': [(geom._infer_prompt(scene),)] * 89,
        'bev_map_with_aux': torch.from_numpy(maps[None]),
        'bboxes_3d_data': [DataContainer({k: torch.from_numpy(g[k]) for k in ('bboxes','classes','masks')})],
        'camera_param': torch.from_numpy(np.repeat(g['camera_param'][None, None], 89, axis=1).astype(np.float32)),
        'frame_emb': torch.from_numpy(g['frame_emb'][None]),
        'fps': torch.tensor([timing['fps']], dtype=torch.float32), 'height': torch.tensor([224]),
        'width': torch.tensor([400]), 'num_frames': torch.tensor([89]),
    }
    geom._atomic_pickle(batch, destination)


def validate_output(path):
    from PIL import Image
    expected = {f'{f:06d}.jpg' for f in range(10, 90)}
    for camera in geom.CAMERAS:
        folder = path / 'future/rgb' / camera
        if {p.name for p in folder.glob('*.jpg')} != expected:
            raise ValueError(f'Incomplete output: {folder}')
        for name in expected:
            with Image.open(folder / name) as im:
                if im.size != (400, 224):
                    raise ValueError(f'Wrong output dimensions: {folder / name}')
                im.verify()


def export(samples_path, scene, output, trajectory_scene=None, *,
           standard_layout=False, relative=None, evidence=None, temporal_mode='legacy-ar17', gpus_per_worker=2):
    import torch
    from PIL import Image
    restore = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    try:
        with samples_path.open('rb') as h:
            samples = geom._normalise_samples(pickle.load(h))
    finally:
        torch.storage._load_from_bytes = restore
    if not torch.isfinite(samples).all():
        raise ValueError('Nonfinite generated pixels')
    pixels = ((samples.permute(0, 2, 3, 4, 1).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    if standard_layout:
        return catk_output.publish_pixels(pixels, output, relative, evidence)
    for c, camera in enumerate(geom.CAMERAS):
        folder = output / 'future/rgb' / camera
        folder.mkdir(parents=True, exist_ok=True)
        for t in range(80):
            Image.fromarray(pixels[c, t]).save(folder / f'{t+10:06d}.jpg', quality=95, subsampling=0)
    shutil.copy2(scene / 'future' / 'ego_trajectory.csv', output / 'future' / 'ego_trajectory.csv')
    source_trajectory = trajectory_scene or scene
    shutil.copy2(source_trajectory / 'future' / 'future_trajectories.csv',
                 output / 'future' / 'future_trajectories.csv')
    import imageio.v2 as imageio
    with imageio.get_writer(str(output / 'future/six_view.mp4'), fps=12, codec='libx264', macro_block_size=1) as writer:
        for t in range(80):
            writer.append_data(np.concatenate([
                np.concatenate([pixels[c, t] for c in (0,1,2)], axis=1),
                np.concatenate([pixels[c, t] for c in (5,4,3)], axis=1)], axis=0))
    validate_output(output)
    meta = json.loads((scene / 'map/map_meta.json').read_text())
    (output / 'generation.json').write_text(json.dumps(dict(
        model='MagicDriveDiT-stage3-40k-ft', input_scene=str(scene),
        history_frames=list(range(1,10)), future_frames=list(range(10,90)),
        future_rgb_conditioning=False, trajectory_conditioning='catk' if trajectory_scene else 'ground_truth',
        ego_pose_conditioning='calibration/reference_ego_poses.json (native full SE3)',
        unavailable_map_channels=meta.get('unavailable_channels', []),
        missing_map_policy='zero placeholders; checkpoint has no unknown-channel mask',
        cameras=geom.CAMERAS, sampling_steps=30, seed=1024,
        window_frames=catk_output.temporal_strategy(temporal_mode)['window_frames'],
        overlap_frames=catk_output.temporal_strategy(temporal_mode)['overlap_frames'],
        sequence_parallel_size=gpus_per_worker, slurm_job_id=os.getenv('SLURM_JOB_ID'),
        future_agent_trajectory_source=str(source_trajectory / 'future' / 'future_trajectories.csv'),
        context_source=str(scene),
        conditioning_digest=catk_output.conditioning_evidence(scene, trajectory_scene)['conditioning_digest']
    ), indent=2) + '\n')


def renderer_command(work, pair, gpus_per_worker=2, temporal_mode='legacy-ar17'):
    settings = catk_output.inference_settings(gpus_per_worker, temporal_mode)
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus_per_worker}',
        'scripts/adv_inference_magicdrive.py',
        'configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py',
        '--outputs', str(work / 'render'), '--iopth_pth', str(pair), '--cfg-options',
        f'model.from_pretrained={REPO}/rd/ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt',
        'model.enable_layernorm_kernel=false', 'model.bbox_embedder_param.enable_layernorm_kernel=false',
        'model.frame_emb_param.enable_layernorm_kernel=false', 'model.force_pad_h_for_sp_size=4',
        'num_frames=full', 'autoregressive_reference='+str(settings['autoregressive_reference']).lower(),
        'sample_seed=1024', 'cpu_offload=true', 'scheduler.type=rflow', 'print_config=false']


def run(root, output, limit=None, trajectory_root=None, *, scene_list=None, standard_layout=False,
        gpus_per_worker=2, temporal_mode='legacy-ar17'):
    paths = scene_paths(root, scene_list)[:limit]
    identity = catk_output.runtime_identity(gpus_per_worker, temporal_mode) if standard_layout else None
    catk_output.inference_settings(gpus_per_worker, temporal_mode)
    failures = []
    for index, relative in enumerate(paths):
        ref = catk_output.standard_scene_ref(output, relative) if standard_layout else None
        final = output / ref.dataset / 'scenes' / ref.scene_id if ref else output / relative
        work = output / '_work' / relative
        if standard_layout:
            work = work / uuid.uuid4().hex
        trajectory_scene = trajectory_root / relative if trajectory_root else None
        try:
            evidence = catk_output.metadata(root / relative, trajectory_scene, identity) if standard_layout else None
            if standard_layout and catk_output.can_resume(output, relative, evidence):
                print(f'[SKIP {index+1}/{len(paths)}] {relative}', flush=True)
                continue
            if not standard_layout and (final / 'generation.json').is_file():
                if trajectory_scene:
                    previous = json.loads((final / 'generation.json').read_text())
                    expected = catk_output.conditioning_evidence(root / relative, trajectory_scene)['conditioning_digest']
                    if previous.get('conditioning_digest') != expected:
                        raise ValueError(f'Existing output conditioning mismatch; preserved at {final}')
                validate_output(final)
                print(f'[SKIP {index+1}/{len(paths)}] {relative}', flush=True)
                continue
            if final.exists():
                raise ValueError(f'Unmarked output exists: {final}')
            work.mkdir(parents=True, exist_ok=True)
            batch, samples = work / 'batch.pkl', work / 'samples.pkl'
            print(f'[START {index+1}/{len(paths)}] {relative}', flush=True)
            if trajectory_scene and not (trajectory_scene / 'future/future_trajectories.csv').is_file():
                raise ValueError(f'Missing CATK trajectory overlay: {trajectory_scene}')
            prepare(root / relative, batch, trajectory_scene)
            pair = work / 'pairs.txt'
            pair.write_text(f'{batch}\n{samples}\n')
            command = renderer_command(work, pair, gpus_per_worker, temporal_mode)
            subprocess.run(command, cwd=REPO / 'rd', check=True)
            if temporal_mode == 'full89':
                if __package__:
                    from .conditioning_timing import conditioning_timing
                else:
                    from conditioning_timing import conditioning_timing
                audit = catk_output.validate_full_audit(str(samples)+'.conditioning.json',gpus_per_worker,
                                                       conditioning_timing(root/relative)['fps'])
                if evidence is not None:
                    evidence['native_conditioning_audit'] = audit
            if standard_layout:
                current = catk_output.conditioning_evidence(root / relative, trajectory_scene)
                if current['conditioning_digest'] != evidence['conditioning_digest']:
                    raise ValueError(f'Conditioning sources changed during rendering: {relative}')
                export(samples, root / relative, output, trajectory_scene,
                       standard_layout=True, relative=relative, evidence=evidence,
                       temporal_mode=temporal_mode, gpus_per_worker=gpus_per_worker)
            else:
                stage = work / 'export'
                export(samples, root / relative, stage, trajectory_scene,
                       temporal_mode=temporal_mode, gpus_per_worker=gpus_per_worker)
                final.parent.mkdir(parents=True, exist_ok=True)
                stage.rename(final)
            batch.unlink()
            samples.unlink()
            print(f'[DONE {index+1}/{len(paths)}] {final}', flush=True)
        except Exception:
            failures.append(str(relative))
            traceback.print_exc()
            # Fail immediately on the first scene: do not repeat runtime failures 500 times.
            if index == 0:
                raise
    report = dict(requested=len(paths), failed=failures, completed=len(paths)-len(failures))
    (output / 'run_status.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise RuntimeError(f'{len(failures)} scenes failed; see run_status.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['audit','preflight','run','prepare'])
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--trajectory-root', type=Path,
                   help='Optional root supplying only future/future_trajectories.csv')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--scene-list', type=Path, help='JSON array of explicit relative scene IDs, preserving order')
    p.add_argument('--standard-layout', action='store_true', help='Publish canonical dataset slugs and status.json')
    p.add_argument('--gpus-per-worker', type=int, choices=(2,4,8), default=2)
    p.add_argument('--temporal-mode', choices=('legacy-ar17','full89'), default='legacy-ar17')
    p.add_argument('--limit', type=int, default=None, help='Optional explicit cap; default uses every selected scene')
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    if args.limit is not None and args.limit <= 0:
        p.error('--limit must be positive')
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.trajectory_root:
        args.trajectory_root = args.trajectory_root.resolve()
    selected = scene_paths(args.input_root, args.scene_list)[:args.limit]
    if args.standard_layout:
        for relative in selected:
            catk_output.standard_scene_ref(args.output_root, relative)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'run':
        run(args.input_root, args.output_root, args.limit,
            args.trajectory_root, scene_list=args.scene_list, standard_layout=args.standard_layout,
            gpus_per_worker=args.gpus_per_worker, temporal_mode=args.temporal_mode)
    elif args.mode == 'prepare':
        relative = selected[args.index]
        prepare(args.input_root / relative, args.output_root / f'batch_{args.index}.pkl',
                args.trajectory_root / relative if args.trajectory_root else None)
    else:
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image
        counts, partial, errors = Counter(), Counter(), []
        identity = catk_output.runtime_identity(args.gpus_per_worker, args.temporal_mode) if args.mode == 'preflight' or args.standard_layout else None
        def audit_scene(relative):
            scene = args.input_root / relative
            try:
                trajectory_scene = args.trajectory_root / relative if args.trajectory_root else None
                ego, agents = read_controls(scene, trajectory_scene)
                build_geometry(scene, ego, agents)
                load_map(scene)
                for c in geom.CAMERAS:
                    for f in range(10):
                        with Image.open(scene / 'history/rgb' / c / f'{f:06d}.jpg') as im:
                            if im.size != (400,224):
                                raise ValueError('History image dimension mismatch')
                            im.verify()
                geom._infer_prompt(scene)
                if args.standard_layout:
                    evidence = catk_output.metadata(scene, trajectory_scene, identity)
                    catk_output.can_resume(args.output_root, relative, evidence)
                meta = json.loads((scene / 'map/map_meta.json').read_text())
                return relative, meta.get('unavailable_channels', []), None
            except Exception as exc:
                return relative, [], str(exc)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (relative, missing, error) in enumerate(pool.map(audit_scene, selected)):
                if error:
                    errors.append(dict(scene=str(relative), error=error))
                else:
                    partial.update(missing)
                    counts[relative.parts[0]] += 1
                if (i + 1) % 10 == 0:
                    print(f'[AUDIT] {i+1}/{len(selected)}; errors={len(errors)}', flush=True)
        report = dict(requested=len(selected), scene_list=str(args.scene_list), scenes=dict(counts),
                      missing_map_channels=dict(partial), errors=errors, runtime_identity=identity)
        (args.output_root / 'input_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        if errors:
            raise RuntimeError('Input audit failed')


if __name__ == '__main__':
    main()

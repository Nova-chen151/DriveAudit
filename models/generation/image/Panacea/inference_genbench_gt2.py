"""Two independent Panacea replicas producing atomically published GT2 scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from datetime import timedelta


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--prediction-root', type=Path, required=True)
    p.add_argument('--generation-set', default='balanced_500')
    p.add_argument('--base', required=True)
    p.add_argument('--ckptpath', required=True)
    p.add_argument('--clip_ckpt', required=True)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--vae_decode_batch_size', type=int, default=1)
    p.add_argument('--reference-mode', choices=['history-anchor', 'autoregressive'], default='history-anchor')
    p.add_argument('--chronological', action='store_true')
    p.add_argument('--ordered-targets', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--scene-indices')
    p.add_argument('--limit', type=int)
    return p


def write_generated_frame(publisher, frame_index, panorama):
    from sgm.data.genbench_gt2_dataset import split_panacea_views
    return [publisher.write_frame(camera, frame_index, image.permute(1, 2, 0),
                                  value_range='minus_one_one')
            for camera, image in split_panacea_views(panorama).items()]


def require_matching_renderer(final, reference_mode, chronological=False, ordered_targets=False):
    """Never silently resume images generated with incompatible conditions."""
    from sgm.data.genbench_gt2_native import RENDERER_VERSION
    from sgm.data.genbench_gt2_dataset import TEMPORAL_ADAPTER_VERSION
    metadata = json.loads((final / 'status.json').read_text())
    version = metadata.get('inference_settings', {}).get('condition_renderer')
    if version != RENDERER_VERSION:
        raise RuntimeError(f'Existing output uses renderer {version!r}, expected {RENDERER_VERSION!r}; archive the old scene before regenerating: {final}')
    temporal = metadata.get('inference_settings', {}).get('temporal_adapter')
    expected = 'first-reference-chronological-v1' if chronological else TEMPORAL_ADAPTER_VERSION
    if ordered_targets:
        expected='ordered-targets-last-reference-v1'
    if temporal != expected:
        raise RuntimeError(f'Existing output uses temporal adapter {temporal!r}, expected {TEMPORAL_ADAPTER_VERSION!r}; archive the old scene before regenerating: {final}')
    if metadata.get('inference_settings', {}).get('reference_mode') != reference_mode:
        raise RuntimeError(f'Existing output uses a different reference mode; archive before regenerating: {final}')


def main():
    args = parser().parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Inference requires Slurm')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    from sgm.data.genbench_gt2_dataset import build_panacea_window, split_panacea_views, prepare_native_panacea_batch, restore_panacea_timeline, reference_windows, TEMPORAL_ADAPTER_VERSION
    from sgm.data.genbench_gt2_native import RENDERER_VERSION
    from genbench_gt2.dataset import load_generation_set, load_scene_inputs
    from genbench_gt2.selection import select_scenes, shard_scenes
    from genbench_gt2.temporal import panacea_plan
    from genbench_gt2.raster import class_index
    from genbench_gt2.output import ScenePublisher, checkpoint_identity, validate_completed_scene, write_failure
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', '0'))
    if world_size not in (2, 4) or torch.cuda.device_count() != world_size:
        raise RuntimeError(f'Exactly {world_size} visible GPUs and distributed ranks required')
    if any('L40' not in torch.cuda.get_device_name(i) for i in range(world_size)):
        raise RuntimeError(f'{world_size} NVIDIA L40 devices required')
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=48))
    config = OmegaConf.load(args.base)
    config.model.params.vae_decode_batch_size = args.vae_decode_batch_size
    for embedder in config.model.params.conditioner_config.params.emb_models:
        if embedder.target == 'sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder':
            embedder.params.version = args.clip_ckpt
            break
    else:
        raise ValueError('OpenCLIP embedder missing')
    model = instantiate_from_config(config.model)
    state = torch.load(args.ckptpath, map_location='cpu')
    if 'deepspeed' in args.ckptpath:
        state = {k.replace('_forward_module.', ''): v for k, v in state.items()}
    else:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Rank {rank}: checkpoint missing={missing}, unexpected={unexpected}', flush=True)
    del state
    model.to(device).eval()
    identity = checkpoint_identity([Path(args.ckptpath), Path(args.clip_ckpt)])
    scenes = select_scenes(load_generation_set(args.data_root, args.generation_set), args.pilot, args.scene_indices, args.limit)
    failures = 0
    for scene in shard_scenes(scenes, rank, world_size):
        final = args.prediction_root / 'panacea' / scene.dataset / 'scenes' / scene.scene_id
        if validate_completed_scene(final, 'panacea', scene).valid:
            require_matching_renderer(final, args.reference_mode, args.chronological, args.ordered_targets)
            print(f'Rank {rank} skip {scene.index}', flush=True)
            continue
        phase = 'load_inputs'
        try:
            torch.manual_seed(args.seed + scene.index)
            inputs = load_scene_inputs(scene)
            condition = None
            l1s, hashes, published, records = [], [], [], []
            with ScenePublisher(args.prediction_root, 'panacea', scene, os.environ['SLURM_JOB_ID'], str(rank)) as publisher:
                for wi, window in enumerate(reference_windows(args.reference_mode)):
                    print(f'Rank {rank} scene {scene.index} window {wi + 1}/12', flush=True)
                    phase = 'build_controls'
                    batch = build_panacea_window(inputs, window, condition)
                    reference_hash = hashlib.sha256(batch['jpg'][0].numpy().tobytes()).hexdigest()
                    if args.reference_mode == 'history-anchor' and records and reference_hash != records[0]['reference_sha256']:
                        raise ValueError('History reference unexpectedly changed between windows')
                    if args.ordered_targets:
                        model_batch = {key:batch[key].roll(-1,0).unsqueeze(0).to(device) for key in ('jpg','cond_img','final_cond_zero')}
                        model_batch['txt']=[batch['txt']]
                    elif args.chronological:
                        model_batch = {key:batch[key].unsqueeze(0).to(device) for key in ('jpg','cond_img','final_cond_zero')}
                        model_batch.update(txt=[batch['txt']], reference_slot=0)
                    else:
                        model_batch = prepare_native_panacea_batch(batch, device)
                    phase = 'inference'
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        output = model.log_images(model_batch, sample_with_muldecoders=False, skip_reconstructions=True)['samples'].detach().float().cpu().clamp(-1, 1)
                    if tuple(output.shape) != (8, 3, 256, 3072):
                        raise ValueError(f'Unexpected generated shape {output.shape}')
                    if args.ordered_targets:
                        output=output.roll(1,0)
                    elif not args.chronological:
                        output = restore_panacea_timeline(output)
                    del model_batch
                    a = torch.stack(list(split_panacea_views(output[0]).values()))
                    b = torch.stack(list(split_panacea_views(batch['jpg'][0]).values()))
                    l1s.append(float((F.interpolate(a, size=(224, 400), mode='bilinear', align_corners=False) - F.interpolate(b, size=(224, 400), mode='bilinear', align_corners=False)).abs().mean() / 2))
                    phase = 'write_frames'
                    for slot, frame in enumerate(window.output_frames, start=1):
                        hashes.append(hashlib.sha256(output[slot].numpy().tobytes()).hexdigest())
                        write_generated_frame(publisher, frame, output[slot])
                        published.append(frame)
                    print(f'Rank {rank} scene {scene.index} window {wi + 1}: wrote frames {window.output_frames[0]}..{window.output_frames[-1]}, condition L1={l1s[-1]:.5f}', flush=True)
                    condition = (batch['jpg'][0] if args.reference_mode == 'history-anchor' else output[len(window.output_frames)]).clone()
                    records.append({'condition_frame': window.condition_frame, 'reference_source': 'history_rgb' if args.reference_mode == 'history-anchor' or wi == 0 else 'generated_rgb', 'reference_sha256': reference_hash, 'native_frame_indices': list(batch['control_frame_indices'][1:]+batch['control_frame_indices'][:1]) if args.ordered_targets else list(batch['control_frame_indices'] if args.chronological else reversed(batch['control_frame_indices'])), 'native_condition_slot': 0 if args.chronological else 7, 'control_frames': list(window.control_frames), 'output_frames': list(window.output_frames), 'condition_l1': l1s[-1]})
                    del batch, output
                phase = 'pilot_condition_gate'
                order_valid = published == list(range(10, 90))
                if not order_valid or (args.pilot and (max(l1s) > .20 or len(set(hashes)) < 2)):
                    raise ValueError(f'Condition gate: max L1={max(l1s)}, hashes={len(set(hashes))}, temporal_order={order_valid}')
                phase = 'publish'
                publisher.publish({'checkpoint': identity, 'seed': args.seed + scene.index,
                    'inference_settings': {'condition_renderer': RENDERER_VERSION, 'temporal_adapter': 'ordered-targets-last-reference-v1' if args.ordered_targets else 'first-reference-chronological-v1' if args.chronological else TEMPORAL_ADAPTER_VERSION, 'reference_mode': args.reference_mode, 'frame_length': 8, 'native_view_size': [512, 256], 'sample_steps': 25, 'guidance_scale': 5.0, 'vae_decode_batch_size': args.vae_decode_batch_size, 'skip_reconstructions': True},
                    'history_frames_used': [9], 'repeated_reference_slots': [],
                    'used_future_ego_trajectory': True, 'used_future_agent_trajectories': True,
                    'available_map_channels': list(inputs.map_condition.available_channels),
                    'unavailable_map_channels': list(inputs.map_condition.unavailable_channels),
                    'dropped_agent_categories': sorted({a.category for agents in inputs.agents_by_frame.values() for a in agents if class_index(a.category) is None}),
                    'temporal_strategy': {'conditioning': 'ordered_targets_last_reference' if args.ordered_targets else 'first_reference' if args.chronological else 'last_reference_reversed', 'reference_mode': args.reference_mode, 'external_order': 'chronological', 'native_order': 'future_forward_order_then_reference' if args.ordered_targets else 'reference_then_future' if args.chronological else 'future_reverse_order_then_reference', 'windows': records},
                    'pilot_condition_l1': max(l1s), 'pilot_distinct_generated_hashes': len(set(hashes)), 'pilot_temporal_order_valid': order_valid})
                print(f'Rank {rank} published scene {scene.index}', flush=True)
        except Exception as error:
            traceback.print_exc()
            write_failure(args.prediction_root, 'panacea', scene, phase, error, rank, os.environ['SLURM_JOB_ID'])
            failures += 1
            torch.cuda.empty_cache()
    count = torch.tensor(failures, device=device)
    dist.all_reduce(count)
    failed = int(count.item())
    dist.destroy_process_group()
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())

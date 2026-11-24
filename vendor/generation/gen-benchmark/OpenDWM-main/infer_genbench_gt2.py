"""Two-rank FSDP GT2 inference with transactional rank-zero publication."""
import argparse
import json
import os
from pathlib import Path


def require_renderer_identity(scene_path):
    from dwm.datasets.genbench_gt2_renderer import RENDERER_VERSION
    status = json.loads((Path(scene_path) / "status.json").read_text())
    if status.get("condition_renderer") != RENDERER_VERSION:
        raise ValueError(
            "Completed output renderer identity differs from {}: {}. "
            "Archive the old output before rerunning; it will not be reused or overwritten.".format(
                RENDERER_VERSION, scene_path))


def identity_paths(base_model, checkpoint):
    base = Path(base_model)
    return [Path(checkpoint)] + [base / name for name in (
        "text_encoder/model.safetensors", "text_encoder_2/model.safetensors",
        "text_encoder_3/model-00001-of-00002.safetensors", "text_encoder_3/model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors")]


def runtime_config(adapter_path, base_model, checkpoint, seed):
    adapter = json.loads(Path(adapter_path).read_text())
    repo = Path(__file__).resolve().parent
    config = json.loads((repo / adapter.pop("base_config")).read_text())
    config.update(device="cuda", ddp_backend="nccl", generator_seed=seed)
    pipeline = config["pipeline"]
    pipeline["pretrained_model_name_or_path"] = str(base_model)
    pipeline["model_checkpoint_path"] = str(checkpoint)
    common = pipeline["common_config"]
    common["distribution_framework"] = "fsdp"
    common["text_encoder_load_args"].pop("quantization_config", None)
    def wrapper(classes):
        return {"sharding_strategy": {"_class_name": "torch.distributed.fsdp.ShardingStrategy", "value": 1},
                "auto_wrap_policy": {"_class_name": "torch.distributed.fsdp.wrap.ModuleWrapPolicy", "module_classes": [{"_class_name": "get_class", "class_name": c} for c in classes]}}
    common["ddp_wrapper_settings"] = wrapper(["diffusers.models.attention.JointTransformerBlock", "dwm.models.crossview_temporal.VTSelfAttentionBlock"])
    common["ddp_wrapper_settings"]["mixed_precision"] = {"_class_name": "torch.distributed.fsdp.MixedPrecision", "param_dtype": {"_class_name": "get_class", "class_name": "torch.float16"}}
    common["t5_fsdp_wrapper_settings"] = wrapper(["transformers.models.t5.modeling_t5.T5Block"])
    pipeline["inference_config"].update(adapter)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "prediction-root", "adapter-config", "base-model", "checkpoint"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--generation-set", default="balanced_500")
    parser.add_argument("--seed", type=int, default=3407)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pilot", action="store_true")
    selection.add_argument("--scene-indices")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Inference requires a Slurm L40 allocation")
    import torch
    import dwm.common
    from dwm.datasets.genbench_gt2 import GenBenchGT2OpenDWMDataset
    from genbench_gt2.dataset import load_generation_set
    from genbench_gt2.selection import select_scenes
    from genbench_gt2.constants import CAMERAS, FUTURE_FRAMES
    from genbench_gt2.output import ScenePublisher, checkpoint_identity, validate_completed_scene, write_failure
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size not in (2, 4):
        raise RuntimeError("FSDP requires a two- or four-process world")
    if torch.cuda.device_count() != world_size or any("L40" not in torch.cuda.get_device_name(i) for i in range(world_size)):
        raise RuntimeError(f"Exactly {world_size} visible L40 GPUs are required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl")
    if torch.distributed.get_world_size() != world_size:
        raise RuntimeError(f"FSDP requires world size {world_size}")
    device = torch.device("cuda", rank)
    config = runtime_config(args.adapter_config, args.base_model, args.checkpoint, args.seed)
    pipeline = dwm.common.create_instance_from_config(config["pipeline"], output_path=None, config=config, device=device)
    scenes = select_scenes(load_generation_set(args.data_root, args.generation_set), args.pilot, args.scene_indices, args.limit)
    dataset = GenBenchGT2OpenDWMDataset(args.data_root, args.generation_set, [s.index for s in scenes])
    identity = checkpoint_identity(identity_paths(args.base_model, args.checkpoint))
    failures = 0
    def sync(value):
        state = torch.tensor([value], device=device, dtype=torch.int32)
        torch.distributed.broadcast(state, 0)
        return bool(state.item())
    for scene in scenes:
        final = args.prediction_root / "opendwm" / scene.dataset / "scenes" / scene.scene_id
        if sync(validate_completed_scene(final, "opendwm", scene).valid if rank == 0 else False):
            require_renderer_identity(final)
            continue
        print("rank={} generating scene={} {}".format(rank, scene.index, scene.relative_path), flush=True)
        error = None
        try:
            item = dataset.build_batch(scene)
        except Exception as exc:
            error = exc
        ready = torch.tensor([int(error is None)], device=device)
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            if rank == 0:
                write_failure(args.prediction_root, "opendwm", scene, "controls", error or RuntimeError("Other rank failed building controls"), rank, os.environ["SLURM_JOB_ID"])
            failures += 1
            continue
        context = item.pop("prediction_context")
        batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in item.items()}
        pipeline.generator.manual_seed(args.seed + scene.index)
        # FSDP model calls must execute on both ranks in identical order.
        try:
            with torch.no_grad():
                output = pipeline.autoregressive_inference_pipeline((1, pipeline.get_latent_sequence_length(19), 6, pipeline.vae.config.latent_channels, 32, 56), batch, "pt")
            images = output["images"].unflatten(0, (80, 6))
            assert images.shape[:3] == (80, 6, 3)
        except Exception as exc:
            if rank == 0:
                write_failure(args.prediction_root, "opendwm", scene, "inference", exc, rank, os.environ["SLURM_JOB_ID"])
            # A failed collective may poison FSDP; torchrun terminates the peer
            # and the next submission resumes from validated published scenes.
            raise
        success = True
        if rank == 0:
            try:
                with ScenePublisher(args.prediction_root, "opendwm", scene, os.environ["SLURM_JOB_ID"], "rank0") as publisher:
                    for view, camera in enumerate(CAMERAS):
                        for frame in FUTURE_FRAMES:
                            publisher.write_frame(camera, frame, images[frame-10, view].permute(1, 2, 0), value_range="zero_one")
                    publisher.publish(dict(context, checkpoint=identity, seed=args.seed + scene.index,
                        inference_settings={"guidance_scale": 4, "inference_steps": 40, "fsdp_world_size": world_size},
                        history_frames_used=[7, 8, 9], repeated_reference_slots=[], used_future_ego_trajectory=True,
                        used_future_agent_trajectories=True, temporal_strategy=dict(config["pipeline"]["inference_config"], iteration_starts=[0,16,32,48,64])))
                print("PUBLISHED scene={} images=480".format(scene.index), flush=True)
            except Exception as exc:
                write_failure(args.prediction_root, "opendwm", scene, "publication", exc, rank, os.environ["SLURM_JOB_ID"])
                success = False
        if not sync(success):
            failures += 1
        del item, batch, output, images
    total = torch.tensor([failures], device=device)
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.destroy_process_group()
    return int(total.item() > 0)


if __name__ == "__main__":
    raise SystemExit(main())

import os
import gc
import sys
import time
import copy
import json
from pprint import pformat
from datetime import timedelta
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
USE_NPU = False
import magicdrivedit.utils.module_contrib

import colossalai
import torch.distributed as dist
from torch.utils.data import Subset
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from mmengine.runner import set_random_seed
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from mmcv.parallel import DataContainer

from magicdrivedit.acceleration.parallel_states import (
    set_sequence_parallel_group,
    get_sequence_parallel_group,
    set_data_parallel_group,
    get_data_parallel_group
)
from magicdrivedit.datasets import save_sample
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.models.text_encoder.t5 import text_preprocessing
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.config_utils import parse_configs, define_experiment_workspace, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from magicdrivedit.utils.inference_utils import (
    apply_mask_strategy,
    get_save_path_name,
    concat_6_views_pt,
    add_null_condition,
)
from magicdrivedit.utils.misc import (
    reset_logger,
    is_distributed,
    is_main_process,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from magicdrivedit.utils.train_utils import sp_vae
from torch.distributed import barrier


TILING_PARAM = {
    "default": dict(),  # it is designed for CogVideoX's 720x480, 4.5 GB
    "384": dict(  # about 14.2 GB
        tile_sample_min_height = 384,  # should be 48n
        tile_sample_min_width = 720,  # should be 40n
    ),
}


def set_omegaconf_key_value(cfg, key, value):
    p, m = key.rsplit(".", 1)
    node = cfg
    for pk in p.split("."):
        node = getattr(node, pk)
    node[m] = value


def make_box_latent_sampler(model, mode, seed):
    """Keep object identity independent of diffusion noise in both render paths."""
    if mode not in {"random", "zero"}:
        raise ValueError(f"Unsupported box latent mode: {mode}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(n_boxes):
        latent = model.sample_box_latent(n_boxes, generator=generator)
        if latent is not None and mode == "zero":
            latent.zero_()
        return latent

    return sample


def build_conditioning_audit(z, reference_latents, masks, model_args, *, rgb_frames,
                             history_rgb_frames, sp_size, sample_seed,
                             box_latent_seed, box_latent_mode):
    """Summarize the actual scheduler inputs without retaining tensor payloads."""
    finite = []

    def summary(value):
        if isinstance(value, dict):
            return {key: summary(item) for key, item in value.items()}
        if not torch.is_tensor(value):
            return None
        is_finite = bool(torch.isfinite(value).all().item())
        finite.append(is_finite)
        return dict(shape=list(value.shape), dtype=str(value.dtype), finite=is_finite)

    required = ("maps", "bbox", "cams", "rel_pos", "fps", "height", "width", "num_frames")
    conditions = {key: summary(model_args.get(key)) for key in required}
    present = all(model_args.get(key) is not None for key in required)
    bbox = model_args.get("bbox")
    present = present and isinstance(bbox, dict) and all(
        torch.is_tensor(bbox.get(key)) for key in ("bboxes", "classes", "masks"))
    temporal_tensors = [model_args.get(key) for key in ("maps", "cams", "rel_pos")]
    if isinstance(bbox, dict):
        temporal_tensors.extend(bbox.values())
    temporal_aligned = all(torch.is_tensor(value) and value.ndim >= 2
                           and value.shape[1] == rgb_frames for value in temporal_tensors)
    frame_count = model_args.get("num_frames")
    temporal_aligned = temporal_aligned and torch.is_tensor(frame_count) and bool(
        (frame_count == rgb_frames).all().item())
    reference_length = 0 if reference_latents is None else int(reference_latents.shape[2])
    reference_aligned = reference_length <= z.shape[2]
    if reference_latents is not None:
        reference_aligned = reference_aligned and torch.equal(
            z[:, :, :reference_length], reference_latents)
    frozen = [] if masks is None else (masks == 0).sum(dim=1).cpu().tolist()
    if masks is not None:
        reference_aligned = reference_aligned and tuple(masks.shape) == (z.shape[0], z.shape[2])
        reference_aligned = reference_aligned and bool((masks[:, :reference_length] == 0).all())
        reference_aligned = reference_aligned and bool((masks[:, reference_length:] == 1).all())
    elif reference_length:
        reference_aligned = False
    report = dict(boundary="scheduler_input", temporal_strategy="full_sequence",
                  rgb_frames=rgb_frames, history_rgb_frames=history_rgb_frames,
                  latent_frames=int(z.shape[2]), reference_latent_frames=reference_length,
                  frozen_latents_per_sample=frozen, sequence_parallel_size=sp_size,
                  sample_seed=sample_seed, box_latent_seed=box_latent_seed,
                  box_latent_mode=box_latent_mode, conditions=conditions,
                  latent=summary(z), reference=summary(reference_latents), mask=summary(masks),
                  required_conditions_present=bool(present),
                  temporal_conditioning_aligned=bool(temporal_aligned),
                  reference_prefix_and_mask_aligned=bool(reference_aligned))
    fps = model_args.get("fps")
    report["fps"] = fps.detach().cpu().flatten().tolist() if torch.is_tensor(fps) else None
    report["all_finite"] = all(finite)
    report["valid"] = bool(present and temporal_aligned and reference_aligned and all(finite))
    return report


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    # == dataset config ==
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
        for (res, data_cfg_name), overrides in zip(
                cfg.data_cfg_names, cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)):
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)
    if cfg.get("use_train", False):
        cfg.dataset = dataset
        tag = cfg.get("tag", "")
        cfg.tag = "train" if tag == "" else f"{tag}_train"
    else:
        cfg.dataset = val_dataset
    # set img_collate_param
    if hasattr(cfg.dataset, "img_collate_param"):
        cfg.dataset.img_collate_param.is_train = False  # Important!
    else:
        for d in cfg.dataset.cfg:
            d[1].img_collate_param.is_train = False  # Important!
    cfg.batch_size = 1
    # for lower cpu memory in dataloading
    cfg.ignore_ori_imgs = cfg.get("ignore_ori_imgs", True)
    if cfg.ignore_ori_imgs:
        cfg.dataset.drop_ori_imgs = True

    # for lower gpu memory in vae decoding
    cfg.vae_tiling = cfg.get("vae_tiling", None)

    # edit annotations
    if cfg.get("allow_class", None) != None:
        cfg.dataset.allow_class = cfg.allow_class
    if cfg.get("del_box_ratio", None) != None:
        cfg.dataset.del_box_ratio = cfg.del_box_ratio
    if cfg.get("drop_nearest_car", None) != None:
        cfg.dataset.drop_nearest_car = cfg.drop_nearest_car

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if USE_NPU:  # disable some kernels
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == init distributed env ==
    if is_distributed():
        # colossalai.launch_from_torch({})
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        cfg.sp_size = dist.get_world_size()
    else:
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0,
            init_method="tcp://localhost:12355")
        cfg.sp_size = 1
    coordinator = DistCoordinator()
    if cfg.sp_size > 1:
        DP_AXIS, SP_AXIS = 0, 1
        dp_size = dist.get_world_size() // cfg.sp_size
        pg_mesh = ProcessGroupMesh(dp_size, cfg.sp_size)
        dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
        set_sequence_parallel_group(sp_group)
        print(f"Using sp_size={cfg.sp_size}")
    else:
        # TODO: sequence_parallel_group unset!
        dp_group = dist.group.WORLD
    set_data_parallel_group(dp_group)
    enable_sequence_parallelism = cfg.sp_size > 1
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init exp_dir ==
    cfg.outputs = cfg.get("outputs", "outputs/test")
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    cfg.save_dir = os.path.join(exp_dir, "generation")
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger ==
    logger = reset_logger(exp_dir)
    if cfg.get("print_config", True):
        logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    cfg.cpu_offload = cfg.get("cpu_offload", False)
    init_device = "cpu" if cfg.cpu_offload else device
    # == build text-encoder and vae ==
    # NOTE: set to true/false,
    # https://github.com/huggingface/transformers/issues/5486
    # if the program gets stuck, try set it to false
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.text_encoder, MODELS, device=init_device)
    vae = build_module(cfg.vae, MODELS).to(init_device, dtype).eval()
    if cfg.vae_tiling:
        vae.module.enable_tiling(**TILING_PARAM[str(cfg.vae_tiling)])
        logger.info(f"VAE Tiling is enabled with {TILING_PARAM[str(cfg.vae_tiling)]}")

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(init_device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    last_hook = None
    if cfg.cpu_offload:
        from accelerate.big_modeling import cpu_offload_with_hook

        # Keep T5 on CPU to avoid VRAM spikes during text encoding.
        text_encoder.t5.model.to("cpu")
        text_encoder.t5.device = torch.device("cpu")
        model.to("cpu")
        vae.to("cpu")
        model, hook_model = cpu_offload_with_hook(model, device)
        vae, last_hook = cpu_offload_with_hook(vae, device, hook_model)
    # == load prompts ==
    # prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)

    # == prepare arguments ==
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)







    # == Iter over all samples ==
    with open(cfg.iopth_pth, 'r') as f:
        iopths = f.readlines()
    import pickle
    # batch = pickle.load(open("debug/batch_125_4blend_per_ato_mask.pkl", "rb"))
    import numpy as np
    import datetime
    # np.random.seed(datetime.datetime.now().microsecond)
    # rand_perm = np.random.permutation(len(iopths) // 2)
    for bidx in range(len(iopths) // 2):
        # rta_st_render_ts = time.time()
        # bidx = pi
        # batch = pickle.load(open(f"obatches/batch_{bidx:04d}.pkl", "rb"))
        # batch = pickle.load(open(f"ebatches/batch_0120_14_manu.pkl", "rb"))
        in_path = iopths[bidx * 2].strip()
        out_path = iopths[bidx * 2 + 1].strip()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) == 0 and is_main_process():
            logger.warning("Found empty output placeholder. Regenerating: %s", out_path)
        # create an empty file
        if is_distributed():
            barrier()
        if is_main_process():
            out_dir = os.path.dirname(out_path)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as f:
                pass
        batch = pickle.load(open(in_path, "rb"))
        # Optional per-batch controls are used by short prefix evaluations to
        # select an identity prior before paying for a full autoregressive run.
        # Ordinary batches retain deterministic defaults.
        box_latent_mode = str(
            batch.pop("box_latent_mode", cfg.get("box_latent_mode", "random"))
        )
        box_latent_seed = int(
            batch.pop("box_latent_seed", cfg.get("box_latent_seed", 1024))
        )
        sample_seed = int(
            batch.pop("sample_seed", cfg.get("sample_seed", 1024))
        )
        if box_latent_mode not in {"random", "zero"}:
            raise ValueError(
                f"box_latent_mode must be random or zero, got {box_latent_mode!r}"
            )
        # print("Infer: ", in_path)
        if cfg.ignore_ori_imgs:
            B, T, NC = 1, *batch["pixel_values_shape"][0].tolist()[:2]
            latent_size = vae.get_latent_size(
                (T, *batch["pixel_values_shape"][0].tolist()[-2:]))
        else:
            B, T, NC = batch["pixel_values"].shape[:3]
            latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
            # == prepare batch prompts ==
            x = batch.pop("pixel_values").to(device, dtype)
            x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
        y = batch.pop("captions")[0]  # B, just take first frame
        maps = batch.pop("bev_map_with_aux").to(device, dtype)  # B, T, C, H, W
        bbox = batch.pop("bboxes_3d_data")
        # B len list (T, NC, len, 8, 3)
        bbox = [bbox_i.data for bbox_i in bbox]
        # B, T, NC, len, 8, 3
        # TODO: `bbox` may have some redundancy on `NC` dim.
        # NOTE: we reshape the data later!
        bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
        # B, T, NC, 3, 7
        cams = batch.pop("camera_param").to(device, dtype)
        cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
        rel_pos = batch.pop("frame_emb").to(device, dtype)
        rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4

        # variable for inference
        batch_prompts = y
        # Optionally anchor the first latent frames with player-visible RGB.
        # Shape: B, NC, C, T_ref, H, W. Batches without this key retain the
        # original unconditional behavior.
        reference_images = batch.pop("reference_images", None)
        has_reference_images = reference_images is not None
        reference_rgb_length = int(reference_images.shape[3]) if has_reference_images else 0
        if has_reference_images:
            reference_images = reference_images.to(device, dtype)
            reference_images = rearrange(
                reference_images, "B NC C T H W -> (B NC) C T H W"
            )
            reference_latents = vae.encode(reference_images)
            reference_latents = rearrange(
                reference_latents, "(B NC) C T H W -> B (C NC) T H W", NC=NC
            )
            refs = [[reference_latents[i]] for i in range(B)]
            reference_length = int(reference_latents.shape[2])
            ms = [f"0,0,0,0,{reference_length},0"] * B
            del reference_images
        else:
            ms = [""] * len(y)
            refs = [""] * len(y)

        # == model input format ==
        model_args = {}
        model_args["maps"] = maps
        model_args["bbox"] = bbox
        model_args["cams"] = cams
        model_args["rel_pos"] = rel_pos
        model_args["fps"] = batch.pop('fps')
        model_args['drop_cond_mask'] = torch.ones((B))  # camera
        model_args['drop_frame_mask'] = torch.ones((B, T))  # box & rel_pos
        model_args["height"] = batch.pop("height")
        model_args["width"] = batch.pop("width")
        model_args["num_frames"] = batch.pop("num_frames")
        model_args = move_to(model_args, device=device, dtype=dtype)
        # no need to move these
        model_args["mv_order_map"] = cfg.get("mv_order_map")
        model_args["t_order_map"] = cfg.get("t_order_map")

        # == Iter over number of sampling for one prompt ==
        save_fps = int(model_args['fps'][0])
        for ns in range(num_sample):
            gc.collect()
            torch.cuda.empty_cache()
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=y[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=ns,
                )
                for idx in range(len(y))
            ]
            if cfg.get("force_daytime", False):
                batch_prompts[0] = batch_prompts[0].lower()
                batch_prompts[0] = "Daytime. " + batch_prompts[0]
                # exclude rain
                batch_prompts[0] = batch_prompts[0].replace("rain", "sunny")
                batch_prompts[0] = batch_prompts[0].replace("water reflections", "")
                batch_prompts[0] = batch_prompts[0].replace("reflections in water", "")
                batch_prompts[0] = batch_prompts[0].replace(" with umbrellas", "")
                batch_prompts[0] = batch_prompts[0].replace(" with umbrella", "")
                batch_prompts[0] = batch_prompts[0].replace(" holds umbrella", "")
                # exclude night
                batch_prompts[0] = batch_prompts[0].replace("night", "")
                batch_prompts[0] = batch_prompts[0].replace(" in dark", "")
                batch_prompts[0] = batch_prompts[0].replace(" dark", "")
                batch_prompts[0] = batch_prompts[0].replace(" difficult lighting", "")
                # city
                batch_prompts[0] = batch_prompts[0].replace("boston-seaport", "singapore-onenorth")
                batch_prompts[0] = batch_prompts[0].replace("singapore-hollandvillage", "singapore-onenorth")
                neg_prompts = ["Rain, Night, water reflections, umbrella"]
            elif cfg.get("force_rainy", False):
                if "rain" not in batch_prompts[0].lower():
                    batch_prompts[0] = "A driving scene image at boston-seaport. Rain. water reflections."
                neg_prompts = ["Daytime. night, onenorth, queenstown"]
            elif cfg.get("force_night", False):
                if "night" not in batch_prompts[0].lower():
                    batch_prompts[0] = "A driving scene image at singapore-hollandvillage. Night, congestion. difficult lighting. very dark."
                neg_prompts = ["Daytime. rain, boston-seaport"]
            else:
                neg_prompts = None

            video_clips = []
            # == sampling ==
            # Keep the diffusion noise independently controllable from the
            # per-object box latent.  Prefix evaluation can then reject a
            # visually unstable rollout without changing any geometry or
            # vehicle identity conditioning.
            torch.manual_seed(sample_seed + ns)
            logger.info("Diffusion sample seed: %d", sample_seed + ns)
            sample_box_latent = make_box_latent_sampler(
                model, box_latent_mode, box_latent_seed + ns)
            full_audit = None
            if (
                has_reference_images
                and T >= 17
                and cfg.get("autoregressive_reference", False)
            ):
                # Legacy continuation mode. Its three frozen latents out of
                # five differ from training, which never masks 17-frame clips.
                # The full-sequence path below avoids this short-window mismatch.
                chunk_length, overlap = 17, 9
                stride = chunk_length - overlap
                if (T - chunk_length) % stride != 0:
                    raise ValueError(
                        f"Autoregressive length {T} is incompatible with "
                        f"window={chunk_length}, overlap={overlap}"
                    )
                # A box latent acts as an object's appearance/identity token.
                # Sampling it independently for every window makes the same
                # track liable to change appearance at each eight-frame seam.
                # Sample once for the full rollout and slice that persistent
                # identity together with the geometry below.
                persistent_bbox = None
                if bbox is not None:
                    logger.info(
                        "Persistent box latent: mode=%s seed=%d",
                        box_latent_mode,
                        box_latent_seed + ns,
                    )
                    persistent_bbox = add_box_latent(
                        bbox, B, NC, T, sample_box_latent
                    )
                current_reference_latents = reference_latents
                stitched_samples = None
                for chunk_start in range(0, T - chunk_length + 1, stride):
                    chunk_end = chunk_start + chunk_length
                    logger.info(
                        "Autoregressive window [%d, %d) / %d",
                        chunk_start,
                        chunk_end,
                        T,
                    )
                    chunk_args = {
                        "maps": model_args["maps"][:, chunk_start:chunk_end],
                        "cams": model_args["cams"][:, chunk_start:chunk_end],
                        "fps": model_args["fps"],
                        "drop_cond_mask": model_args["drop_cond_mask"],
                        "drop_frame_mask": torch.ones(
                            (B, chunk_length), device=device, dtype=dtype
                        ),
                        "height": model_args["height"],
                        "width": model_args["width"],
                        "num_frames": torch.full_like(
                            model_args["num_frames"], chunk_length
                        ),
                        "mv_order_map": model_args["mv_order_map"],
                        "t_order_map": model_args["t_order_map"],
                    }
                    rel_window = model_args["rel_pos"][:, chunk_start:chunk_end]
                    # The inference model runs in BF16, while torch.linalg.inv
                    # requires FP32 on CUDA.  Normalise each window against its
                    # own first Ego pose before casting back to model precision.
                    rel_start_inv = torch.linalg.inv(
                        model_args["rel_pos"][
                            :, chunk_start:chunk_start + 1
                        ].float()
                    ).to(dtype=rel_window.dtype)
                    chunk_args["rel_pos"] = torch.matmul(rel_window, rel_start_inv)

                    if persistent_bbox is not None:
                        chunk_bbox = {
                            key: value[:, chunk_start:chunk_end]
                            for key, value in persistent_bbox.items()
                        }
                        # sample_box_latent creates FP32 tensors. Match the
                        # original non-windowed path by moving every bbox field
                        # to the model dtype after slicing the shared latent.
                        chunk_args["bbox"] = move_to(
                            {
                                key: rearrange(
                                    value, "B T NC ... -> (B NC) T ..."
                                )
                                for key, value in chunk_bbox.items()
                            },
                            device=device,
                            dtype=dtype,
                        )

                    if cfg.scheduler.type == "dpm-solver" and cfg.scheduler.cfg_scale == 1.0 or (
                        cfg.scheduler.type in ["rflow-slice",]
                    ):
                        chunk_sample_args = copy.deepcopy(chunk_args)
                    else:
                        chunk_sample_args = add_null_condition(
                            copy.deepcopy(chunk_args),
                            model.camera_embedder.uncond_cam.to(device),
                            model.frame_embedder.uncond_cam.to(device),
                            prepend=(cfg.scheduler.type == "dpm-solver"),
                        )

                    chunk_latent_size = vae.get_latent_size(
                        (
                            chunk_length,
                            int(model_args["height"][0]),
                            int(model_args["width"][0]),
                        )
                    )
                    z = torch.randn(
                        len(batch_prompts),
                        vae.out_channels * NC,
                        *chunk_latent_size,
                        device=device,
                        dtype=dtype,
                    )
                    chunk_refs = [
                        [current_reference_latents[i]] for i in range(B)
                    ]
                    chunk_reference_length = int(current_reference_latents.shape[2])
                    chunk_ms = [
                        f"0,0,0,0,{chunk_reference_length},0"
                    ] * B
                    masks = apply_mask_strategy(z, chunk_refs, chunk_ms, 0, align=None)
                    chunk_samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts,
                        neg_prompts=neg_prompts,
                        device=device,
                        additional_args=chunk_sample_args,
                        progress=verbose >= 1,
                        mask=masks,
                    )
                    chunk_samples = rearrange(
                        chunk_samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC
                    )
                    if cfg.sp_size > 1:
                        chunk_samples = sp_vae(
                            chunk_samples.to(dtype),
                            partial(
                                vae.decode,
                                num_frames=chunk_sample_args["num_frames"],
                            ),
                            get_sequence_parallel_group(),
                        )
                    else:
                        chunk_samples = vae.decode(
                            chunk_samples.to(dtype),
                            num_frames=chunk_sample_args["num_frames"],
                        )
                    chunk_samples = rearrange(
                        chunk_samples, "(B NC) C T ... -> B NC C T ...", NC=NC
                    )
                    if stitched_samples is None:
                        stitched_samples = chunk_samples
                    else:
                        stitched_samples = torch.cat(
                            [stitched_samples, chunk_samples[:, :, :, overlap:]], dim=3
                        )

                    if chunk_end < T:
                        next_reference = rearrange(
                            chunk_samples[:, :, :, -overlap:],
                            "B NC C T H W -> (B NC) C T H W",
                        )
                        current_reference_latents = vae.encode(next_reference)
                        current_reference_latents = rearrange(
                            current_reference_latents,
                            "(B NC) C T H W -> B (C NC) T H W",
                            NC=NC,
                        )
                    del chunk_samples, chunk_sample_args
                    gc.collect()
                    torch.cuda.empty_cache()
                samples = stitched_samples
            else:
                z = torch.randn(len(batch_prompts), vae.out_channels * NC, *latent_size, device=device, dtype=dtype)

                # == sample box ==
                if bbox is not None:
                    bbox = add_box_latent(bbox, B, NC, T, sample_box_latent)
                    new_bbox = {}
                    for k, v in bbox.items():
                        new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")
                    model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)

                if cfg.scheduler.type == "dpm-solver" and cfg.scheduler.cfg_scale == 1.0 or (
                    cfg.scheduler.type in ["rflow-slice",]
                ):
                    _model_args = copy.deepcopy(model_args)
                else:
                    _model_args = add_null_condition(
                        copy.deepcopy(model_args),
                        model.camera_embedder.uncond_cam.to(device),
                        model.frame_embedder.uncond_cam.to(device),
                        prepend=(cfg.scheduler.type == "dpm-solver"),
                    )

                masks = apply_mask_strategy(z, refs, ms, 0, align=None)
                if is_main_process():
                    full_audit = build_conditioning_audit(
                        z, reference_latents if has_reference_images else None,
                        masks, _model_args, rgb_frames=T, history_rgb_frames=reference_rgb_length,
                        sp_size=cfg.sp_size, sample_seed=sample_seed + ns,
                        box_latent_seed=box_latent_seed + ns, box_latent_mode=box_latent_mode)
                    full_audit.update(scheduler=cfg.scheduler.type,
                                      sampling_steps=cfg.scheduler.get("num_sampling_steps"),
                                      cfg_scale=cfg.scheduler.get("cfg_scale"),
                                      checkpoint=cfg.model.get("from_pretrained"),
                                      completed=False)
                    with open(out_path + ".conditioning.json", "w") as audit_file:
                        json.dump(full_audit, audit_file, indent=2)
                    logger.info("Full-sequence conditioning: RGB=%d history=%d latents=%d frozen=%s FPS=%s SP=%d valid=%s",
                                T, reference_rgb_length, full_audit["latent_frames"],
                                full_audit["frozen_latents_per_sample"], full_audit["fps"],
                                cfg.sp_size, full_audit["valid"])
                    if not full_audit["valid"]:
                        raise ValueError(f"Invalid full-sequence conditioning; see {out_path}.conditioning.json")
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    neg_prompts=neg_prompts,
                    device=device,
                    additional_args=_model_args,
                    progress=verbose >= 1,
                    mask=masks,
                )
                if full_audit is not None:
                    full_audit["diffusion_output_finite"] = bool(torch.isfinite(samples).all().item())
                samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
                if cfg.sp_size > 1:
                    samples = sp_vae(
                        samples.to(dtype),
                        partial(vae.decode, num_frames=_model_args["num_frames"]),
                        get_sequence_parallel_group(),
                    )
                else:
                    samples = vae.decode(samples.to(dtype), num_frames=_model_args["num_frames"])
                samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
                if full_audit is not None:
                    full_audit["decoded_shape"] = list(samples.shape)
                    full_audit["decoded_finite"] = bool(torch.isfinite(samples).all().item())
                    full_audit["completed"] = bool(full_audit["diffusion_output_finite"]
                                                    and full_audit["decoded_finite"]
                                                    and samples.shape[3] == T)
                    with open(out_path + ".conditioning.json", "w") as audit_file:
                        json.dump(full_audit, audit_file, indent=2)
                    if not full_audit["completed"]:
                        raise ValueError(f"Invalid full-sequence output; see {out_path}.conditioning.json")
            if cfg.cpu_offload and last_hook is not None:
                last_hook.offload()
            if is_main_process():
                # pickle.dump(samples, open(f"esamples_{bidx:04d}_14_manu.pkl", "wb"))
                pickle.dump(samples, open(out_path, "wb"))
                vid_samples = []
                for sample in samples:
                    vid_samples.append(
                        concat_6_views_pt(sample, oneline=False)
                    )
                samples = torch.stack(vid_samples, dim=0)
                video_clips.append(samples)
                del vid_samples
            del samples
            coordinator.block_all()

            # == save samples ==
            torch.cuda.empty_cache()
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 1:
                        logger.info(f"Prompt: {batch_prompt}")
                        if neg_prompts is not None:
                            logger.info(f"Neg-prompt: {neg_prompts[idx]}")
                    save_path = save_paths[idx]
                    video = [video_clips[0][idx]]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        high_quality=True,
                        verbose=verbose >= 2,
                        save_per_n_frame=cfg.get("save_per_n_frame", -1),
                        force_image=cfg.get("force_image", False),
                    )
            del video_clips
            coordinator.block_all()
        
        # save_gt
        if is_main_process() and not cfg.ignore_ori_imgs:
            torch.cuda.empty_cache()
            samples = rearrange(x, "(B NC) C T H W -> B NC C T H W", NC=NC)
            for idx, sample in enumerate(samples):
                vid_sample = concat_6_views_pt(sample, oneline=False)
                save_path = save_sample(
                    vid_sample,
                    fps=save_fps,
                    save_path=os.path.join(save_dir, f"gt_{start_idx + idx:04d}"),
                    high_quality=True,
                    verbose=verbose >= 2,
                    save_per_n_frame=cfg.get("save_per_n_frame", -1),
                    force_image=cfg.get("force_image", False),
                )
            del samples, vid_sample
        coordinator.block_all()
        start_idx += len(batch_prompts)
        # rta_ed_render_ts = time.time()
        # print("👿👿👿 rendering time: ", rta_ed_render_ts - rta_st_render_ts)

    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx - cfg.get("start_index", 0), save_dir)
    coordinator.destroy()


if __name__ == "__main__":
    main()

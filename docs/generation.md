# Participant inputs, CAT-K and future RGB

Run the commands below from a DriveAudit checkout. The input is a GenBench `nuscenes_ego_conditioned_v1` participant archive. CAT-K takes observed traffic and map context, predicts future actors and its own ego trajectory jointly, and exports the actors at the 80 participant target timestamps. The renderer uses the public ego and camera motion supplied in the archive. CAT-K's predicted ego trajectory is available for diagnostics.

## Install the native inference source offline

The native sources are tracked under [trajectory/CAT-K](../models/generation/trajectory/CAT-K) and [image](../models/generation/image), with one directory per renderer. The offline installer reconstructs the project paths expected by the upstream launchers. The upstream SHA-256 inventory is in `vendor/generation/bundles`. It contains official CAT-K Python source, seven renderer integrations and their model components, the relevant diffusers and 3D detection variants, and the editable dependencies used by the native environments. The installer copies these sources into a new engine root, applies this checkout's participant adapters, copies ABI-specific compiled extensions from a compatible build, and links the model assets. Every installed Python source file is a regular file under the new root.

Set these environment variables to the resource locations on your site.

| Variable | Existing resource |
| --- | --- |
| `WEIGHTS` | Checkpoint cache with the renderer releases listed in `driveaudit/generation/installation.py`, including BaseDreamer's CLIP directory. |
| `CATK_ASSETS` | CAT-K asset root with `checkpoints/clsft_E9.ckpt` and the three token vocabulary files under `src/smart/tokens`. |
| `V2_CKPTS`, `V2_PRETRAINED` | MagicDrive-V2 DiT checkpoint tree and the CogVideoX VAE/T5 pretrained tree. |
| `ABI_ROOT` | The compatible compiled gen-benchmark tree. Its project-relative BaseDreamer, DreamForge, MagicDrive and Panacea extension paths are listed in the installer. These are the compiled parts of the tested Linux/Python/CUDA environments. |
| `CONDA_ENVS` | The native Python environments used by the individual models. |

```bash
ENGINE_CODE="$PWD/work/native"
python scripts/geninstall_native.py --destination "$ENGINE_CODE" \
  --weights-root "$WEIGHTS" --catk-assets "$CATK_ASSETS" \
  --v2-checkpoints "$V2_CKPTS" --v2-pretrained "$V2_PRETRAINED" \
  --extensions-root "$ABI_ROOT"
python scripts/geninstall_native.py --destination "$ENGINE_CODE" --verify
```

`NATIVE_INSTALL.json` records hashes for the installed source and compiled extensions and the seven explicit weight/vocabulary links. The tested installation contains 3,193 source/config/license files and 29 compiled extensions. A machine without the GPU environments can unpack and verify the full code for inspection with `python scripts/geninstall_native.py --source-only --destination work/native`. The source checkout can be installed with networking disabled. An installed wheel exposes the same installer as `python -m driveaudit.generation.native`. To reproduce the distribution from an existing source installation, use `scripts/genpack_native.py --source-code SOURCE_TREE --dependencies-root ABI_ROOT --output native-sources.tar.gz`. The resulting compressed archive and its sibling `native-sources.tar.json` are deterministic for a fixed input snapshot. [Native source inventory](../vendor/generation/bundles/native-sources.tar.json) · [Source packer](../scripts/genpack_native.py) · [Offline installer](../scripts/geninstall_native.py)

## Configure a generation run

Set `ENGINE_CODE` to the native engine directory described below, `WEIGHTS` to your checkpoint cache, `CONDA_ENVS` to the directory of native model environments, `ARCHIVE` to the participant tarball, and `WORK` to a new run directory. Paths in the run JSON resolve relative to that JSON. A Linux filesystem and the native model installations are required for GPU inference.

```bash
python -m driveaudit.generation init \
  --engine-code-root "$ENGINE_CODE" --conda-envs "$CONDA_ENVS" \
  --archive "$ARCHIVE" --work-dir "$WORK" --weights-root "$WEIGHTS" \
  --scene case_000001 case_000002 --output run.json
```

The installation provides the following native layout.

```text
ENGINE_CODE/
  catk/third_party/catk-official/{src,checkpoints}
  MagicDrive-V2/rd/{magicdrivedit,scripts,ckpts,pretrained,...}
  gen-benchmark/
    DriveAudit-video/WorldDreamer/{BaseDreamer,DreamForge,.build/dit/dreamforgedit,...}
    MagicDrive-main/
    MagicDrive-video/
    MagicDrive-bevfusion/
    OpenDWM-main/{src,examples,...}
    HorizonDrive-main/{horizondrive,videox_fun,...}
    panacea-main/{sgm,configs,...}
```

The MagicDrive and DriveAudit implementations use their corresponding video trees and compiled dependencies. The inline `models` section created by `--weights-root` records every environment and checkpoint path and can be edited for a different storage layout. No previous experiment directory or generated images are needed for fresh inference.

On the original HPC system also pass `--cuda-prefix /share/apps` to select the original CUDA toolkit per renderer. Omitting `--weights-root` selects the existing-installation profile and sets `models_json` to the installation's sibling `configs/models.json`. If the installation moved, update the per-model Python and checkpoint paths in a private copy of that model registry. Each adapter resolves its interpreter and checkpoints using the registry fields below.

| Model | Python and checkpoints |
| --- | --- |
| BaseDreamer, MagicDrive | `python.renderer`, `render_checkpoint`, `sd15` |
| DreamForge | Same fields plus `single_checkpoint` |
| HorizonDrive | `python.renderer`, `render_checkpoint`, `wan_checkpoint`. Place `horizondrive-vae.pkl` next to the renderer checkpoint. |
| OpenDWM | `python.renderer`, `render_checkpoint`, `base_model` |
| Panacea | `python.renderer`, `render_checkpoint`, `clip_checkpoint` |
| MagicDrive-V2 | `v2_python` in the run JSON. The native `MagicDrive-V2/rd/ckpts` and `rd/pretrained` trees provide the DiT, CogVideoX VAE and T5 weights. |

To use independently installed CAT-K assets, set `catk_official` to the directory containing `src/smart` and `checkpoints/clsft_E9.ckpt`, and `catk_python` to its Python executable. Default source and vocabulary checksums are in `vendor/generation/catk/official_source.sha256` and `original_config.yaml`. The CLI preserves the original deterministic seed 20260914 unless the run JSON supplies `traffic_seed`.

## Generate and stage traffic

```bash
srun -p L40 --gres=gpu:l40:1 -c 4 --mem=16G \
  python -m driveaudit.generation traffic --config run.json
python -m driveaudit.generation stage --config run.json
```

The traffic stage verifies selected archive members, runs the pretrained CAT-K decoder, validates the native submission CSV, and loads each renderer input before publishing `WORK/traffic/COMPLETED.json`. Actor outputs are under `WORK/traffic/submission/scenes/<case>/future/future_trajectories.csv`. The report distinguishes predicted actors, held static objects, and inactive historical tracks. The local virtual LiDAR rig used by the DriveAudit adapters is declared in each staged scene.

To stage a previously completed traffic run, set `catk_run` to its directory in the run JSON and execute `stage`. This reuses its predictions. Staging checks the archive digest, actor CSV digests and byte identity of the public ego trajectory. It hashes every staged scene file, including history RGB, map and calibration, then publishes `WORK/conditions` by atomic rename. Rendering verifies these scene digests before loading a model. A new work directory is required to change the conditions of an existing run.

## Render future RGB

```bash
for model in basedreamer dreamforge horizondrive magicdrive opendwm panacea; do
  srun -p L40 --gres=gpu:l40:2 -c 12 --mem=96G \
    python -m driveaudit.generation render --config run.json --model "$model" || exit 1
done
srun -p L40 --gres=gpu:l40:4 -c 24 --mem=256G \
  python -m driveaudit.generation render --config run.json --model magicdrivev2
```

The original native adapters use Slurm L40 allocations, with two GPUs for the first six models and four for MagicDrive-V2. The process launcher stays in the foreground so the scheduler tracks failures. Each run uses a content-addressed private view of the installed native source and shares external model weights. Source search paths explicitly select the migrated diffusers, xFormers and 3D detection variants before environment defaults. `WORK/logs/<model>/attempt-*/` contains the exact argument vectors, worker logs, wall time, process return codes, and verification output. A re-run in the same conditioned directory follows each adapter's native resume checks. Existing status files seen at launch are recorded explicitly.

Outputs are `WORK/conditions/predictions/<model>/nuscenes/scenes/<case>/future/rgb/<camera>/<frame>.jpg`. Verification requires valid native completion metadata and decodes every 400 × 224 RGB JPEG across six cameras and frames 10–89. Each model and scene produces 480 images. Comparison of new generator settings or checkpoints requires a fresh work directory to avoid accidental reuse.

| Model | Native inference mode | Tested Python / PyTorch |
| --- | --- | --- |
| BaseDreamer | Per-scene worker, released single-reference pipeline | 3.9 / 1.10.2+cu113 |
| DreamForge | Per-scene worker, 20 denoising steps | 3.9 / 2.0.0+cu118 |
| HorizonDrive | Accelerate distributed evaluation, bf16 | 3.10 / 2.8.0+cu128 |
| MagicDrive | Per-scene worker, released video pipeline | 3.8 / 1.10.2+cu113 |
| OpenDWM | Two-rank FSDP, 40 denoising steps | 3.10 / 2.5.1+cu121 |
| Panacea | Two ranks, 25 steps, rolling generated reference and ordered future slots | 3.8 / 1.13.1+cu117 |
| MagicDrive-V2 | Four GPUs, 89-frame sequence, 9 history RGB frames and 80 generated frames | 3.10 / 2.4.0 |

Panacea puts the reference in the model's final slot, supplies the future targets in chronological order, and takes the last generated frame from each window as the next reference.

## What conditions reach each renderer

The actor CSV supplies box centers, dimensions, headings and categories at each target timestamp. The map comes from the participant archive. The public ego poses and calibrated camera rig put both the boxes and map in each target camera's frame. The shared loader validates the supplied camera-to-anchor records. The renderers below derive their camera geometry from ego poses and calibrated camera-to-ego transforms. All seven receive a scene prompt.

| Renderer | RGB reference in this profile | Traffic, map and motion supplied to inference |
| --- | --- | --- |
| BaseDreamer | Frame 9 initializes the first step. Each following step uses the preceding generated six-view frame. | The pipeline receives a four-channel bird's-eye-view (BEV) road/map raster, 14-channel projected layouts, visible 3D box corner/class/mask tokens, camera matrices and relative ego pose between the target and reference. |
| DreamForge | Both initial reference slots contain frame 9. Later windows use the preceding window's last two generated frames. | The pipeline takes a BEV map, a per-camera map/actor layout, 3D box tokens and visibility masks, calibrated projection matrices and CAN-bus-style motion derived from the public ego trajectory in the local frame. |
| HorizonDrive | History frames 0–9 plus a repeated frame 9 fill 11 reference slots. | Its configured conditioning projections receive per-camera bbox and HD-map images plus consecutive ego-motion actions (local delta x, delta y and delta yaw). Calibration is used to project the rasters. |
| MagicDrive | Layout-only inference. This adapter generates both windows from noise. | Two overlapping 61-frame windows use ego-aligned semantic BEV maps, visible 3D box corners/classes/masks and camera matrices. The public ego motion enters through per-frame map and box coordinates. |
| OpenDWM | RGB frames 7–9 initialize the rollout. Later 19-frame windows carry three overlap frames. | The native six-channel image adapter receives colored projected box edges and public HD-map contours on all frames. Its explicit view model also takes normalized camera intrinsics, rig extrinsics composed with public ego poses, and a neighboring-view mask. |
| Panacea | Frame 9 initializes the first reference. Later windows use their last generated target. | The 19-channel panorama combines three projected box channels, ten class-specific box-depth channels, three vector-map channels and three calibrated viewing-ray channels. Each future slot receives its own layout transformed with the public ego pose. |
| MagicDrive-V2 | The VAE encodes frames 1–9 as a fixed three-latent prefix within the 89-frame sequence. | The denoiser receives warped semantic BEV maps, 3D corner/class/visibility tensors, camera calibration, relative ego poses and the measured clip FPS. Missing map channels are zeros because the checkpoint lacks an unknown-layer mask. |

**OpenDWM condition path.** The GenBench dataset rasterizes the valid traffic boxes for every view into `3dbox_images`. It separately draws available drivable-area and pedestrian-crossing contours into `hdmap_images`. The participant map lacks native lane polygons, so the lane layer is absent. Box categories outside the renderer's color table are skipped. The released `ctsd_unimlvg_6views_video_generation.json` enables `condition_on_all_frames`, `explicit_view_modeling` and a six-channel `condition_image_adapter_config`. In the native `CrossviewTemporalSD.get_conditions` method, concatenating the two three-channel rasters produces `condition_image_tensor`. The DiT forward pass sends this tensor into the image adapter and uses its residuals. The same profile sets `cond_with_action=false` and omits `added_time_ids`, so there is no separate numerical action/FPS conditioning vector. Its `pts` array determines sequence length in this sampling path. These are input-path properties. The strength of the visual response requires an output ablation. [OpenDWM dataset adapter](../vendor/generation/gen-benchmark/OpenDWM-main/src/dwm/datasets/genbench_gt2.py) · [traffic and map rasterizer](../vendor/generation/gen-benchmark/OpenDWM-main/src/dwm/datasets/genbench_gt2_renderer.py) · [native sampling call](../vendor/generation/gen-benchmark/OpenDWM-main/infer_genbench_gt2.py)

Run the geometry diagnostic to compare box and map conditioning with and without traffic actors.

```bash
"$CONDA_ENVS/opendwm/bin/python" scripts/gencheck_opendwm_conditions.py \
  --conditions "$WORK/conditions" --engine-code-root "$ENGINE_CODE"
```

## Slurm submission

The Slurm scheduler entry accepts three starting points. Use `bash scripts/submit_demo.sh CONFIG fresh` for a new traffic run, `staged` to render and evaluate a different generator using an existing verified CAT-K conditions directory, or `rendered` for evaluation of existing renderer results. For each additional renderer select its ID in `CONFIG` and a distinct evaluation output directory. All renderers can share the same generation work directory and traffic.

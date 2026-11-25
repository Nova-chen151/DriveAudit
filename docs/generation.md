# Participant inputs, CAT-K and future RGB

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](generation.md) [![简体中文](https://img.shields.io/badge/简体中文-777777?style=flat-square)](generation.zh-CN.md)

[Home](../README.md) · [Evaluation guide](evaluation.md#evaluation-scope)

Run the commands below from a DriveAudit checkout. The input is a GenBench `nuscenes_ego_conditioned_v1` participant archive. CAT-K takes observed traffic and map context, predicts future actors and its own ego trajectory jointly, and exports the actors at the 80 participant target timestamps. The renderer uses the public ego and camera motion supplied in the archive. CAT-K's predicted ego trajectory is available for diagnostics.

<a id="install-the-native-inference-source-offline"></a>

## Install the native inference source offline

Native sources are under [CAT-K](../models/generation/trajectory/CAT-K) and [renderers](../models/generation/image). The installer combines these sources, participant adapters, compatible compiled extensions, and external weights into `work/native`.

Model-specific inference and closed-loop workers live alongside each model. Shared participant readers, rendering conditions and frame validation live in `driveaudit/generation/`. The [adapter inventory](../driveaudit/generation/bundles/adapters.json) maps those files to the paths expected by the native engines.

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

`NATIVE_INSTALL.json` records installed source and extension hashes and asset links. For source-only inspection, use `python scripts/geninstall_native.py --source-only --destination work/native`. See the [source inventory](../driveaudit/generation/bundles/native-sources.tar.json) for provenance.

<a id="model-weights"></a>

## Model weights

Use `weights/` as the shared root: `catk/` for traffic weights and vocabularies, `renderers/` and `magicdrivev2/` for renderer assets, `agents/` for policy assets, and `evaluation/` for LoFTR, DINO, and CLIP-IQA+. A provisioned store records its inventory in `weights/inventory.json`; weights are not downloaded by cloning the repository.

Use `python scripts/collect_weights.py --help` to import an existing store. The native installer also accepts `--weight-store WEIGHTS` with `--extensions-root ABI_ROOT`. Shared weights may be hard-linked: replace files with new versions rather than overwriting them in place. Rebuild the native installation after changing model source. Policy environments are covered in the [evaluation guide](evaluation.md).

<a id="configure-a-generation-run"></a>

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

Use `--cuda-prefix` to set a custom CUDA toolkit directory. Omitting `--weights-root` selects the existing-installation profile and sets `models_json` to the installation's sibling `configs/models.json`. If the installation moved, update the per-model Python and checkpoint paths in a private copy of that model registry. Each adapter resolves its interpreter and checkpoints using the registry fields below.

| Model | Python and checkpoints |
| --- | --- |
| BaseDreamer, MagicDrive | `python.renderer`, `render_checkpoint`, `sd15` |
| DreamForge | Same fields plus `single_checkpoint` |
| HorizonDrive | `python.renderer`, `render_checkpoint`, `wan_checkpoint`. Place `horizondrive-vae.pkl` next to the renderer checkpoint. |
| OpenDWM | `python.renderer`, `render_checkpoint`, `base_model` |
| Panacea | `python.renderer`, `render_checkpoint`, `clip_checkpoint` |
| MagicDrive-V2 | `v2_python` in the run JSON. The native `MagicDrive-V2/rd/ckpts` and `rd/pretrained` trees provide the DiT, CogVideoX VAE and T5 weights. |

To use independently installed CAT-K assets, set `catk_official` to the directory containing `src/smart` and `checkpoints/clsft_E9.ckpt`, and `catk_python` to its Python executable. Default source and vocabulary checksums are in `models/generation/trajectory/CAT-K/official_source.sha256` and `original_config.yaml`. The CLI preserves the original deterministic seed 20260914 unless the run JSON supplies `traffic_seed`.

<a id="generate-and-stage-traffic"></a>

## Generate and stage traffic

```bash
srun -p L40 --gres=gpu:l40:1 -c 4 --mem=16G \
  python -m driveaudit.generation traffic --config run.json
python -m driveaudit.generation stage --config run.json
```

The traffic stage verifies selected archive members, runs the pretrained CAT-K decoder, validates the native submission CSV, and loads each renderer input before publishing `WORK/traffic/COMPLETED.json`. Actor outputs are under `WORK/traffic/submission/scenes/<case>/future/future_trajectories.csv`. The report distinguishes predicted actors, held static objects, and inactive historical tracks. The local virtual LiDAR rig used by the DriveAudit adapters is declared in each staged scene.

To stage a previously completed traffic run, set `catk_run` to its directory in the run JSON and execute `stage`. This reuses its predictions. Staging checks the archive digest, actor CSV digests and byte identity of the public ego trajectory. It hashes every staged scene file, including history RGB, map and calibration, then publishes `WORK/conditions` by atomic rename. Rendering verifies these scene digests before loading a model. A new work directory is required to change the conditions of an existing run.

<a id="render-future-rgb"></a>

## Render future RGB

```bash
for model in basedreamer dreamforge horizondrive magicdrive opendwm panacea; do
  srun -p L40 --gres=gpu:l40:2 -c 12 --mem=96G \
    python -m driveaudit.generation render --config run.json --model "$model" || exit 1
done
srun -p L40 --gres=gpu:l40:4 -c 24 --mem=256G \
  python -m driveaudit.generation render --config run.json --model magicdrivev2
```

These example allocations use two L40 GPUs for the first six models and four for MagicDrive-V2. Adjust them to your cluster. Logs and verification output are under `WORK/logs/<model>/attempt-*/`. Re-running follows the adapter's native resume checks; use a fresh work directory for changed conditions or checkpoints.

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

<a id="what-conditions-reach-each-renderer"></a>

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

OpenDWM conditions all frames on projected traffic boxes and available HD-map contours, with calibrated camera geometry. The participant map has no native lane polygons; unsupported box categories are skipped. This profile uses no separate numerical action/FPS vector. See the [dataset adapter](../models/generation/image/OpenDWM/src/dwm/datasets/genbench_gt2.py) and [rasterizer](../models/generation/image/OpenDWM/src/dwm/datasets/genbench_gt2_renderer.py).

Run the geometry diagnostic to compare box and map conditioning with and without traffic actors.

```bash
"$CONDA_ENVS/opendwm/bin/python" scripts/gencheck_opendwm_conditions.py \
  --conditions "$WORK/conditions" --engine-code-root "$ENGINE_CODE"
```

<a id="slurm-submission"></a>

## Slurm submission

The Slurm scheduler entry accepts three starting points. Use `bash scripts/submit_demo.sh CONFIG fresh` for a new traffic run, `staged` to render and evaluate a different generator using an existing verified CAT-K conditions directory, or `rendered` for evaluation of existing renderer results. For each additional renderer select its ID in `CONFIG` and a distinct evaluation output directory. All renderers can share the same generation work directory and traffic.

<a id="participant-inputs-and-submission"></a>

## Participant inputs and submission

The participant package is specified by `archive` in the generation configuration. It contains 250 nuScenes scenes, task instructions, the submission format and a public validator. The example selects the first two public scene IDs.

Each scene provides six-view RGB for frames 0–9, historical traffic actors, a map, camera calibration and timestamps. Ego and camera poses for frames 10–89 are given conditions. Traffic trajectories use the frame-9 ego coordinate system: x forward, y left and z up, with lengths in metres. Sampling times are read from `timeline.json`. CAT-K predicts other traffic actors, and the selected renderer generates the corresponding future images.

<a id="export-the-submission-directory"></a>

## Export the submission directory

Run from the repository root. Set `PARTICIPANT_DIR` to the extracted `participant/` directory. The [challenge specification](challenge.md) defines the trajectory CSV and RGB frame names.

```bash
python -m driveaudit.submission \
  --inputs "$PARTICIPANT_DIR" \
  --conditions work/two-scene-generation/conditions --renderer opendwm \
  --cases case_000001 case_000002 \
  --output work/two-scene-submission
```

Each scene exports one anonymous traffic-trajectory CSV and 80 RGB JPEGs at 400 × 224 for each of six cameras. File names and symbolic-link restrictions follow the validator included in the participant package. The program invokes that validator for every selected scene and publishes the submission directory only after all scenes pass. The two-scene demonstration covers only the selected IDs; a full official submission must cover the complete scene list released by the organizer.

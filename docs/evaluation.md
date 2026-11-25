# Policies and paired evaluation

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](evaluation.md) [![Simplified Chinese](https://img.shields.io/badge/Simplified_Chinese-777777?style=flat-square)](evaluation.zh-CN.md)

[Home](../README.md) · [Generation guide](generation.md#participant-inputs-and-submission)

The same frozen camera policy runs on generated and time-matched recorded observations. Both branches share timestamps, map, public ego odometry, commands, and seeds; each uses its own traffic for collision and TTC scoring.

<a id="evaluation-scope"></a>

The scoring implementation lives in [driveaudit/evaluation/driveaudit_eval](../driveaudit/evaluation/driveaudit_eval). The [DriveAudit scoring entry point](../driveaudit/evaluation/score.py) imports it for both recorded and generated inputs.

## Evaluation scope

The manuscript reports a 550-scenario, six-policy benchmark under **non-reactive replay**. Its additional 20-scenario closed-loop study is a separate protocol. The repository example uses two nuScenes scenes; results describe only the selected scenes. A single-policy diagnostic or a verified closed-loop feedback chain is not formal six-policy QTES.

<a id="policy-setup-and-inference"></a>

## Policy setup and inference

DriveAudit includes the model and feature-building source from the six DrivingAgents checkouts used in the prior evaluator. Each policy has an isolated Python environment because NAVSIM forks use the same import names and the MMDetection packages pin different versions.

| Policy | Vendored source | Native camera inputs | Service name |
| --- | --- | --- | --- |
| DrivoR | `models/agents/DrivoR` | front, back, front-left, front-right | `drivor` |
| UniAD | `models/agents/UniAD` | six | `uniad` |
| VAD | `models/agents/VAD` | six | `vad` |
| SparseDriveV2 | `models/agents/SparseDriveV2` | front-left, front, front-right | `sparsedrivev2` |
| DiffusionDriveV2 | `models/agents/DiffusionDriveV2` | front-left, front, front-right | `diffusiondrivev2` |
| GoalFlow | `models/agents/GoalFlow` | front-left, front, front-right | `goalflow` |

<a id="drivor-paired-replay"></a>

### DrivoR paired replay

Use the DrivoR environment on a CUDA node. Set `DRIVEAUDIT_AGENT_ASSETS` to the parent directory containing the six case-sensitive model directories. The DrivoR defaults resolve to `DrivoR/weights/drivor_Nav1_25epochs.pth` and `DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`. Both paths accept CLI overrides. Set `NUPLAN_DEVKIT_ROOT` to a nuPlan devkit checkout when the package is unavailable in the environment. The measured environment uses Python 3.9, PyTorch 2.1/CUDA 12.1 and timm 1.0.15. The source release's setup and weight download instructions are in [the DrivoR upstream README](../models/agents/DrivoR/UPSTREAM.md).

Run both commands from the repository root. `PUBLIC_SCENES` contains the public case directories. `GENERATED_SCENES` contains the renderer-specific case directories, each with `future/rgb/CAM_*/000010.jpg` and subsequent frames. `REAL_MAPPING` is the organizer's JSON array of `public_scene` and `source_path` records. Each source path resolves to the corresponding reference scene directory.

```bash
python -m driveaudit.agents.predict --source generated \
  --public-scenes "$PUBLIC_SCENES" --rgb-scenes "$GENERATED_SCENES" \
  --cases case_000001 case_000002 --output work/ego/drivor/generated.jsonl

python -m driveaudit.agents.predict --source real \
  --public-scenes "$PUBLIC_SCENES" --real-mapping "$REAL_MAPPING" \
  --cases case_000001 case_000002 --output work/ego/drivor/real.jsonl
```

DrivoR uses its released four-camera subset with calibration, velocity, acceleration, and a shared driving command. The demo defaults to STRAIGHT; use `--command` to select another common command. Each branch replays frames 10–89 at public ego poses; predicted actions do not alter subsequent observations. Receipts record model/configuration and input-image hashes, native eight-pose trajectories, and six scoring waypoints at +0.5 through +3 s (metres, x forward, y left).

<a id="replay-with-any-of-the-six-services"></a>

### Replay with any of the six services

The managed replay client starts a local service in the selected environment, submits the case in temporal order and writes the same six-waypoint JSONL scoring format. It checks the actual required checkpoints, anchors and native config. Give the client a Python environment with NumPy, SciPy and Pillow and pass the policy's interpreter with `--service-python`. Source directories default to `models/agents`. `DRIVEAUDIT_AGENT_SOURCES` can point to the parent of full compatible model checkouts. Assets use the original checkpoint layouts within `DRIVEAUDIT_AGENT_ASSETS`.

```bash
python -m driveaudit.agents.service sparsedrivev2 --preflight
python -m driveaudit.agents.replay --policy sparsedrivev2 \
  --service-python "$SPARSEDRIVE_ENV/bin/python" --source generated \
  --public-scenes "$PUBLIC_SCENES" --rgb-scenes "$GENERATED_SCENES" \
  --cases case_000001 case_000002 --output work/ego/sparsedrivev2/generated.jsonl
```

For paired evaluation use the same policy, command, assets and cases, switch to `--source real --real-mapping "$REAL_MAPPING"`, and choose a separate file. Pass the two JSONLs to the policy evaluator. `--max-decisions 1` runs the first public future decision per case for a diagnostic. Full replay uses every decision.

UniAD and VAD warm their temporal state with all ten public history images per camera before the first scored future frame. Their six-camera input uses the original view order, observation calibration, public ego pose, and the declared virtual rig in `participant_render_frame.json`. Their right/forward outputs are transformed to forward/left metres. The other four services receive their native feature-builder envelopes. GoalFlow and DiffusionDriveV2 default to the prior evaluator's zero-LiDAR compatibility mode. Their frozen multimodal architectures consume a fixed zero tensor in that mode.

Every managed replay row binds the service adapter, checkpoint, config and named assets by SHA-256. Temporal rows also fingerprint the ordered public history frames and exact camera bytes used for warm-up. The camera-only receipt describes the enforced application protocol. The client additionally checks both multimodal services report an all-zero LiDAR tensor for each decision.

Each policy requires its own upstream packages or compiled CUDA operators. SparseDriveV2 includes the CUDA sources in `models/agents/SparseDriveV2/navsim/agents/sparsedrive/ops`. Build the operators there with `python setup.py build_ext --inplace` using the SparseDriveV2 environment, or point `DRIVEAUDIT_SPARSE_OPS` at an existing ABI-compatible operator directory. For the NAVSIM models, `NUPLAN_DEVKIT_ROOT` may point to the matching external nuPlan devkit checkout. Model setup and release files appear in each vendor's `UPSTREAM.md`.

The dedicated DrivoR replay preserves the native eight-pose trajectory. The six-policy panel uses managed replay for consistent sensor, history and model receipts. Source licenses are in the respective vendor directories, and the adapter notice is in `models/agents/NOTICE.md`.

<a id="run"></a>

## Run

From the repository root in a Python environment with NumPy, SciPy, Shapely and PyYAML, run the following after predictions and organizer staging are complete.

```bash
python -m driveaudit.evaluation \
  --generated work/ego/drivor/generated.jsonl \
  --real work/ego/drivor/real.jsonl \
  --scenes work/evaluation/staged/scenes.json \
  --trs work/evaluation/trs/result.json \
  --quality work/evaluation/quality.json \
  --output work/evaluation/paired
```

Omit `--quality` or `--trs` while those measurements are pending. Policy scores and support counts remain available. Rerun with the evidence to complete the combined report. Runtime paths must be provisioned by the organizer.

<a id="inputs-and-checks"></a>

## Inputs and checks

- The scene mapping contains one key per public case, with the absolute `generated` and `real` scene directories. Each scene follows the native evaluation format, including the history, future traffic, time-indexed ego poses, drivable-area raster and camera frames.
- Each prediction file contains one row for every public future decision. It records the case, native `frame_index`, zero-based future `frame_id`, `reference_time_us`, `source`, `policy`, `request_seed`, `ego_state`, `command`, SHA-256 digests for model/config/backbone and `rgb_sha256` by camera. The scorer verifies each consumed JPEG, the shared route/map/time identity, paired decision timestamps, model identity, supplied controls and seeds. It rejects duplicate/missing decisions. The DrivoR camera set is front, front-left, front-right and rear.
- `trajectory_xy` contains six absolute positions in the decision's ego frame at 0.5, 1, 1.5, 2, 2.5 and 3 s, measured in metres with x forward and y left. The last observations without three seconds of recorded future support remain in inference receipts and are counted separately.
- `--quality` accepts paired `generated` and `real` native local quality records keyed by case. Each record supplies the dataset, scene ID, raw scores for `clip_iqa`, `dino_t`, `epi_3px` and `tvc`, computation and dimension statuses, measured coverage, DINO `temporal_pairs` and the TVC component mask. The native quality kernel checks coverage and identical support.
- `--trs` accepts the paired population evaluator's `result.json`. Scene identities and candidate/reference source hashes must match the selected scene trees.

<a id="reading-the-results"></a>

## Reading the results

`summary.json` holds scene scores, input fingerprints, paired decision counts and the policy identity. `paired.csv` contains a real and generated row for every scene. The per-branch JSON files preserve window scores, collision evidence and the replay's termination and route projection trace.

RPDMS uses native `score_window` with the observed-map DAC protocol. Each eligible three-second plan contributes safety, comfort and route-progress terms. An overlap triggers responsibility attribution. Unavailable attribution or inadequate drivable-map support leaves that window unscored. Both branches' reported RPDMS means use the exact same supported window IDs. The one-policy TE requires at least 80% common coverage. Every result reports the numerator and denominator.

Route Completion uses the original logged-state receding-horizon replay with terminal horizon commit, dense collision checks and continuous projection onto the full logged route. Each decision reanchors the predicted prefix at recorded ego odometry. The route column is available only when native projection coverage satisfies the frozen threshold. The traffic and images come from fixed sequences.

For each scene, the native paper quality function computes local retention `B` from CLIP-IQA+, DINO-T, Epi@3 and TVC. With complete support, `single_policy_QTES_diagnostic` applies `100 × TRS_population^0.25 × mean_scene(TE_policy × B)`. The two-scene TRS is a population measure. It enters the expression once. The numeric diagnostic requires evidence for every selected scene. The formal six-policy QTES field remains null when the one-policy demonstration is run.

<a id="full-six-policy-panel"></a>

## Full six-policy panel

The panel consumes the two complete camera-only captures for each of UniAD, VAD, DrivoR, SparseDriveV2, DiffusionDriveV2 and GoalFlow. It verifies asset/config and adapter digests, warmed temporal history, real-vs-zero LiDAR protocol, timestamps, odometry, command and input images. These sensor receipts describe the application inputs and native adapter responses. The organizer's quality collector binds its results to decoded scene evidence and measured traffic files. Place each policy's files under a common root with `<policy>/generated.jsonl` and `<policy>/real.jsonl`. This demo uses the receipt-bearing captures under `work/ego/panel-input/`. Optional `--capture-dir POLICY=PATH` overrides one directory.

```bash
python -m driveaudit.evaluation.panel \
  --ego-root work/ego/panel-input \
  --scenes work/evaluation/staged/scenes.json --quality work/evaluation/quality.json \
  --trs work/evaluation/trs/result.json --output work/evaluation/panel --workers 6
```

The evaluator freezes one common window mask across all twelve branch/policy combinations per scene. At least ten valid windows and 80% coverage are required. It computes paired RPDMS on those indices and passes matched records to the original six-agent `paired_comparison` and `paper_population_v1` aggregation. Missing captures or unverified pairing prevent a numeric result. `panel.json` records TE, QTES, local retentions, source identities and common window indices. `panel.csv` puts the real and generated RPDMS and route results, TE and common support side by side for each scene and policy. Scores apply to the selected scenes.

<a id="qtes"></a>

## QTES

The formal panel score combines paired policy degradation (`TE`), local quality retention (`B`), population traffic realism (`TRS`), and validity (`V`):

$$
\mathrm{QTES}_{\mathcal P}
=100\,\mathrm{TRS}_{\mathcal P}^{1/4}
\frac{1}{|\mathcal P|}\sum_{n\in\mathcal P}V_n\,\mathrm{TE}_n\,B_n.
$$

`B` uses reference-relative CLIP-IQA+, DINO-T, Epi@3, and TVC. TRS is evaluated over the selected scene population, not per frame. A valid reference self-comparison has QTES = 50 and TE = 0.5; these are anchors, not pass/fail thresholds. Higher RPDMS means better policy performance; higher QTES means greater quality-aware testing effectiveness. Missing quality evidence or invalid common support cannot produce a complete score.

# Paired policy evaluation

The evaluation compares the same frozen camera policy on a submitted future sequence and its time-matched recorded counterpart. The generated branch uses the participant's own future traffic for collision and TTC scoring. The reference branch uses recorded actors. Both share native timestamps, map and public ego odometry. This measures the combined change in rendered observations and traffic. The original six-policy benchmark and the one-policy diagnostic have distinct output fields.

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

## Inputs and checks

- The scene mapping contains one key per public case, with the absolute `generated` and `real` scene directories. Each scene follows the native evaluation format, including the history, future traffic, time-indexed ego poses, drivable-area raster and camera frames.
- Each prediction file contains one row for every public future decision. It records the case, native `frame_index`, zero-based future `frame_id`, `reference_time_us`, `source`, `policy`, `request_seed`, `ego_state`, `command`, SHA-256 digests for model/config/backbone and `rgb_sha256` by camera. The scorer verifies each consumed JPEG, the shared route/map/time identity, paired decision timestamps, model identity, supplied controls and seeds. It rejects duplicate/missing decisions. The DrivoR camera set is front, front-left, front-right and rear.
- `trajectory_xy` contains six absolute positions in the decision's ego frame at 0.5, 1, 1.5, 2, 2.5 and 3 s, measured in metres with x forward and y left. The last observations without three seconds of recorded future support remain in inference receipts and are counted separately.
- `--quality` accepts paired `generated` and `real` native local quality records keyed by case. Each record supplies the dataset, scene ID, raw scores for `clip_iqa`, `dino_t`, `epi_3px` and `tvc`, computation and dimension statuses, measured coverage, DINO `temporal_pairs` and the TVC component mask. The native quality kernel checks coverage and identical support.
- `--trs` accepts the paired population evaluator's `result.json`. Scene identities and candidate/reference source hashes must match the selected scene trees.

## Reading the results

`summary.json` holds scene scores, input fingerprints, paired decision counts and the policy identity. `paired.csv` contains a real and generated row for every scene. The per-branch JSON files preserve window scores, collision evidence and the replay's termination and route projection trace.

RPDMS uses native `score_window` with the observed-map DAC protocol. Each eligible three-second plan contributes safety, comfort and route-progress terms. An overlap triggers responsibility attribution. Unavailable attribution or inadequate drivable-map support leaves that window unscored. Both branches' reported RPDMS means use the exact same supported window IDs. The one-policy TE requires at least 80% common coverage. Every result reports the numerator and denominator.

Route Completion uses the original logged-state receding-horizon replay with terminal horizon commit, dense collision checks and continuous projection onto the full logged route. Each decision reanchors the predicted prefix at recorded ego odometry. The route column is available only when native projection coverage satisfies the frozen threshold. The traffic and images come from fixed sequences.

For each scene, the native paper quality function computes local retention `B` from CLIP-IQA+, DINO-T, Epi@3 and TVC. With complete support, `single_policy_QTES_diagnostic` applies `100 × TRS_population^0.25 × mean_scene(TE_policy × B)`. The two-scene TRS is a population measure. It enters the expression once. The numeric diagnostic requires evidence for every selected scene. The formal six-policy QTES field remains null when the one-policy demonstration is run.

## Full six-policy panel

The panel consumes the two complete camera-only captures for each of UniAD, VAD, DrivoR, SparseDriveV2, DiffusionDriveV2 and GoalFlow. It verifies asset/config and adapter digests, warmed temporal history, real-vs-zero LiDAR protocol, timestamps, odometry, command and input images. These sensor receipts describe the application inputs and native adapter responses. The organizer's quality collector binds its results to decoded scene evidence and measured traffic files. Place each policy's files under a common root with `<policy>/generated.jsonl` and `<policy>/real.jsonl`. This demo uses the receipt-bearing captures under `work/ego/panel-input/`. Optional `--capture-dir POLICY=PATH` overrides one directory.

```bash
python -m driveaudit.evaluation.panel \
  --ego-root work/ego/panel-input \
  --scenes work/evaluation/staged/scenes.json --quality work/evaluation/quality.json \
  --trs work/evaluation/trs/result.json --output work/evaluation/panel --workers 6
```

The evaluator freezes one common window mask across all twelve branch/policy combinations per scene. At least ten valid windows and 80% coverage are required. It computes paired RPDMS on those indices and passes matched records to the original six-agent `paired_comparison` and `paper_population_v1` aggregation. Missing captures or unverified pairing prevent a numeric result. `panel.json` records TE, QTES, local retentions, source identities and common window indices. `panel.csv` puts the real and generated RPDMS and route results, TE and common support side by side for each scene and policy. Scores apply to the selected scenes.

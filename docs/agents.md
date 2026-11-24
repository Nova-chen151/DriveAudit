# Ego policies

DriveAudit includes the model and feature-building source from the six DrivingAgents checkouts used in the prior evaluator. Each policy has an isolated Python environment because NAVSIM forks use the same import names and the MMDetection packages pin different versions.

| Policy | Vendored source | Native camera inputs | Service name |
| --- | --- | --- | --- |
| DrivoR | `models/agents/DrivoR` | front, back, front-left, front-right | `drivor` |
| UniAD | `models/agents/UniAD` | six | `uniad` |
| VAD | `models/agents/VAD` | six | `vad` |
| SparseDriveV2 | `models/agents/SparseDriveV2` | front-left, front, front-right | `sparsedrivev2` |
| DiffusionDriveV2 | `models/agents/DiffusionDriveV2` | front-left, front, front-right | `diffusiondrivev2` |
| GoalFlow | `models/agents/GoalFlow` | front-left, front, front-right | `goalflow` |

## DrivoR paired replay

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

At each decision the model sees the four images for that public frame, calibration, current velocity and acceleration, and a driving command. Cameras retain their native intra-frame exposure offsets. The reference timestamp identifies the frame group on both branches. Finite differences use the public poses up to the decision, with elapsed time from the public timeline. Both velocity and acceleration are expressed in the current ego axes. The demonstration fixes the command to STRAIGHT on both branches because the public cases supply no navigation command. Use `--command` to explicitly select a different common command. DrivoR consumes the four views configured by its released checkpoint.

The demonstration replays all 80 decision frames per case, indices 10–89. Each JSONL row includes the public `frame_index`, zero-based future `frame_id`, and `reference_time_us` measured relative to frame zero. `native_trajectory` contains the selected eight absolute local (x, y, yaw) waypoints. `trajectory_xy` contains the first six xy waypoints in metres, x forward and y left, at +0.5 through +3 s. The pose at every new decision follows the public camera path. A policy's proposed waypoints do not change the next observation.

The two branches share public odometry, calibration, command and request seed. Each row records the checkpoint, backbone, config and four input-image SHA-256 values, the selected proposal index and score, and the strict state-dictionary tensor count. The generated command accepts the renderer tree alone for RGB. The organizer mapping is read by the real command.

## Replay with any of the six services

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

To run a standalone service for an existing client, use e.g. `python -m driveaudit.agents.service vad --port 11001`. The DrivoR, GoalFlow, SparseDriveV2 and DiffusionDriveV2 services accept a JSON `model_input` envelope on `POST /driver-api/` and return the prediction on `GET /driver-get/`. UniAD and VAD use the original sequential stacked-image interface. Each reset clears scene state.

The dedicated DrivoR replay preserves the native eight-pose trajectory. The six-policy panel uses managed replay for consistent sensor, history and model receipts. Source licenses are in the respective vendor directories, and the adapter notice is in `driveaudit/agents/native/NOTICE.md`.

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-777777?style=flat-square)](README.zh-CN.md)

# DriveAudit Closed-loop Testing

Each round renders six-view images from the current ego pose and CAT-K traffic state. DrivoR predicts a trajectory using its native camera subset; the executor applies its first 0.5 s in five 0.1 s substeps. CAT-K writes the executed state back to traffic history, which conditions the next rendering round.

The goal is the final ego pose in the public participant conditions. The navigation adapter converts its relative bearing to DrivoR's left, right, or straight command. Arrival within the goal radius enters an absorbing terminal state.

<a id="run"></a>
## Run a Scene

Use the same CAT-K, policy, and renderer weights as generation. Link `weights` to the shared weight store and `work/resources` to the configured resources. Install native runtimes under `work/native` following the [generation guide](../docs/generation.md).

Run from the repository root with a Python environment containing NumPy:

- `PUBLIC_SCENE`: an extracted public participant scene.
- `RENDER_SCENE`: the corresponding scene produced by the generation `stage` step.
- Replace `basedreamer` with another renderer ID as needed. Traffic and RGB are regenerated each round.

```bash
python closed_loop/configure.py --renderer basedreamer \
  --participant-scene "$PUBLIC_SCENE" --render-scene "$RENDER_SCENE" \
  --output work/closed-loop/config.json

# Run inside a Slurm GPU allocation.
python closed_loop/run_goal.py --config work/closed-loop/config.json \
  --output work/closed-loop/run-01 --steps 2
python closed_loop/verify_run.py work/closed-loop/run-01
```

<a id="resources"></a>
## Resources and Verification

| Renderers | Default GPUs |
| --- | ---: |
| BaseDreamer, DreamForge, HorizonDrive, MagicDrive, Panacea | 2 |
| OpenDWM, MagicDrive-V2 | 5, including four rendering workers |

Use a fresh output directory for each run. The verifier checks execution traces, next-round poses, generated-image fingerprints, and terminal-state behavior. Per-round records and timing are written under ignored `work/`. Passing verification establishes feedback-chain consistency; [paired policy scoring](../docs/evaluation.md) is a separate evaluation.

MagicDrive-V2's four rendering processes each load the model in host memory and offload weights to GPUs. A five-GPU, single-scene Slurm job can request `--gres=gpu:l40:5 --cpus-per-task=20 --mem=256G`.

[← Project overview](../README.md)

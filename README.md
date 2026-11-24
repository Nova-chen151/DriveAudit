<div align="center" id="top">

<p align="center">
  <img src="assets/readme/title.gif" width="880" alt="DriveAudit: animated blue trajectory tracing the project wordmark">
</p>

## Beyond Visual Realism: Benchmarking Generated Driving Scenarios as Policy Tests

**Does a generated driving scenario merely look realistic—or work as a credible policy test?**

[![Benchmark](https://img.shields.io/badge/Benchmark-550_Scenarios-2878B5?style=for-the-badge)](#dataset)
[![Generators](https://img.shields.io/badge/Generators-7_Baselines-766091?style=for-the-badge)](#model-zoo)
[![Policies](https://img.shields.io/badge/Driving_Policies-6_Models-438563?style=for-the-badge)](#model-zoo)
[![Closed Loop](https://img.shields.io/badge/Closed_Loop-Run_Guide-C57A48?style=for-the-badge)](closed_loop/README.md)

[Overview](#overview) · [Results](#benchmark-results) · [Getting Started](#getting-started) · [Closed-loop Testing](#closed-loop-testing) · [Documentation](#documentation)

<img src="assets/readme/overview.png" width="100%" alt="DriveAudit overview: multi-source driving histories, scenario generation, and four levels of policy-test evaluation">

*From realistic observations to faithful scenarios, meaningful policy responses, and quality-aware testing effectiveness.*

</div>

---

## 🔥 Highlights

- **Evaluate generated worlds as tests, not just videos.** Four coupled diagnostic levels connect observation quality, scenario fidelity, policy response, and testing effectiveness.
- **Quality-gated testing effectiveness.** QTES credits paired policy degradation only together with retained visual, geometric, image–state, and traffic quality; breaking a policy with corrupted observations is not sufficient.
- **Responsibility-aware policy scoring.** RPDMS refines collision penalties with event-aligned ego attribution while retaining safety, comfort, and progress terms.
- **Multi-source, multi-model evaluation.** The manuscript studies **550 scenarios**, **7 scenario-generating baselines**, and **6 independent driving policies** using matched scene identities and scoring anchors.
- **Generation, replay, and feedback in one repository.** Modular CAT-K traffic generation, six-view rendering, paired evaluation, and a dedicated DrivoR closed-loop runner.

> [!IMPORTANT]
> **Benchmark scope vs. this branch.** The main benchmark uses paired **non-reactive replay**: policy actions do not change later observations or traffic. This branch also provides a **closed-loop feedback runner** that updates the executed ego state before regenerating traffic and RGB. Closing the feedback loop is not, by itself, reproduction of the manuscript's full multi-policy QTES evaluation.

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Benchmark Results](#benchmark-results)
- [Model Zoo](#model-zoo)
- [Getting Started](#getting-started)
- [Paired Policy Evaluation](#paired-policy-evaluation)
- [Closed-loop Testing](#closed-loop-testing)
- [Documentation](#documentation)
- [Acknowledgements and Terms](#acknowledgements-and-terms)

<a id="overview"></a>
## 🔎 Overview

A photorealistic video can disagree with the traffic states used for scoring. A degraded rendering can also break a driving policy for the wrong reason. **DriveAudit asks whether a generated scenario remains credible as a test**, rather than treating visual realism or policy failure alone as evidence of testing value.

The generation task separates **future traffic trajectory generation** from **multi-view rendering**. Two matched conditions—recorded trajectories plus rendering, and generated trajectories plus rendering—help distinguish rendering effects from the additional effects of alternative traffic futures.

| Level | Question | Headline measurements |
| --- | --- | --- |
| **Observation quality** | Are observations realistic and temporally coherent? | FVD ↓, CLIP-IQA+ ↑, DINO-T ↑ |
| **Scenario fidelity** | Do images and submitted traffic describe the same plausible world? | Epi@3 ↑, Trajectory–Visual Consistency (**TVC**) ↑, Traffic Realism Score (**TRS**) ↑ |
| **Policy response** | How does an independent driving policy respond? | Responsibility-aware PDMS (**RPDMS**) ↑, Route Completion ↑ |
| **Testing effectiveness** | Is policy degradation supported by retained scenario quality? | Quality-aware Testing Effectiveness Score (**QTES**) ↑ |

### Quality-aware Testing Effectiveness

QTES combines scene-level paired testing effectiveness with local quality retention and population-level traffic realism:

$$
\mathrm{QTES}_{\mathcal P}
=100\,\mathrm{TRS}_{\mathcal P}^{1/4}
\frac{1}{|\mathcal P|}\sum_{n\in\mathcal P}V_n\,\mathrm{TE}_n\,B_n.
$$

- **TE** measures paired policy degradation relative to recorded reference observations; `TE = 0.5` denotes reference-equivalent response.
- **B** retains reference-relative CLIP-IQA+, DINO-T, Epi@3, and TVC evidence.
- **TRS** measures traffic realism over the evaluated scene population, not independently per frame.
- **V** enforces benchmark validity requirements. Missing or unsupported evidence must not be treated as a valid complete score.

A valid reference self-comparison gives **QTES = 50**. This is an anchor, not a pass/fail threshold. Higher RPDMS means better **policy performance**; higher QTES means greater **quality-aware testing effectiveness**. They answer different questions.

<a id="dataset"></a>
## 📦 Dataset

The manuscript's benchmark contains **550 scenario histories** with a common six-view interface, preserving timestamps, calibration, traffic states, and provenance.

| Input group | Sources | Scenarios | View provenance |
| --- | --- | ---: | --- |
| Native multi-camera | nuScenes, Argoverse 2, nuPlan, PandaSet | 500 | Recorded multi-camera histories |
| Monocular-derived multi-view | Nexar | 50 | Recorded front view; five missing views synthesized through generative view completion |
| **Total** | **Five sources** | **550** | **Observed and synthesized inputs remain explicitly distinguished** |

The repository's runnable example uses **two nuScenes scenes**, `case_000001` and `case_000002`; it is not the full 550-scenario benchmark. It generates **80 future frames × 6 cameras** per scene. Participant inputs, checkpoints, native GPU environments, and organizer reference assets must be provisioned separately. See the [participant package guide](<docs/participant.md>) and [generation guide](<docs/generation.md>).

<a id="benchmark-results"></a>
## 📊 Benchmark Results

**Visual fidelity and testing effectiveness do not induce the same ordering.** BaseDreamer leads CLIP-IQA+ in the main comparison, while OpenDWM obtains the highest QTES. Generated traffic changes policy response even when visual-quality measurements remain comparatively stable.

The following values are transcribed from the supplied manuscript, **not measured by running this checkout**. Policy-response results use the manuscript's six-policy panel. On narrow screens, scroll the tables horizontally to view all metric columns.

### Generated trajectories + rendering

| Generator | FVD ↓ | CLIP-IQA+ ↑ | DINO-T ↑ | Epi@3 ↑ | TVC ↑ | TRS ↑ | RPDMS ↑ | Route Comp. ↑ | QTES ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real video (reference) | 0.000 | 0.619 | 0.792 | 0.379 | 0.584 | 1.000 | 0.788 | 0.931 | 50.000 |
| MagicDrive-V2 | 217.146 | 0.437 | 0.744 | 0.179 | 0.447 | 0.782 | 0.718 | 0.870 | 39.509 |
| OpenDWM | **93.383** | 0.531 | 0.730 | **0.222** | **0.543** | 0.782 | 0.724 | 0.878 | **44.079** |
| Panacea | 280.866 | 0.581 | 0.728 | 0.164 | 0.374 | 0.782 | 0.717 | **0.881** | 39.146 |
| DreamForge | 272.880 | 0.573 | 0.705 | 0.168 | 0.495 | 0.782 | **0.729** | **0.881** | 41.311 |
| BaseDreamer | 630.836 | **0.661** | 0.677 | 0.143 | 0.410 | 0.782 | 0.721 | 0.880 | 38.522 |
| HorizonDrive | 178.708 | 0.593 | **0.842** | 0.116 | 0.519 | 0.782 | 0.725 | 0.880 | 40.108 |
| MagicDrive | 378.423 | 0.530 | 0.681 | 0.132 | 0.363 | 0.782 | 0.670 | 0.855 | 37.282 |

*Bold marks the best generator value in each column, excluding the real reference. TRS is shared because these rows use the same generated traffic population; it is not a renderer-specific score.*

<details>
<summary><b>Fixed recorded trajectories + rendering</b></summary>

| Generator | FVD ↓ | CLIP-IQA+ ↑ | DINO-T ↑ | Epi@3 ↑ | TVC ↑ | TRS ↑ | RPDMS ↑ | Route Comp. ↑ | QTES ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MagicDrive-V2 | 216.224 | 0.435 | 0.740 | 0.179 | 0.469 | 1.000 | 0.764 | 0.906 | 39.868 |
| OpenDWM | **93.767** | 0.537 | 0.733 | **0.225** | **0.573** | 1.000 | 0.774 | 0.916 | **44.631** |
| Panacea | 278.398 | 0.581 | 0.729 | 0.163 | 0.408 | 1.000 | 0.770 | **0.922** | 39.637 |
| DreamForge | 279.270 | 0.573 | 0.704 | 0.169 | 0.534 | 1.000 | **0.782** | 0.920 | 41.573 |
| BaseDreamer | 615.745 | **0.662** | 0.674 | 0.144 | 0.448 | 1.000 | 0.777 | 0.921 | 38.765 |
| HorizonDrive | 176.006 | 0.593 | **0.840** | 0.117 | 0.551 | 1.000 | 0.777 | 0.920 | 40.507 |
| MagicDrive | 372.394 | 0.534 | 0.684 | 0.136 | 0.383 | 1.000 | 0.721 | 0.887 | 37.839 |

</details>

<a id="model-zoo"></a>
## 🏛️ Model Zoo

Native model sources are included in the repository. **Checkpoints and isolated model environments are separate resources**; the orchestration package does not replace upstream CUDA dependencies.

| Component | Models / source | Guide |
| --- | --- | --- |
| Traffic generation | [CAT-K](<models/generation/trajectory/CAT-K>) | [Traffic and renderer setup](<docs/generation.md>) |
| Six-view rendering | [BaseDreamer](<models/generation/image/BaseDreamer>), [DreamForge](<models/generation/image/DreamForge>), [HorizonDrive](<models/generation/image/HorizonDrive>), [MagicDrive](<models/generation/image/MagicDrive>), [MagicDrive-V2](<models/generation/image/MagicDrive-V2>), [OpenDWM](<models/generation/image/OpenDWM>), [Panacea](<models/generation/image/Panacea>) | [Native inference modes and conditions](<docs/generation.md>) |
| Driving policies | [UniAD](<models/agents/UniAD>), [VAD](<models/agents/VAD>), [DrivoR](<models/agents/DrivoR>), [SparseDriveV2](<models/agents/SparseDriveV2>), [DiffusionDriveV2](<models/agents/DiffusionDriveV2>), [GoalFlow](<models/agents/GoalFlow>) | [Policy adapters and sensor protocol](<docs/agents.md>) |

The panel enforces camera-only application inputs. GoalFlow and DiffusionDriveV2 retain their frozen multimodal architectures with **fixed zero-LiDAR tensors** in the compatibility mode. See the [policy guide](<docs/agents.md>) for native view subsets, temporal warm-up, checkpoints, and input receipts.

<a id="getting-started"></a>
## 🎯 Getting Started

### 1. Prepare the orchestration environment

Run from the repository root with **Python ≥ 3.9**:

```bash
python -m pip install -e '.[evaluation]'
python -m driveaudit --help
```

GPU inference requires Linux, compatible CUDA extensions, native model environments, and checkpoints. Follow the [native installation guide](<docs/generation.md>) and [model/weight layout](<models/README.md>) before launching generation. Each policy uses its own compatible environment.

### 2. Prepare resources and configuration

| Configuration | Purpose |
| --- | --- |
| [generation.json](<configs/generation.json>) | Participant archive, scene IDs, CAT-K, renderer assets, and native interpreters |
| [pipeline.json](<configs/pipeline.json>) | Generation config, renderer selection, driving policy, organizer mapping, and evaluation outputs |
| [quality.yaml](<configs/quality.yaml>) | Visual-quality evaluation settings |

Paths inside JSON configurations resolve **relative to the configuration file**. Replace the example resource paths or provision local resource links under `work/resources/`. The generation side needs the participant archive, native environments, CUDA toolkits, and model weights. Organizer evaluation additionally needs the real-scene mapping, reference assets, evaluation environment, and quality-model dependencies.

### 3. Inspect and submit the two-scene example

The example uses **OpenDWM + DrivoR** on `case_000001` and `case_000002`:

```bash
# Inspect the stage commands before submitting GPU work.
python -m driveaudit run --config configs/pipeline.json --plan

# Submit the staged pipeline to the configured Slurm cluster.
bash scripts/submit_demo.sh configs/pipeline.json
```

Adjust Slurm partitions, GPU types, and resource paths to your cluster. Use a distinct evaluation output directory for each renderer. Participants can stop after `render`; recorded future RGB, reference traffic, and private identity mappings belong to the organizer's separate reference/evaluation stages.

For standalone traffic generation, rendering, reuse, and resume options, see the [generation guide](<docs/generation.md>).

<a id="paired-policy-evaluation"></a>
## ⚖️ Paired Policy Evaluation

The same frozen policy reads generated observations and their time-matched recorded counterparts. Both branches share scene identities, timestamps, public ego odometry, calibration, commands, and seeds; generated and reference traffic remain bound to their respective branches.

### Full six-policy panel

After generated RGB, reference staging, and quality measurements are ready:

```bash
bash scripts/submit_panel.sh configs/pipeline.json
```

The panel runs all six policies on both branches and writes results under the configured `output/panel/`. To inspect an individual policy invocation:

```bash
python -m driveaudit panel infer --config configs/pipeline.json \
  --policy goalflow --source generated --plan
```

Formal QTES requires common valid paired windows across all six policies, local quality evidence, and population-level TRS for the selected scenes. The evaluator checks model/configuration fingerprints, input-image receipts, temporal history, sensor compatibility, and common support. Reports include paired policy scores, TE, QTES, quality retentions, and coverage.

> [!NOTE]
> **A single-policy diagnostic is not formal panel QTES.** The DrivoR demonstration writes separate diagnostic outputs under the configured `output/paired/`; missing panel captures or quality evidence must not be presented as a complete benchmark score. A two-scene result describes those scenes, not the full benchmark.

See the [evaluation protocol](<docs/evaluation.md>) for standalone scoring commands, support requirements, and output fields.

<a id="closed-loop-testing"></a>
## 🔁 Closed-loop Testing

This branch connects **CAT-K → six-view rendering → DrivoR → ego execution → updated traffic history**. At each round, the runner executes the first **0.5 s** of the plan in five **0.1 s** substeps, then regenerates observations from the updated state. The goal is derived from the final public ego condition, with an absorbing terminal state after arrival.

Set `PUBLIC_SCENE` to an extracted participant scene and `RENDER_SCENE` to the corresponding staged renderer-condition scene. With weights and native environments configured, run:

```bash
python closed_loop/configure.py --renderer basedreamer \
  --participant-scene "$PUBLIC_SCENE" --render-scene "$RENDER_SCENE" \
  --output work/closed-loop/config.json

# Run inside a suitable Slurm GPU allocation; use a fresh output directory.
python closed_loop/run_goal.py --config work/closed-loop/config.json \
  --output work/closed-loop/run-01 --steps 2

python closed_loop/verify_run.py work/closed-loop/run-01
```

BaseDreamer, DreamForge, HorizonDrive, MagicDrive, and Panacea use two GPUs in the default closed-loop layout. OpenDWM and MagicDrive-V2 use five, including four distributed rendering workers. The verifier checks execution traces, next-round poses, image fingerprints, and terminal-state behavior.

**Verification establishes feedback-chain consistency, not a complete paper score.** The manuscript separately reports a targeted 20-scenario multi-policy closed-loop study; it is distinct from the main 550-scenario non-reactive comparison above. See the [closed-loop run and verification guide](<closed_loop/README.md>) for resource details.

<a id="documentation"></a>
## 🗂️ Documentation

| Task | Entry point |
| --- | --- |
| Prepare participant inputs and export submissions | [Participant guide](<docs/participant.md>) |
| Install CAT-K and seven renderers; generate future RGB | [Generation guide](<docs/generation.md>) |
| Configure six policies, assets, and camera inputs | [Policy guide](<docs/agents.md>) |
| Understand paired metrics, coverage, and panel scoring | [Evaluation guide](<docs/evaluation.md>) |
| Run and verify action-feedback testing | [Closed-loop guide](<closed_loop/README.md>) |
| Locate native sources and unified model weights | [Model and weight guide](<models/README.md>) |
| Trace manuscript figures and result provenance | [README asset notes](<assets/readme/README.md>) |

<details>
<summary><b>Repository layout</b></summary>

```text
driveaudit/
├── generation/       CAT-K and seven-renderer orchestration
├── agents/           Native policy services and paired replay
├── organizer/        Reference pairing and quality/traffic measurements
└── evaluation/       Policy response and quality-aware panel scoring
closed_loop/          Action-feedback runner and verification
models/
├── generation/       Native traffic and image-generation sources
└── agents/           Six native driving-policy implementations
configs/              Two-scene example and quality settings
scripts/              Installation and Slurm entry points
tests/                Data-contract and scoring-protocol regression tests
vendor/               Generation adapters and evaluation kernels
```

Runtime environments and generated outputs belong outside the tracked source tree or under ignored `work/`. Checkpoints are provisioned separately under ignored `weights/`.

</details>

<a id="acknowledgements-and-terms"></a>
## 🤝 Acknowledgements and Terms

DriveAudit builds on the work of the upstream traffic-generation, rendering, driving-policy, and evaluation communities, including [NAVSIM](https://github.com/autonomousvision/navsim) and the models listed in the [Model Zoo](#model-zoo). README organization is inspired by [SimScale](https://github.com/OpenDriveLab/SimScale).

Upstream code retains the license and notices supplied in its corresponding model/vendor directory. Source datasets, pretrained weights, and derived data remain subject to their original usage and redistribution terms; this README does not assign a new blanket license to those assets. Benchmark results do not establish real-world deployment safety, and responsibility estimates are model-based diagnostics, not legal-liability determinations.

**Paper and citation.** The supplied manuscript is titled *Beyond Visual Realism: Benchmarking Generated Driving Scenarios as Policy Tests* and currently lists anonymous authors. Public paper, project-page, dataset-download, and final citation metadata are not provided here; links and BibTeX should be added once confirmed.

<div align="right">

[↑ Back to top](#top)

</div>

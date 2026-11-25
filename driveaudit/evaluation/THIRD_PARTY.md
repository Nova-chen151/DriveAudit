# Third-party runtime dependencies

The six driving policies have source code and inference entry points under `models/agents/<model>/`. Their default weight directory is `weights/agents/<model>/`. Set `DRIVEAUDIT_AGENT_SOURCES` and `DRIVEAUDIT_AGENT_ASSETS` to use external directories. Older evaluation configurations also support the `DRIVEAUDIT_DEPENDENCIES/DrivingAgents` layout. Each model directory contains its license. See the [adapter notices](../../models/agents/NOTICE.md).

DINOv2 uses a separate Torch Hub source tree and a configured checkpoint under its Apache-2.0 license. I3D, CLIP, CLIP-IQA+, and LoFTR require the weights specified in the evaluation configuration. If required assets are missing, report the corresponding scores as unavailable.

BEVFormer is an optional perception diagnostic outside the nine core metrics. To run it, specify its repository, configuration, and checkpoint. The optional Onsite Ego Testing distribution reference supplements the current ground-truth inputs. The table distinguishes runtime dependencies from projects cited for their methods or conventions.

| Project | Source | Use in DriveAudit | License observed |
|---|---|---|---|
| CARS | [CARS-code](https://github.com/RoboSafe-Lab/CARS-code) | FSM, CC-JP and RSS replay collisions to assess avoidability. | [Apache-2.0](driveaudit_eval/vendor/cars/LICENSE) · [Source notice](driveaudit_eval/vendor/cars/NOTICE.md) |
| OnSite Unified Train-Test Scenario Generation Track Evaluation Tool | https://github.com/Nova-chen151/Onsite_UT2SG_Testing | Observability-based traffic-rule selection and geometry conventions, implemented in the DriveAudit traffic-compliance module; no runtime import | No LICENSE or NOTICE present anywhere in supplied source snapshot; do not infer a license from another OnSite repository |
| Onsite Ego Testing | https://github.com/Nova-chen151/Onsite_Ego_Testing | Historical trajectory/evaluation reference; optional distribution-reference dataset; no runtime Python import | BSD-3-Clause, Copyright (c) 2026 Nova-chen151; complete notice below |
| CAT-K | External `catk_root` configured in `tools/configs/catk.yaml`; checkpoint and token vocabulary stay outside the core package | Existing trajectory generator and checkpoint class definitions; not imported for cached TRS reproduction | NVIDIA Source Code License in the supplied CAT-K repository; not covered by this project's Apache-2.0 license |
| BEVFormer | https://github.com/fundamentalvision/BEVFormer | Optional image-perception diagnostic, external source/config/checkpoint paths | Apache-2.0 in supplied source |
| ACT-Bench | https://github.com/turingmotors/ACT-Bench | Conceptual motion/instruction comparison; local adapted metrics are not its official score | Apache-2.0 in supplied source |
| DrivingGen | https://github.com/youngzhou1999/DrivingGen | Trajectory alignment and distribution references; local implementations do not import its repository | Apache-2.0 in supplied source |
| WorldLens | https://github.com/worldbench/WorldLens | Benchmark organization and temporal/multiview consistency concepts; no repository import | Apache-2.0 in supplied source |

The OnSite UT2SG reference paths were `av_ego_scoring/EvaluationSystem-onsite2025/evaluateUtils/standard_parameter.py` and `TrafficRule_compliance/{detect_behaviors.py,rules_dict.py,pipeline.py,get_speed_limit_function.py}`. DriveAudit uses its own Trajectory schema and map context. Removing the downloaded repository does not remove the attribution or resolve the pre-existing absence of a license declaration. The local traffic module's source credit is retained; no new UT2SG source is vendored during this cleanup.

## Onsite Ego Testing complete license

```text
BSD 3-Clause License

Copyright (c) 2026, Nova-chen151

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

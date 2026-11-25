# Third-party runtime dependencies

The only embedded third-party algorithm is the minimal Apache-2.0 CARS subset in `driveaudit_eval/vendor/cars/`. Its 15 original definitions cover FSM, CC-JP LaneAware, RSS, buffered collision checks and finite-horizon replay. It requires NumPy and Python `abc`; it does not require CARS models, training code, HDF5 loaders, CTG, Pplan, or trajdata. See its LICENSE, NOTICE.md and WHY_REQUIRED.md.

CARS 是正式运行的条件依赖：traffic evaluator 的碰撞责任链与 episode terminal collision 链调用 cars_adapter，再由 evidence/attribution 形成责任证据。有效 CARS 可避免性证据会影响责任比例；不可用时仍保留其他轨迹证据。不能因为无碰撞窗口不调用或部分场景 unavailable 就声称 runtime unused。

保留的四文件为 responsibility_models.py、LICENSE、NOTICE.md、WHY_REQUIRED.md；它们提供控制器及其必要的碰撞/回放辅助函数。删除该源文件会破坏启用 CARS 的正式碰撞证据路径，因此保留最小 vendor，而非整个上游仓库。责任公式、证据权重和源字节不因整理改变。

六个自车模型的源码与推理入口位于 `models/agents/<模型名>/`。权重默认从 `weights/agents/<模型名>/` 读取。可用 `DRIVEAUDIT_AGENT_SOURCES` 和 `DRIVEAUDIT_AGENT_ASSETS` 指定外部目录。旧版评测配置仍支持 `DRIVEAUDIT_DEPENDENCIES/DrivingAgents` 布局。各模型许可证位于对应源码目录，另见[适配器许可说明](../../models/agents/NOTICE.md)。DINOv2 使用独立 Torch Hub 源及配置 checkpoint，遵循其 Apache-2.0 许可。I3D、CLIP、CLIP-IQA+、LoFTR 等实际模型权重按配置提供，缺少资源不得伪造分数。

BEVFormer 是外部可选感知诊断，不是九项核心指标的必需模型；研究配置必须明确 repo/config/checkpoint。Onsite Ego Testing 分布参考是可选外部数据，不替代当前 input-gt。下列概念/来源参考不构成当前 Python runtime import，保留署名不意味着其完整仓库必须随软件分发。

| Project | Source | Use in DriveAudit | License observed |
|---|---|---|---|
| CARS | https://github.com/RoboSafe-Lab/CARS-code | Exact controller/replay subset in `driveaudit_eval/vendor/cars/responsibility_models.py`; counterfactual responsibility evidence | Apache-2.0; full LICENSE and modification NOTICE next to source |
| OnSite Unified Train-Test Scenario Generation Track Evaluation Tool | https://github.com/Nova-chen151/Onsite_UT2SG_Testing | Observability-based traffic-rule selection and geometry conventions, implemented in the DriveAudit traffic-compliance module; no runtime import | No LICENSE or NOTICE present anywhere in supplied source snapshot; do not infer a license from another OnSite repository |
| Onsite Ego Testing | https://github.com/Nova-chen151/Onsite_Ego_Testing | Historical trajectory/evaluation reference; optional distribution-reference dataset; no runtime Python import | BSD-3-Clause, Copyright (c) 2026 Nova-chen151; complete notice below |
| CAT-K | External `catk_root` configured in `tools/configs/catk.yaml`; checkpoint and token vocabulary stay outside the core package | Existing trajectory generator and checkpoint class definitions; not imported for cached TRS reproduction | NVIDIA Source Code License in the supplied CAT-K repository; not covered by this project's Apache-2.0 license |
| BEVFormer | https://github.com/fundamentalvision/BEVFormer | Optional image-perception diagnostic, external source/config/checkpoint paths | Apache-2.0 in supplied source |
| ACT-Bench | https://github.com/turingmotors/ACT-Bench | Conceptual motion/instruction comparison; local adapted metrics are not its official score | Apache-2.0 in supplied source |
| DrivingGen | https://github.com/youngzhou1999/DrivingGen | Trajectory alignment and distribution references; local implementations do not import its repository | Apache-2.0 in supplied source |
| WorldLens | https://github.com/worldbench/WorldLens | Benchmark organization and temporal/multiview consistency concepts; no repository import | Apache-2.0 in supplied source |

The OnSite UT2SG reference paths were `av_ego_scoring/EvaluationSystem-onsite2025/evaluateUtils/standard_parameter.py` and `TrafficRule_compliance/{detect_behaviors.py,rules_dict.py,pipeline.py,get_speed_limit_function.py}`. DriveAudit uses its own Trajectory schema and map context. Removing the downloaded repository does not remove the attribution or resolve the pre-existing absence of a license declaration. The local traffic module's source credit is retained; no new UT2SG source is vendored during this cleanup.

CARS original source comments credit Mattas et al. (2022), UN R157 Annex 4 Appendix 3, Shalev-Shwartz et al. (2017), and consultation of JRC-FSM. The copied definitions retain those comments. Vendored LICENSE、NOTICE.md 与 WHY_REQUIRED.md 保留来源和实际运行理由。

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

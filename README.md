# DriveAudit 闭环测试

本分支的入口是 [闭环运行与验证](closed_loop/README.md)。CAT-K 交通、六视角渲染与 DrivoR 通过每轮 0.5 秒的执行结果更新下一轮场景。下文保留基础生成和配对回放的使用方法。

DriveAudit 用选手输入包生成未来交通轨迹和六视角视频，再让冻结的主车策略读取这些图像，按同一源场景的真实回放计算配对指标。仓库把交通生成、渲染、主车策略、评测分成四个可独立运行的模块，目录组织参考 [NAVSIM](https://github.com/autonomousvision/navsim)。

```text
选手输入包 ─→ CAT-K 轨迹 ─→ 渲染条件 ─→ 六视角未来 RGB ─→ 主车规划
                                      主办方真实回放 ─→ 同一主车规划
                        配对场景、两侧规划、视觉质量和总体 TRS ─→ 评分
```

## 运行两个 nuScenes 场景

按 [生成说明](docs/generation.md) 安装原生模型至 `work/native`。从仓库根目录执行。示例采用 OpenDWM、DrivoR 和场景 `case_000001`、`case_000002`。

```bash
cd driveaudit
python -m driveaudit run --config configs/pipeline.json --plan
bash scripts/submit_demo.sh configs/pipeline.json
```

选手只需 [generation.json](configs/generation.json)，其中指定输入包、场景、CAT-K、渲染权重和运行环境。主办方用 [pipeline.json](configs/pipeline.json) 引用该配置，选择渲染器并指定主车、真实回放映射及评分输出。视觉评价参数见 [quality.yaml](configs/quality.yaml)。

配置路径相对于配置文件。运行前可改成安装路径，或在仓库的 `work/resources/` 准备本机资源链接。生成侧使用 `participant.tar.gz`、`envs/<模型环境>/bin/python` 和 `cuda/cuda-<版本>`。主办方另需 `scene_mapping.json`、`evaluation-env`、`agent-deps`（主车运行依赖）和 `cache`（含 DINO 源码）。检查点位于仓库的 `weights/`。Slurm 脚本中的分区和 GPU 类型应与本机集群一致。

第一条命令展开各阶段的实际调用。第二条按阶段申请 CPU、L40 和内存，并打印作业号。各阶段可在合适的计算节点单独运行，例如只执行交通预测可用 `--start traffic --stop traffic`。生成器、运行环境和输出目录在 [流水线配置](configs/pipeline.json) 和 [生成配置](configs/generation.json) 中设置。比较不同渲染器时，为每个渲染器指定单独的评分输出目录。已有轨迹可通过生成配置的 `catk_run` 复用，已有六视角图像可通过流水线配置的 `conditions` 复用，并从 `policy-generated` 开始。主办方配对目录按单次实验创建；已有配对完成后，可从 `quality` 或 `score` 继续。

| 模块 | 输入与输出 | 使用说明 |
| --- | --- | --- |
| 交通生成与渲染 | 公开历史、地图和未来自车位姿 → CAT-K 交通 → 每个场景 80 帧 × 6 视角 RGB | [七种渲染器与原生环境](docs/generation.md) |
| 主车策略 | 生成 RGB 和允许的自车状态 → 按决策时刻记录预测轨迹、相机与权重指纹 | [六种主车策略](docs/agents.md) |
| 主办方配对与评分 | 真实回放与生成结果 → 质量分量、TRS、RPDMS、路线完成率与 QTES | [评价协议和运行入口](docs/evaluation.md) |

选手侧运行到 `render` 即可交付。真实未来 RGB、交通真值与匿名身份映射仅由主办方在独立真实基线和评价阶段读取。这里的主车评测采用固定图像和交通的回放协议，每个决策时刻按记录的自车状态预测未来轨迹。闭环测试还要求按执行后位姿重新生成下一轮观测。

## 目录

```text
driveaudit/generation/    CAT-K 与七种渲染器的统一入口
driveaudit/agents/        主车输入适配、原生服务与逐时刻推理
driveaudit/organizer/     生成结果与参考配对、视觉和交通测量
driveaudit/evaluation/    策略响应、质量保留及配对评分
models/generation/trajectory/   交通轨迹模型 CAT-K
models/generation/image/        七个六视角图像生成模型，每个模型一个目录
models/agents/                  六个主车模型
weights/                  HPC 统一权重目录
vendor/                   适配实现及评价内核
configs/                  两场景示例与评价参数
scripts/                  Slurm 任务入口
tests/                    数据和评分协议回归检查
```

`python -m driveaudit` 可在检出的仓库中直接运行。模型继续使用各自兼容的 CUDA、Python 环境和检查点，具体路径见各模块说明。参与者数据包、检查点、GPU 环境和运行产物存放于 Git 工作树之外或忽略的 `work/` 下。原有上游代码的许可证保存在对应 `vendor/` 子目录。

## 运行完整六策略评分

同一批生成图像、真实回放和质量测量就绪后，运行下面的命令。若上面的两场景任务仍在运行，可把其最后的 `score` 作业号作为第二个参数。脚本在两侧各运行六种策略，并把逐决策记录交给正式面板评分器。

```bash
bash scripts/submit_panel.sh configs/pipeline.json
```

结果在流水线配置的 `output/panel/`。单独调试某个接入可以运行 `python -m driveaudit panel infer --config configs/pipeline.json --policy goalflow --source generated --plan` 查看完整命令。正式 QTES 需要六策略共同有效的配对决策、局部质量和所选场景的总体 TRS。DrivoR 单策略诊断结果单独保存在 `output/paired/`。


[选手输入包与标准提交目录导出](docs/participant.md)。


[模型源码位置、统一权重与安装方式](models/README.md)。

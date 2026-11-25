# 选手输入、CAT-K 与未来 RGB

[![English](https://img.shields.io/badge/English-777777?style=flat-square)](generation.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](generation.zh-CN.md)

[首页](../README.zh-CN.md) · [评估指南](evaluation.zh-CN.md#evaluation-scope)

请在 DriveAudit 仓库目录中运行下列命令。输入是 GenBench 的 `nuscenes_ego_conditioned_v1` 选手归档包。CAT-K 根据已观测的交通与地图上下文，联合预测未来交通参与者及其自身的自车轨迹，并在选手包的 80 个目标时间戳导出交通参与者。渲染器使用归档包提供的公开自车与相机运动。CAT-K 预测的自车轨迹可用于诊断。

<a id="install-the-native-inference-source-offline"></a>

## 离线安装原生推理源码

原生源码位于 [CAT-K](../models/generation/trajectory/CAT-K) 和[渲染器目录](../models/generation/image)。安装器将这些源码、选手数据适配器、兼容的编译扩展及外部权重整合到 `work/native`。

各模型的推理入口和闭环执行代码位于其模型目录中。选手包读取、渲染条件和帧校验等共用代码位于 `driveaudit/generation/`。[适配文件清单](../driveaudit/generation/bundles/adapters.json) 给出源码与运行时路径的对应关系。

请将以下环境变量设为本地站点的资源位置。

| 变量 | 已有资源 |
| --- | --- |
| `WEIGHTS` | 检查点缓存，包含 `driveaudit/generation/installation.py` 所列的渲染器发布权重，以及 BaseDreamer 的 CLIP 目录。 |
| `CATK_ASSETS` | CAT-K 资源根目录，包含 `checkpoints/clsft_E9.ckpt` 及 `src/smart/tokens` 下的三个 token 词表文件。 |
| `V2_CKPTS`, `V2_PRETRAINED` | MagicDrive-V2 DiT 检查点目录树，以及 CogVideoX VAE/T5 预训练目录树。 |
| `ABI_ROOT` | 兼容的已编译 gen-benchmark 目录树。安装器列出了 BaseDreamer、DreamForge、MagicDrive 和 Panacea 扩展相对于项目的路径。这些是已测试 Linux/Python/CUDA 环境的编译组件。 |
| `CONDA_ENVS` | 各模型使用的原生 Python 环境。 |

```bash
ENGINE_CODE="$PWD/work/native"
python scripts/geninstall_native.py --destination "$ENGINE_CODE" \
  --weights-root "$WEIGHTS" --catk-assets "$CATK_ASSETS" \
  --v2-checkpoints "$V2_CKPTS" --v2-pretrained "$V2_PRETRAINED" \
  --extensions-root "$ABI_ROOT"
python scripts/geninstall_native.py --destination "$ENGINE_CODE" --verify
```

`NATIVE_INSTALL.json` 记录已安装源码和扩展的哈希，以及资源链接。如只需检查源码，可运行 `python scripts/geninstall_native.py --source-only --destination work/native`。来源信息见[源码清单](../driveaudit/generation/bundles/native-sources.tar.json)。

<a id="model-weights"></a>

## 模型权重

使用 `weights/` 作为共享根目录：`catk/` 存放交通模型权重和词表，`renderers/` 与 `magicdrivev2/` 存放渲染器资源，`agents/` 存放驾驶策略资源，`evaluation/` 存放 LoFTR、DINO 和 CLIP-IQA+。配置好的权重库将清单记录在 `weights/inventory.json`；克隆仓库不会下载权重。

运行 `python scripts/collect_weights.py --help`，查看如何导入已有权重库。原生安装器也支持同时使用 `--weight-store WEIGHTS` 和 `--extensions-root ABI_ROOT`。共享权重可能采用硬链接：更新时应以新版本替换文件，而不是原地覆写。修改模型源码后应重新构建原生安装。策略环境见[评估指南](evaluation.zh-CN.md)。

<a id="configure-a-generation-run"></a>

## 配置生成任务

将 `ENGINE_CODE` 设为本文所述的原生引擎目录，`WEIGHTS` 设为检查点缓存，`CONDA_ENVS` 设为原生模型环境目录，`ARCHIVE` 设为选手 tar 包，`WORK` 设为新的运行目录。运行 JSON 中的相对路径以该 JSON 所在位置为基准解析。GPU 推理需要 Linux 文件系统及原生模型安装。

```bash
python -m driveaudit.generation init \
  --engine-code-root "$ENGINE_CODE" --conda-envs "$CONDA_ENVS" \
  --archive "$ARCHIVE" --work-dir "$WORK" --weights-root "$WEIGHTS" \
  --scene case_000001 case_000002 --output run.json
```

安装后提供以下原生目录结构。

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

MagicDrive 和 DriveAudit 实现使用各自对应的视频目录树及编译依赖。`--weights-root` 创建的内联 `models` 部分记录所有环境与检查点路径，可按其他存储布局修改。全新推理不需要先前的实验目录或已生成图像。

在原始 HPC 系统上，还需传入 `--cuda-prefix /share/apps`，为各渲染器选择原始 CUDA 工具包。省略 `--weights-root` 会选择已有安装配置，并将 `models_json` 设为安装目录的同级 `configs/models.json`。如果安装位置发生变化，请在模型注册表的私有副本中更新各模型的 Python 和检查点路径。各适配器通过以下注册表字段解析解释器与检查点。

| 模型 | Python 与检查点 |
| --- | --- |
| BaseDreamer, MagicDrive | `python.renderer`, `render_checkpoint`, `sd15` |
| DreamForge | 相同字段，另加 `single_checkpoint` |
| HorizonDrive | `python.renderer`, `render_checkpoint`, `wan_checkpoint`。将 `horizondrive-vae.pkl` 放在渲染器检查点旁。 |
| OpenDWM | `python.renderer`, `render_checkpoint`, `base_model` |
| Panacea | `python.renderer`, `render_checkpoint`, `clip_checkpoint` |
| MagicDrive-V2 | 运行 JSON 中的 `v2_python`。原生 `MagicDrive-V2/rd/ckpts` 和 `rd/pretrained` 目录树提供 DiT、CogVideoX VAE 和 T5 权重。 |

如需使用独立安装的 CAT-K 资源，将 `catk_official` 设为包含 `src/smart` 与 `checkpoints/clsft_E9.ckpt` 的目录，将 `catk_python` 设为其 Python 可执行文件。默认源码和词表校验和位于 `models/generation/trajectory/CAT-K/official_source.sha256` 与 `original_config.yaml`。除非运行 JSON 提供 `traffic_seed`，否则 CLI 保留原始确定性种子 20260914。

<a id="generate-and-stage-traffic"></a>

## 生成并暂存交通数据

```bash
srun -p L40 --gres=gpu:l40:1 -c 4 --mem=16G \
  python -m driveaudit.generation traffic --config run.json
python -m driveaudit.generation stage --config run.json
```

交通阶段校验选定的归档成员，运行预训练 CAT-K 解码器，验证原生提交 CSV，并在发布 `WORK/traffic/COMPLETED.json` 前加载各渲染器输入。交通参与者输出位于 `WORK/traffic/submission/scenes/<case>/future/future_trajectories.csv`。报告区分预测的交通参与者、保持静止的物体及不活跃的历史轨迹。DriveAudit 适配器使用的局部虚拟 LiDAR 配置在每个已暂存场景中声明。

如需暂存先前已完成的交通任务，在运行 JSON 中将 `catk_run` 设为该任务目录，再执行 `stage`，即可复用其预测。暂存阶段检查归档摘要、交通参与者 CSV 摘要及公开自车轨迹的逐字节一致性。它对每个暂存场景文件（包括历史 RGB、地图与标定）计算哈希，再通过原子重命名发布 `WORK/conditions`。渲染时会在加载模型前验证这些场景摘要。如需更改已有任务的条件，必须使用新的工作目录。

<a id="render-future-rgb"></a>

## 渲染未来 RGB

```bash
for model in basedreamer dreamforge horizondrive magicdrive opendwm panacea; do
  srun -p L40 --gres=gpu:l40:2 -c 12 --mem=96G \
    python -m driveaudit.generation render --config run.json --model "$model" || exit 1
done
srun -p L40 --gres=gpu:l40:4 -c 24 --mem=256G \
  python -m driveaudit.generation render --config run.json --model magicdrivev2
```

示例为前六个模型各分配两张 L40 GPU，为 MagicDrive-V2 分配四张。请按集群情况调整。日志和验证输出位于 `WORK/logs/<model>/attempt-*/`。重新运行遵循适配器原生的恢复检查；更改条件或检查点后应使用新的工作目录。

输出位于 `WORK/conditions/predictions/<model>/nuscenes/scenes/<case>/future/rgb/<camera>/<frame>.jpg`。验证要求原生完成元数据有效，并对六个相机、帧 10–89 的每张 400 × 224 RGB JPEG 进行解码。每个模型在每个场景生成 480 张图像。比较新的生成器设置或检查点时必须使用新的工作目录，避免意外复用结果。

| 模型 | 原生推理模式 | 已测试 Python / PyTorch |
| --- | --- | --- |
| BaseDreamer | 每场景一个工作进程，发布版单参考帧流水线 | 3.9 / 1.10.2+cu113 |
| DreamForge | 每场景一个工作进程，20 个去噪步骤 | 3.9 / 2.0.0+cu118 |
| HorizonDrive | Accelerate 分布式评估，bf16 | 3.10 / 2.8.0+cu128 |
| MagicDrive | 每场景一个工作进程，发布版视频流水线 | 3.8 / 1.10.2+cu113 |
| OpenDWM | 两个 rank 的 FSDP，40 个去噪步骤 | 3.10 / 2.5.1+cu121 |
| Panacea | 两个 rank，25 步，滚动生成参考帧及有序未来槽位 | 3.8 / 1.13.1+cu117 |
| MagicDrive-V2 | 四张 GPU，89 帧序列，9 帧历史 RGB 与 80 帧生成图像 | 3.10 / 2.4.0 |

Panacea 将参考帧放在模型的最后一个槽位，按时间顺序提供未来目标，并将每个窗口最后一张生成帧作为下一个参考帧。

<a id="what-conditions-reach-each-renderer"></a>

## 各渲染器实际接收的条件

交通参与者 CSV 提供各目标时间戳的包围盒中心、尺寸、朝向及类别。地图来自选手归档包。公开自车位姿与标定的相机配置将包围盒及地图变换到各目标相机坐标系。共享加载器验证给定的相机到锚点变换记录。下列渲染器根据自车位姿及标定的相机到自车变换推导相机几何。七个模型均接收场景提示词。

| 渲染器 | 本配置中的 RGB 参考 | 推理接收的交通、地图与运动信息 |
| --- | --- | --- |
| BaseDreamer | 第 9 帧初始化首步。随后每步使用前一步生成的六视角帧。 | 流水线接收四通道鸟瞰（BEV）道路/地图栅格、14 通道投影布局、可见的 3D 包围盒角点/类别/掩码 token、相机矩阵及目标相对参考帧的自车位姿。 |
| DreamForge | 两个初始参考槽位均为第 9 帧。后续窗口使用前一窗口最后两张生成帧。 | 流水线接收 BEV 地图、逐相机地图/交通参与者布局、3D 包围盒 token 和可见性掩码、标定投影矩阵，以及根据公开自车轨迹在局部坐标系推导的 CAN 总线风格运动信息。 |
| HorizonDrive | 历史帧 0–9 加上重复的第 9 帧填满 11 个参考槽位。 | 配置的条件投影接收逐相机 bbox 和高精地图图像，以及连续自车运动动作（局部 delta x、delta y 和 delta yaw）。标定用于投影栅格。 |
| MagicDrive | 仅布局条件推理。此适配器的两个窗口均从噪声生成。 | 两个有重叠的 61 帧窗口使用自车对齐的语义 BEV 地图、可见 3D 包围盒角点/类别/掩码和相机矩阵。公开自车运动通过逐帧地图和包围盒坐标进入模型。 |
| OpenDWM | RGB 帧 7–9 初始化滚动生成。后续 19 帧窗口保留三帧重叠。 | 原生六通道图像适配器在所有帧接收彩色投影包围盒边缘与公开高精地图轮廓。其显式视角模型还接收归一化相机内参、与公开自车位姿复合的相机配置外参，以及相邻视角掩码。 |
| Panacea | 第 9 帧初始化首个参考帧。后续窗口使用各自最后一个生成目标帧。 | 19 通道全景图组合三个投影包围盒通道、十个按类别划分的包围盒深度通道、三个矢量地图通道及三个标定视线通道。每个未来槽位接收通过公开自车位姿变换后的独立布局。 |
| MagicDrive-V2 | VAE 将帧 1–9 编码为 89 帧序列中的固定三个潜变量前缀。 | 去噪器接收经过扭曲变换的语义 BEV 地图、3D 角点/类别/可见性张量、相机标定、相对自车位姿及实测片段 FPS。由于检查点没有未知图层掩码，缺失地图通道填零。 |

OpenDWM 使用投影交通包围盒、可用高精地图轮廓及标定相机几何作为所有帧的条件。选手地图没有原生车道多边形；不支持的包围盒类别将被跳过。此配置不使用单独的数值动作/FPS 向量。详见[数据集适配器](../models/generation/image/OpenDWM/src/dwm/datasets/genbench_gt2.py)与[栅格化器](../models/generation/image/OpenDWM/src/dwm/datasets/genbench_gt2_renderer.py)。

运行几何诊断，对比包含和不包含交通参与者时的包围盒与地图条件。

```bash
"$CONDA_ENVS/opendwm/bin/python" scripts/gencheck_opendwm_conditions.py \
  --conditions "$WORK/conditions" --engine-code-root "$ENGINE_CODE"
```

<a id="slurm-submission"></a>

## Slurm 提交

Slurm 调度入口支持三个起点。新建交通任务时使用 `bash scripts/submit_demo.sh CONFIG fresh`；使用 `staged` 可复用已验证的 CAT-K 条件目录，渲染并评估另一生成器；使用 `rendered` 可评估已有渲染结果。对每个额外渲染器，请在 `CONFIG` 中选择其 ID，并指定独立的评估输出目录。所有渲染器可共享同一生成工作目录与交通数据。

<a id="participant-inputs-and-submission"></a>

## 选手输入与提交

选手包由生成配置的 `archive` 指定。包内有 250 个 nuScenes 场景、赛题说明、提交格式和公开校验器。示例选用前两个公开场景编号。

每个场景提供帧 0–9 的六视角 RGB、历史交通参与者、地图、相机标定及时间戳。帧 10–89 的自车和相机位姿作为给定条件。交通轨迹采用第 9 帧自车坐标系，x 向前、y 向左、z 向上，长度以米计。采样时间读取 `timeline.json`。CAT-K 预测其他交通参与者，再由所选渲染器生成对应的未来图像。

<a id="export-the-submission-directory"></a>

## 导出提交目录

从仓库根目录运行，指定解压后的公开选手目录、渲染条件目录和实际方法声明。`PARTICIPANT_DIR` 指向解压后的选手目录，`SUBMISSION_METADATA` 指向自己的方法声明 JSON。字段按选手包的提交规范填写。

```bash
python -m driveaudit.submission \
  --inputs "$PARTICIPANT_DIR" \
  --conditions work/two-scene-generation/conditions --renderer opendwm \
  --cases case_000001 case_000002 \
  --metadata "$SUBMISSION_METADATA" \
  --output work/two-scene-submission
```

每个场景导出一份匿名交通轨迹 CSV，以及六路各 80 张 400 × 224 的 RGB JPEG。文件名和符号链接限制遵循选手包自带的校验器。程序对所选的每个场景调用该校验器，全部通过后发布提交目录。两场景演示只覆盖所选编号，正式全量提交还需覆盖主办方发布的完整场景清单。

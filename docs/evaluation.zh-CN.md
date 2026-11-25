# 驾驶策略与配对评估

[![English](https://img.shields.io/badge/English-777777?style=flat-square)](evaluation.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](evaluation.zh-CN.md)

[首页](../README.zh-CN.md) · [生成指南](generation.zh-CN.md#participant-inputs-and-submission)

同一个冻结的相机驾驶策略分别在生成观测与时间匹配的实录观测上运行。两个分支共享时间戳、地图、公开自车里程计、指令和种子；碰撞与 TTC 评分则分别使用各自的交通数据。

<a id="evaluation-scope"></a>

[driveaudit/evaluation/driveaudit_eval](../driveaudit/evaluation/driveaudit_eval) 存放评分引擎。[DriveAudit 评分入口](../driveaudit/evaluation/score.py) 调用同一套实现，为真实回放和生成场景计算指标。

## 评估范围

论文报告的是 **非反应式回放（non-reactive replay）** 下的 550 场景、六策略基准。额外的 20 场景闭环研究采用独立协议。仓库示例使用两个 nuScenes 场景，结果仅描述所选场景。单策略诊断或已验证的闭环反馈链均不等同于正式六策略 QTES。

<a id="policy-setup-and-inference"></a>

## 策略配置与推理

DriveAudit 包含先前评估器使用的六个 DrivingAgents 源码检出版本中的模型和特征构建源码。由于 NAVSIM 分支使用相同的导入名称，而 MMDetection 包固定了不同版本，因此各策略均使用独立 Python 环境。

| 策略 | 随仓库提供的源码 | 原生相机输入 | 服务名称 |
| --- | --- | --- | --- |
| DrivoR | `models/agents/DrivoR` | 前、后、左前、右前 | `drivor` |
| UniAD | `models/agents/UniAD` | 六视角 | `uniad` |
| VAD | `models/agents/VAD` | 六视角 | `vad` |
| SparseDriveV2 | `models/agents/SparseDriveV2` | 左前、前、右前 | `sparsedrivev2` |
| DiffusionDriveV2 | `models/agents/DiffusionDriveV2` | 左前、前、右前 | `diffusiondrivev2` |
| GoalFlow | `models/agents/GoalFlow` | 左前、前、右前 | `goalflow` |

<a id="drivor-paired-replay"></a>

### DrivoR 配对回放

请在 CUDA 节点上使用 DrivoR 环境。将 `DRIVEAUDIT_AGENT_ASSETS` 设为包含六个区分大小写的模型目录的父目录。DrivoR 默认路径为 `DrivoR/weights/drivor_Nav1_25epochs.pth` 和 `DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`，两个路径均可通过 CLI 覆盖。如果环境中没有相应包，将 `NUPLAN_DEVKIT_ROOT` 设为 nuPlan devkit 源码检出目录。实测环境为 Python 3.9、PyTorch 2.1/CUDA 12.1 和 timm 1.0.15。源码发布版的安装与权重下载说明见 [DrivoR 上游 README](../models/agents/DrivoR/UPSTREAM.md)。

从仓库根目录运行以下两条命令。`PUBLIC_SCENES` 包含公开场景目录。`GENERATED_SCENES` 包含对应渲染器的场景目录，每个目录包含 `future/rgb/CAM_*/000010.jpg` 及后续帧。`REAL_MAPPING` 是主办方提供的 JSON 数组，记录 `public_scene` 和 `source_path`。每个源路径解析为对应的参考场景目录。

```bash
python -m driveaudit.agents.predict --source generated \
  --public-scenes "$PUBLIC_SCENES" --rgb-scenes "$GENERATED_SCENES" \
  --cases case_000001 case_000002 --output work/ego/drivor/generated.jsonl

python -m driveaudit.agents.predict --source real \
  --public-scenes "$PUBLIC_SCENES" --real-mapping "$REAL_MAPPING" \
  --cases case_000001 case_000002 --output work/ego/drivor/real.jsonl
```

DrivoR 使用其发布版的四相机子集，以及标定、速度、加速度和共享驾驶指令。演示默认为 STRAIGHT；可用 `--command` 选择其他共同指令。各分支在公开自车位姿下回放帧 10–89，预测动作不会改变后续观测。回执记录模型/配置与输入图像哈希、原生八位姿轨迹，以及 +0.5 至 +3 s 的六个评分路点（单位为米，x 向前、y 向左）。

<a id="replay-with-any-of-the-six-services"></a>

### 使用六个服务中的任意一个进行回放

托管回放客户端在所选环境中启动本地服务，按时间顺序提交场景，并写入相同的六路点 JSONL 评分格式。它检查实际所需的检查点、锚点和原生配置。客户端 Python 环境需包含 NumPy、SciPy 和 Pillow，并通过 `--service-python` 传入策略解释器。源码目录默认为 `models/agents`。`DRIVEAUDIT_AGENT_SOURCES` 可指向完整兼容模型检出目录的父目录。资源使用 `DRIVEAUDIT_AGENT_ASSETS` 中的原始检查点布局。

```bash
python -m driveaudit.agents.service sparsedrivev2 --preflight
python -m driveaudit.agents.replay --policy sparsedrivev2 \
  --service-python "$SPARSEDRIVE_ENV/bin/python" --source generated \
  --public-scenes "$PUBLIC_SCENES" --rgb-scenes "$GENERATED_SCENES" \
  --cases case_000001 case_000002 --output work/ego/sparsedrivev2/generated.jsonl
```

配对评估时使用相同的策略、指令、资源与场景，切换为 `--source real --real-mapping "$REAL_MAPPING"`，并选择另一个输出文件。将两个 JSONL 文件交给策略评估器。`--max-decisions 1` 仅运行每个场景的首个公开未来决策，用于诊断。完整回放使用所有决策。

UniAD 和 VAD 在首个计分未来帧之前，使用各相机全部十张公开历史图像预热时序状态。其六相机输入使用原始视角顺序、观测标定、公开自车位姿，以及 `participant_render_frame.json` 声明的虚拟相机配置。其右/前坐标输出转换为以米为单位的前/左坐标。其他四个服务接收各自原生特征构建器的封装输入。GoalFlow 和 DiffusionDriveV2 默认采用先前评估器的零 LiDAR 兼容模式。在该模式下，其冻结的多模态架构接收固定的全零张量。

每行托管回放记录均通过 SHA-256 绑定服务适配器、检查点、配置和具名资源。时序记录还对有序公开历史帧及预热实际使用的相机字节生成指纹。纯相机回执描述强制执行的应用协议。客户端还检查两个多模态服务在每次决策中均报告全零 LiDAR 张量。

各策略需要各自的上游包或已编译 CUDA 算子。SparseDriveV2 的 CUDA 源码位于 `models/agents/SparseDriveV2/navsim/agents/sparsedrive/ops`。使用 SparseDriveV2 环境在该目录运行 `python setup.py build_ext --inplace` 构建算子，或将 `DRIVEAUDIT_SPARSE_OPS` 指向已有且 ABI 兼容的算子目录。对于 NAVSIM 模型，可将 `NUPLAN_DEVKIT_ROOT` 指向匹配的外部 nuPlan devkit 检出目录。模型安装与发布文件见各第三方目录的 `UPSTREAM.md`。

专用 DrivoR 回放保留原生八位姿轨迹。六策略面板使用托管回放，以统一传感器、历史数据和模型回执。源码许可证位于各第三方目录，适配器声明位于 `models/agents/NOTICE.md`。

<a id="run"></a>

## 运行

预测及主办方暂存准备完成后，在含有 NumPy、SciPy、Shapely 和 PyYAML 的 Python 环境中，从仓库根目录运行以下命令。

```bash
python -m driveaudit.evaluation \
  --generated work/ego/drivor/generated.jsonl \
  --real work/ego/drivor/real.jsonl \
  --scenes work/evaluation/staged/scenes.json \
  --trs work/evaluation/trs/result.json \
  --quality work/evaluation/quality.json \
  --output work/evaluation/paired
```

相应测量尚未完成时，可省略 `--quality` 或 `--trs`。策略分数和支持计数仍然可用。取得证据后重新运行，以补全综合报告。运行时路径必须由主办方配置。

<a id="inputs-and-checks"></a>

## 输入与检查

- 场景映射为每个公开场景提供一个键，包含绝对 `generated` 与 `real` 场景目录。各场景遵循原生评估格式，包括历史数据、未来交通、按时间索引的自车位姿、可行驶区域栅格和相机帧。
- 每个预测文件为每个公开未来决策提供一行，记录场景、原生 `frame_index`、从零开始的未来 `frame_id`、`reference_time_us`、`source`、`policy`、`request_seed`、`ego_state`、`command`、模型/配置/骨干网络的 SHA-256 摘要，以及各相机的 `rgb_sha256`。评分器验证实际使用的每张 JPEG、共享路线/地图/时间身份、配对决策时间戳、模型身份、给定控制量与种子，并拒绝重复或缺失决策。DrivoR 相机集合为前、左前、右前和后。
- `trajectory_xy` 包含决策自车坐标系中 0.5、1、1.5、2、2.5 和 3 s 时的六个绝对位置，以米为单位，x 向前、y 向左。末尾不足三秒实录未来支持的观测仍保留在推理回执中，并单独计数。
- `--quality` 接收按场景索引的 `generated` 与 `real` 配对原生局部质量记录。各记录提供数据集、场景 ID、`clip_iqa`、`dino_t`、`epi_3px` 和 `tvc` 原始分数、计算与维度状态、实测覆盖率、DINO `temporal_pairs` 及 TVC 分量掩码。原生质量内核检查覆盖率和支持集一致性。
- `--trs` 接收配对总体评估器的 `result.json`。场景身份及候选/参考源哈希必须与所选场景目录树匹配。

<a id="reading-the-results"></a>

## 解读结果

`summary.json` 包含场景分数、输入指纹、配对决策计数及策略身份。`paired.csv` 为每个场景提供一行实录结果和一行生成结果。各分支的 JSON 文件保留窗口分数、碰撞证据，以及回放终止与路线投影轨迹。

RPDMS 使用原生 `score_window` 和基于观测地图的 DAC 协议。每个符合条件的三秒规划贡献安全性、舒适性和路线进度项。发生重叠时会触发责任归因。无法归因或可行驶地图支持不足时，该窗口不计分。两个分支报告的 RPDMS 均值使用完全相同的受支持窗口 ID。单策略 TE 要求至少 80% 的共同覆盖率。每个结果均报告分子与分母。

路线完成度（Route Completion）使用原始实录状态的滚动时域回放，包含终端时域提交、密集碰撞检查，以及对完整实录路线的连续投影。每次决策都将预测前缀重新锚定于实录自车里程计。只有原生投影覆盖率达到冻结阈值时，路线列才可用。交通与图像均来自固定序列。

对于每个场景，原生论文质量函数根据 CLIP-IQA+、DINO-T、Epi@3 和 TVC 计算局部保持率 `B`。支持完整时，`single_policy_QTES_diagnostic` 使用 `100 × TRS_population^0.25 × mean_scene(TE_policy × B)`。两场景 TRS 是总体指标，在表达式中只计入一次。数值诊断要求所有选定场景均有证据。运行单策略演示时，正式六策略 QTES 字段保持为 null。

<a id="full-six-policy-panel"></a>

## 完整六策略面板

面板接收 UniAD、VAD、DrivoR、SparseDriveV2、DiffusionDriveV2 和 GoalFlow 每个策略的两份完整纯相机采集记录。它验证资源/配置及适配器摘要、已预热的时序历史、真实与零 LiDAR 协议、时间戳、里程计、指令和输入图像。这些传感器回执描述应用输入与原生适配器响应。主办方的质量收集器将结果绑定到解码后的场景证据和实测交通文件。请将各策略文件放在共同根目录下的 `<policy>/generated.jsonl` 和 `<policy>/real.jsonl`。本演示使用 `work/ego/panel-input/` 下带有回执的采集记录。可选的 `--capture-dir POLICY=PATH` 可覆盖单个目录。

```bash
python -m driveaudit.evaluation.panel \
  --ego-root work/ego/panel-input \
  --scenes work/evaluation/staged/scenes.json --quality work/evaluation/quality.json \
  --trs work/evaluation/trs/result.json --output work/evaluation/panel --workers 6
```

评估器为每个场景的全部十二种分支/策略组合冻结一个共同窗口掩码，要求至少十个有效窗口和 80% 覆盖率。它在这些索引上计算配对 RPDMS，并将匹配记录传递给原始六智能体 `paired_comparison` 和 `paper_population_v1` 聚合。采集记录缺失或配对未经验证时不会产生数值结果。`panel.json` 记录 TE、QTES、局部保持率、源身份及共同窗口索引。`panel.csv` 为每个场景和策略并列展示实录与生成的 RPDMS、路线结果、TE 和共同支持集。分数仅适用于所选场景。

<a id="qtes"></a>

## QTES

正式面板分数结合配对策略退化（`TE`）、局部质量保持率（`B`）、总体交通真实性（`TRS`）与有效性（`V`）：

$$
\mathrm{QTES}_{\mathcal P}
=100\,\mathrm{TRS}_{\mathcal P}^{1/4}
\frac{1}{|\mathcal P|}\sum_{n\in\mathcal P}V_n\,\mathrm{TE}_n\,B_n.
$$

`B` 使用相对于参考的 CLIP-IQA+、DINO-T、Epi@3 和 TVC。TRS 在所选场景总体上评估，而不是逐帧评估。有效的参考自比较应得到 QTES = 50 和 TE = 0.5；这些是锚点，不是通过/失败阈值。RPDMS 越高表示策略表现越好；QTES 越高表示质量感知测试有效性越强。缺失质量证据或共同支持集无效时，无法产生完整分数。

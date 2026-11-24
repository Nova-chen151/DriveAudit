# 模型源码和权重

生成与主车算法的原生源码按文件纳入 Git，可以直接阅读和修改。生成模型的选手输入适配在 `vendor/generation`，主车输入适配和推理入口在 `driveaudit/agents`。

| 任务 | 模型源码 |
| --- | --- |
| 轨迹生成 | [CAT-K](generation/trajectory/CAT-K) |
| 六视角图像生成 | [BaseDreamer](generation/image/BaseDreamer)、[DreamForge](generation/image/DreamForge)、[HorizonDrive](generation/image/HorizonDrive)、[MagicDrive](generation/image/MagicDrive)、[MagicDrive-V2](generation/image/MagicDrive-V2)、[OpenDWM](generation/image/OpenDWM)、[Panacea](generation/image/Panacea) |
| 主车规划 | [DrivoR](agents/DrivoR)、[UniAD](agents/UniAD)、[VAD](agents/VAD)、[SparseDriveV2](agents/SparseDriveV2)、[GoalFlow](agents/GoalFlow)、[DiffusionDriveV2](agents/DiffusionDriveV2) |

MagicDrive 的模型实现位于 [magicdrive](generation/image/MagicDrive/magicdrive)，运行配置位于 [configs](generation/image/MagicDrive/configs)，BEVFusion、xFormers 和 Diffusers 集中在 [third_party](generation/image/MagicDrive/third_party)。MagicDrive-V2 是另一套模型，其实现位于 [magicdrivedit](generation/image/MagicDrive-V2/magicdrivedit)。原生安装器会为各模型重建推理脚本期望的上游路径。

## HPC 权重

统一根目录为 `weights/`。其中 `catk/` 保存交通模型及词表，`renderers/` 和 `magicdrivev2/` 保存七种渲染器的权重及原生配置，`agents/` 按主车模型名称保存检查点、锚点和模板，`evaluation/` 保存当前协议的 LoFTR、DINO 与 CLIP-IQA+ 权重。文件清单及字节数在 `weights/inventory.json`。桌面与 HPC 同步 Git 源码；GPU 权重集中保存在 HPC，由原生环境按需加载。

同一文件系统的预训练权重通过硬链接归集。更新权重时请创建新版本文件并替换目录项，以免原地写入影响共享同一 inode 的其他安装。已有 HPC 配置直接使用统一目录。在新机器上可用 `python scripts/collect_weights.py --help` 查看导入参数。生成运行环境接受 `python scripts/geninstall_native.py --destination NEW_RUNTIME --weight-store WEIGHTS --extensions-root ABI_ROOT`，由仓库源码、外部 CUDA 扩展和统一权重组成；修改原生生成源码后可据此重建安装。

原生 Python 环境、nuPlan devkit、编译算子和 DINO 源码属于运行依赖。各策略的推理与 GPU 环境见 [主车接入](../docs/agents.md)，七个渲染器的条件与启动方式见 [生成说明](../docs/generation.md)。

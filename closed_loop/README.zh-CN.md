[![英文版](https://img.shields.io/badge/英文版-777777?style=flat-square)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](README.zh-CN.md)

# DriveAudit 闭环测试

每轮由当前自车位姿和 CAT-K 交通状态生成六视角图像。DrivoR 使用其原生相机子集预测轨迹，执行器将前 0.5 秒分成五个 0.1 秒子步。CAT-K 将执行结果写回交通历史，作为下一轮渲染的条件。

目标点取自选手公开自车条件的最后一帧，导航适配器将相对方位转换为 DrivoR 的左转、右转或直行命令。进入目标半径后，使用吸收终态。

<a id="run"></a>
## 运行一个场景

使用与生成配置相同的 CAT-K、策略和渲染器权重。将 `weights` 链接到统一权重目录，`work/resources` 指向已配置的资源目录。按[生成指南](../docs/generation.zh-CN.md)将原生运行环境安装到 `work/native`。

在仓库根目录，使用装有 NumPy 的 Python 运行：

- `PUBLIC_SCENE`：解压后的公开选手场景。
- `RENDER_SCENE`：生成流程 `stage` 阶段输出的对应场景。
- 如需其他渲染器，将 `basedreamer` 替换为相应 ID。每轮都会重新生成交通状态和 RGB。

```bash
python closed_loop/configure.py --renderer basedreamer \
  --participant-scene "$PUBLIC_SCENE" --render-scene "$RENDER_SCENE" \
  --output work/closed-loop/config.json

# 在 Slurm GPU 分配内执行。
python closed_loop/run_goal.py --config work/closed-loop/config.json \
  --output work/closed-loop/run-01 --steps 2
python closed_loop/verify_run.py work/closed-loop/run-01
```

<a id="resources"></a>
## 资源与验证

| 渲染器 | 默认 GPU 数量 |
| --- | ---: |
| BaseDreamer、DreamForge、HorizonDrive、MagicDrive、Panacea | 2 |
| OpenDWM、MagicDrive-V2 | 5，其中四张用于渲染进程 |

每次使用新的输出目录。验证器检查执行轨迹、下一轮位姿、生成图像指纹和终态行为。逐轮记录与耗时写入 Git 忽略的 `work/`。核验通过表示反馈链路一致；[策略配对评分](../docs/evaluation.zh-CN.md)是独立的评测过程。

MagicDrive-V2 的四个渲染进程分别在主机内存中加载模型并向 GPU 卸载权重。单场景五卡 Slurm 作业可用 `--gres=gpu:l40:5 --cpus-per-task=20 --mem=256G` 申请资源。

[← 项目概览](../README.zh-CN.md)

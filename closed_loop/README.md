# DriveAudit 闭环测试

每轮由当前自车位姿和 CAT-K 交通状态生成六视角图像。DrivoR 使用其原生相机子集预测轨迹，执行器将前 0.5 秒分成五个 0.1 秒子步。CAT-K 将执行结果写回交通历史，渲染器在下一轮读取更新后的状态。目标点取自选手公开自车条件的最后一帧，导航适配器将相对方位转换为 DrivoR 的左转、右转或直行命令。进入目标半径后使用吸收终态。

## 运行一个场景

使用与生成配置相同的 CAT-K、主车和视频权重。将此工作目录的 `weights` 链接到现有统一权重目录，将 `work/resources` 指向已配置的资源目录。本分支的原生运行环境安装在 `work/native`，安装方式见 [生成说明](../docs/generation.md)。

在仓库根目录使用装有 NumPy 的 Python。指定同一公开场景的两种输入，`PUBLIC_SCENE` 是解压后的选手场景目录，`RENDER_SCENE` 是生成流程 stage 阶段输出的对应场景目录。闭环每轮重新计算交通状态和 RGB。若选用其他渲染器，将 `basedreamer` 替换为相应 ID。

```bash
python closed_loop/configure.py --renderer basedreamer \
  --participant-scene "$PUBLIC_SCENE" --render-scene "$RENDER_SCENE" \
  --output work/closed-loop/config.json
# 在 Slurm GPU 分配内执行
python closed_loop/run_goal.py --config work/closed-loop/config.json \
  --output work/closed-loop/run-01 --steps 2
python closed_loop/verify_run.py work/closed-loop/run-01
```

BaseDreamer、DreamForge、HorizonDrive、MagicDrive 和 Panacea 的默认布局使用两张 GPU。OpenDWM 与 MagicDrive-V2 使用五张，其中四张用于分布式渲染。每次使用新的输出目录。验证器检查执行轨迹、下一轮位姿、生成图像指纹和到达终点后的状态。逐轮记录与耗时写入 Git 忽略的 `work/`。闭环核验通过表示反馈链路成立；论文的配对评分另有独立入口。

MagicDrive-V2 的四个渲染进程分别在主机内存中加载模型并向 GPU 卸载权重。单场景五卡 Slurm 作业可用 `--gres=gpu:l40:5 --cpus-per-task=20 --mem=256G` 申请资源。

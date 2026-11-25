# DriveAudit 赛题：生成未来交通轨迹和六路相机图像

[![英文版](https://img.shields.io/badge/英文版-777777?style=flat-square)](challenge.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](challenge.zh-CN.md)

对选手包中的 250 个 nuScenes 场景，先预测交通参与者的未来运动，再根据给定的自车和相机位姿，渲染车载六路相机在各时刻看到的画面。图像和轨迹必须描述同一时刻的同一场景。请仅使用给定的历史观测和生成条件，勿查询未公开的未来交通轨迹或图像。

## 输入

选手包的默认路径为 `work/resources/participant.tar.gz`。解压后得到 `participant/` 目录，其中的 `dataset.json` 列出了正式场景编号。每个场景包含 10 帧历史观测（第 0–9 帧）和 80 个目标帧（第 10–89 帧）。

| `participant/scenes/<scene_id>/` 下的路径 | 内容 |
| --- | --- |
| `history/rgb/<camera>/`、`history/agents.csv`、`history/ego.csv` | 六路相机的历史 RGB 图像、匿名交通参与者轨迹和自车历史轨迹 |
| `calibration/`、`map/` | 相机标定、历史位姿和静态地图 |
| `conditioning/ego_trajectory.csv`、`conditioning/ego_poses.json`、`conditioning/camera_poses.json` | 给定的自车运动与相机位姿 |
| `timeline.json`、`task.json` | 采样时间、目标帧、相机名称和图像尺寸 |

从 `timeline.json` 读取时间戳。所有轨迹均使用历史第 9 帧的自车坐标系，x 轴向前、y 轴向左、z 轴向上。

## 提交内容

每个场景提交一份轨迹 CSV 和 480 张生成的 RGB JPEG 图像。以下是第一个场景的目录结构。请为 `dataset.json` 中的每个场景按相同结构生成文件，并使用常规文件。

```text
submission/
  scenes/
    case_000001/
      future/
        future_trajectories.csv
        rgb/
          CAM_FRONT_LEFT/    000010.jpg … 000089.jpg
          CAM_FRONT/         000010.jpg … 000089.jpg
          CAM_FRONT_RIGHT/   000010.jpg … 000089.jpg
          CAM_BACK_LEFT/     000010.jpg … 000089.jpg
          CAM_BACK/          000010.jpg … 000089.jpg
          CAM_BACK_RIGHT/    000010.jpg … 000089.jpg
```

**交通轨迹。** `future_trajectories.csv` 必须使用以下表头及列顺序。

```csv
frame_index,agent_id,category,is_dynamic,x,y,z,width,length,height,yaw,vx,vy,valid
```

- 沿用历史中每个交通参与者的匿名编号和类别。每个参与者在每个目标帧均需有一行记录，缺席时填写 `valid=False`。对新生成的参与者，使用场景内唯一的编号（如 `new_0001`），从首次出现起到第 89 帧逐帧填写。
- 位置和尺寸的单位为米，航向角为弧度，速度为米每秒。所有数值均须有限，包括标记为无效的行。长、宽、高必须为正数。`is_dynamic` 和 `valid` 使用布尔值。
- 新参与者可使用的类别为 `car`、`vehicle`、`truck`、`bus`、`trailer`、`construction_vehicle`、`motorcycle`、`bicycle`、`pedestrian`、`animal`、`traffic_cone`、`barrier` 和 `other`。同一参与者在各帧的类别必须一致。

**相机图像。** 每个目标帧的每路相机各写入一张 400 × 224 的 RGB JPEG 图像。文件名使用轨迹 CSV 中对应的帧号，左侧补零至六位，如 `000010.jpg`。

## 校验提交

安装 Python 3、NumPy 和 Pillow 后，在包含已解压的 `participant/` 与待提交的 `submission/` 的目录中运行以下命令。

```bash
# 生成交通轨迹后，先校验一个场景。
python participant/validate_submission.py --inputs participant --submission submission \
  --scene case_000001 --trajectories-only

# 如需使用 DriveAudit 渲染器，可为同一场景准备渲染输入。
python participant/prepare_render_inputs.py --inputs participant --trajectories submission \
  --scene case_000001 --output render_inputs

# 所有场景渲染完成后，校验整份提交。
python participant/validate_submission.py --inputs participant --submission submission
```

完整校验通过后，终端会打印 `FORMAT_PASS 250 scenes`，并提示该结果仅表示文件格式通过校验。提交上文所列的 `submission/` 目录内容即可。选手输入包和中间渲染输入保存在本地。

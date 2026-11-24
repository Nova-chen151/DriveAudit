# 选手输入和提交

选手包由生成配置的 `archive` 指定。包内有 250 个 nuScenes 场景、赛题说明、提交格式和公开校验器。示例选用前两个公开场景编号。

每个场景提供帧 0–9 的六视角 RGB、历史交通参与者、地图、相机标定及时间戳。帧 10–89 的自车和相机位姿作为给定条件。交通轨迹采用第 9 帧自车坐标系，x 向前、y 向左、z 向上，长度以米计。采样时间读取 `timeline.json`。CAT-K 预测其他交通参与者，再由所选渲染器生成对应的未来图像。

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

# 选手输入包

把下载的 nuScenes 选手包复制或软链接为本目录下的 `participant.tar.gz`，保留压缩格式。也可以在 [generation.json](../../configs/generation.json) 中把 `archive` 指向现有文件。

`envs/` 用于连接各模型的 Python 环境。主办方进行配对评分时，还需要在此提供 `scene_mapping.json` 并配置真实参考数据。选手生成轨迹和视频只需公开输入。

Place or symlink the participant archive here as `participant.tar.gz`. The runner extracts the selected scenes. See the [generation guide](../../docs/generation.md).

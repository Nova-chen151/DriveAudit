# 模型权重

将模型文件按以下子目录放置，保留各模型要求的文件名和层级。

| 目录 | 内容 |
| --- | --- |
| `catk/` | CAT-K 检查点和轨迹词表 |
| `renderers/`、`magicdrivev2/` | 七个图像生成模型的权重 |
| `agents/` | 六个主车策略的权重 |
| `evaluation/` | 视觉评价模型的权重及缓存 |

已有权重库可按[生成指南](../docs/generation.zh-CN.md#model-weights)接入。Git 会忽略本目录中的权重文件。

Model checkpoints go here. See the [English weights guide](../docs/generation.md#model-weights) for setup.

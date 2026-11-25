# 模型权重

[![英文版](https://img.shields.io/badge/英文版-777777?style=flat-square)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](README.zh-CN.md)

将权重放入下列目录，保留各模型要求的文件名和目录层级。

| 目录 | 内容 |
| --- | --- |
| `catk/` | CAT-K 检查点和轨迹词表 |
| `renderers/` | BaseDreamer、DreamForge、HorizonDrive、MagicDrive、OpenDWM 和 Panacea 的检查点 |
| `magicdrivev2/` | MagicDrive-V2 检查点及预训练组件 |
| `agents/` | 六个驾驶策略的检查点 |
| `evaluation/` | 视觉质量指标所用的权重及缓存 |

Git 会忽略此目录中的权重文件。已有权重库的接入方法见[模型权重指南](../docs/generation.zh-CN.md#model-weights)。

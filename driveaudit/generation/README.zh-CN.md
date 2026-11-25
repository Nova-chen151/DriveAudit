[![English](https://img.shields.io/badge/English-777777?style=flat-square)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-2878B5?style=flat-square)](README.zh-CN.md)

# 生成模块源码来源

- **来源：**选手交通适配器、约束采样器、渲染器适配器及共享 GenBench 加载器来自 2026-11-23 的 Participant Visual Demos 快照。原生模型主体位于 `models/generation`；检查点与兼容的编译扩展需另行配置。
- **集成：**适配器传入公开选手场景和显式模型路径。CAT-K 从指定位置加载官方源码、检查点及词表，并核对源码与校验和。此快照包含推理集成代码与配置。
- **许可证：**原生项目及捆绑库保留原有声明。CAT-K 保留 NVIDIA 研究/评测许可证及 SMART 声明；MagicDrive-V2 及其 `rd` 代码保留各自许可证。完整源码与许可证清单见 [native-sources.tar.json](bundles/native-sources.tar.json)。

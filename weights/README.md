# Model weights

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](README.md) [![Simplified Chinese](https://img.shields.io/badge/Simplified_Chinese-777777?style=flat-square)](README.zh-CN.md)

Place checkpoints in the directories below. Keep the filenames and directory structure expected by each model.

| Directory | Contents |
| --- | --- |
| `catk/` | CAT-K checkpoint and token vocabularies |
| `renderers/` | Checkpoints for BaseDreamer, DreamForge, HorizonDrive, MagicDrive, OpenDWM, and Panacea |
| `magicdrivev2/` | MagicDrive-V2 checkpoints and pretrained components |
| `agents/` | Checkpoints for the six driving policies |
| `evaluation/` | Weights and caches for the visual quality metrics |

Git ignores weight files in this directory. To use an existing weight store, see the [model weights guide](../docs/generation.md#model-weights).

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-777777?style=flat-square)](README.zh-CN.md)

# Generation source provenance

- **Source:** participant traffic adapter, constrained sampler, renderer adapters, and shared GenBench loader from the Participant Visual Demos snapshot dated 2026-11-23. Native model backbones are tracked under `models/generation`; checkpoints and compatible compiled extensions are provisioned separately.
- **Integration:** adapters supply public participant scenes and explicit model paths. CAT-K loads the configured official source, checkpoint, and vocabularies with source/checksum verification. The snapshot contains inference integration code and configuration.
- **Licenses:** native projects and bundled libraries retain their original notices. CAT-K retains NVIDIA's research/evaluation license and the SMART notices; MagicDrive-V2 and its `rd` code retain their respective licenses. The complete source and license inventory is in [native-sources.tar.json](bundles/native-sources.tar.json).

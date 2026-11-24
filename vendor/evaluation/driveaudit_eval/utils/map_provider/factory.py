from __future__ import annotations



def build_map_provider(provider_type, **kwargs):
    if provider_type == "nuscenes":
        from driveaudit_eval.utils.map_provider.nuscenes_provider import NuScenesMapProvider
        return NuScenesMapProvider(**kwargs)
    if provider_type == "opendrive":
        from driveaudit_eval.utils.map_provider.opendrive_provider import OpenDriveMapProvider
        path = kwargs.pop("path")
        return OpenDriveMapProvider.from_file(path, kwargs.pop("sample_step_m", 0.5))
    raise ValueError("unsupported map provider: %s" % provider_type)

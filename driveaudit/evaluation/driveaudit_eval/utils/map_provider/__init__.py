"""Map providers are imported only when requested."""
from driveaudit_eval.utils.map_provider.base import MapProvider


def build_map_provider(provider_type, **kwargs):
    from driveaudit_eval.utils.map_provider.factory import build_map_provider as build
    return build(provider_type, **kwargs)


def __getattr__(name):
    if name == 'NuScenesMapProvider':
        from driveaudit_eval.utils.map_provider.nuscenes_provider import NuScenesMapProvider
        return NuScenesMapProvider
    if name == 'OpenDriveMapProvider':
        from driveaudit_eval.utils.map_provider.opendrive_provider import OpenDriveMapProvider
        return OpenDriveMapProvider
    raise AttributeError(name)

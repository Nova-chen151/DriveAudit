"""Compatibility imports for the native camera protocol."""
from driveaudit.agents.camera_only import INPUT_KEYS, META_KEYS, inference_inputs, seed_temporal_request, stable_request_seed


def vad_camera_only_forward_test(*args, **kwargs):
    from driveaudit.agents.registry import load_adapter
    return load_adapter("vad").vad_camera_only_forward_test(*args, **kwargs)

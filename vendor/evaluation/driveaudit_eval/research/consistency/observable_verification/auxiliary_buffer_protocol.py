from __future__ import annotations

AUXILIARY_SCHEMA = {
    "instance_id_mask": {"formats": ["png16", "png32"], "background_id": 0, "alignment": "camera_frame_timestamp"},
    "depth": {"formats": ["exr_float32", "npy_float32", "npz_float32"], "unit": "metres", "semantics": ["camera_z", "ray_depth"]},
    "motion_vector": {"formats": ["npy_float32", "npz_float32"], "channels": 2, "unit": "pixel_per_frame",
                      "direction": ["forward", "backward"], "includes_ego_and_camera_motion": True},
}


def status(config: dict) -> dict:
    root = config["auxiliary_protocol"].get("root")
    return {"status": "not_run_optional_auxiliary_protocol" if not root else "pending_validation",
            "schema": AUXILIARY_SCHEMA, "blocks_tcvv_o": False, "submitted_buffers_are_ground_truth": False}

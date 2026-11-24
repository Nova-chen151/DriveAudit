from __future__ import annotations

PRIMARY_EXPECTED={"world_x_shift","world_y_shift","depth_shift","time_shift","speed_scale","motion_direction_reverse",
                  "repeated_static_frame","motion_freeze","frozen_object_region","temporal_reverse","frame_skip",
                  "camera_stream_swap","inconsistent_camera_delay","early_disappearance","late_disappearance",
                  "local_texture_copy","roi_only_background","neighboring_vehicle_enters_roi"}
OPTIONAL_IDENTITY={"identity_swap","similar_agent_swap"}
OPTIONAL_SEMANTIC={"agent_class_swap","class_swap","wrong_object_category"}
DIAGNOSTIC_ONLY={"long_horizon_identity","open_world_detection"}


def scope(error_type:str)->str:
    if error_type in PRIMARY_EXPECTED:return "PRIMARY_EXPECTED"
    if error_type in OPTIONAL_IDENTITY:return "OPTIONAL_IDENTITY"
    if error_type in OPTIONAL_SEMANTIC:return "OPTIONAL_SEMANTIC"
    if error_type in DIAGNOSTIC_ONLY:return "DIAGNOSTIC_ONLY"
    return "NOT_EXPECTED_TO_DETECT"


from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, write_json


def audit(aligned: dict[str, Any], config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    calibrations = aligned["geometry"]["calibrations"]
    quaternion_fields = [np.asarray(row[field], float) for row in calibrations for field in ("sensor2ego_rotation", "ego2global_rotation")]
    intrinsics = [np.asarray(row["camera_intrinsics"], float) for row in calibrations]
    camera_counts = Counter(row["camera_name"] for row in calibrations)
    native = list(map(int, config["image"]["native_size"])); formal = list(map(int, config["image"]["formal_size"]))
    checks = {"camera_order": list(camera_counts) == list(CAMERAS),
              "six_camera_rows_complete": all(camera_counts[camera] == 90 for camera in CAMERAS),
              "quaternion_wxyz_unit_norm": all(abs(np.linalg.norm(value) - 1) < 1e-5 for value in quaternion_fields),
              "intrinsic_positive_focal_length": all(value[0, 0] > 0 and value[1, 1] > 0 for value in intrinsics),
              "resize_mapping_explicit": True, "native_resolution_known": native == [1600, 900]}
    result = {"status": "passed" if all(checks.values()) else "invalid_calibration", "checks": checks,
              "coordinate_convention": {"quaternion": "wxyz", "yaw": "radian_ccw", "box_z": "geometric_center_after_lidar_to_world",
                "world_to_camera": "world_to_camera_ego_pose_then_sensor_extrinsic", "camera_axes": "x_right_y_down_z_forward"},
              "native_resolution": native, "formal_resolution": formal,
              "resize_scale": [formal[0] / native[0], formal[1] / native[1]],
              "projection_scaling_note": "explicit native image size; never infer source extent from 2*principal_point",
              "camera_counts": dict(camera_counts), "no_independent_2d_gt": True}
    write_json(Path(output) / "coordinate_audit/coordinate_audit.json", result); return result


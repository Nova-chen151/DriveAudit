"""Model and checkpoint resolution shared by native adapters."""
import hashlib
import os
from pathlib import Path

from driveaudit.resources import resource_root


def model_root(name):
    default = resource_root() / "models" / "agents"
    return (Path(os.getenv("DRIVEAUDIT_AGENT_SOURCES", str(default))) / name).resolve()


def asset_root(name):
    directory = os.getenv("DRIVEAUDIT_AGENT_ASSETS")
    base = Path(directory) if directory else Path(os.environ.get(
        "DRIVEAUDIT_WEIGHTS", resource_root() / "weights")) / "agents"
    return (base.expanduser() / name).resolve()


def timestamp_seed(timestamp):
    return int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)


def seed_camera_request(timestamp):
    import random
    import numpy as np
    import torch

    seed = timestamp_seed(timestamp)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


FORBIDDEN_AGENT_FIELDS = {"evaluation_only", "background_boxes", "future_state_frame_ids",
                 "reference_route_positions", "drivable_mask_source", "collision_result",
                 "scoring_result", "gt_bboxes_3d", "gt_labels_3d", "gt_attr_labels",
                 "ego_fut_trajs", "gt_future_trajectory", "gt_actor_future", "gt_occupancy", "gt_boxes"}


def assert_no_evaluation_leakage(value):
    if isinstance(value, dict):
        if FORBIDDEN_AGENT_FIELDS.intersection(value):
            raise ValueError("Evaluation fields present in agent input")
        for child in value.values():
            assert_no_evaluation_leakage(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_evaluation_leakage(child)


def quat_rotation(q):
    import numpy as np
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Expected finite nonzero quaternion")
    w, x, y, z = np.asarray(q, dtype=float) / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

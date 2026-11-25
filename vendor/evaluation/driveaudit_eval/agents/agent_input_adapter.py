"""Evaluator access to the public camera request and coordinate protocol."""
import numpy as np

from driveaudit.agents.runtime import FORBIDDEN_AGENT_FIELDS, assert_no_evaluation_leakage, quat_rotation
from driveaudit.agents.temporal import (
    UNIAD_CAMERA_ORDER, WORLD_DREAMER_CAMERA_ORDER, build_http_request, compose_worlddreamer_png,
    ego_pose_matrix, model_geometry, parse_output, planning_to_unified_ego,
)


def select_vad_plan(per_step_offsets, command_one_hot, mode_scores=None):
    """Apply VAD's official command selection and exactly one cumsum."""
    offsets = np.asarray(per_step_offsets, dtype=float)
    command = np.asarray(command_one_hot, dtype=float).reshape(-1)
    active = np.flatnonzero(command)
    if len(active) != 1:
        raise ValueError("VAD command must be one-hot")
    command_index = int(active[0])
    branch = offsets[command_index]
    mode_index, mode_score = command_index, None
    if branch.ndim == 3:
        scores = np.asarray(mode_scores, dtype=float)[command_index]
        mode_index = int(np.argmax(scores))
        mode_score = float(scores[mode_index])
        branch = branch[mode_index]
    if branch.ndim != 2 or branch.shape[1] != 2:
        raise ValueError(f"unexpected VAD planning shape: {offsets.shape}")
    branch = branch.copy()
    branch[np.abs(branch) < 0.01] = 0.0
    return np.cumsum(branch, axis=0), command_index, mode_index, mode_score

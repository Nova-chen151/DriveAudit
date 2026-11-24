"""Agent-visible input adapters.  This module intentionally does not import evaluation_only."""
from __future__ import annotations
from typing import Any, Dict
from driveaudit_eval.agents.core import UNIAD_CAMERA_ORDER, WORLD_DREAMER_CAMERA_ORDER, compose_worlddreamer_png, quat_rotation
import numpy as np

FORBIDDEN_AGENT_FIELDS = {
    "background_boxes", "future_state_frame_ids", "reference_route_positions",
    "drivable_mask_source", "collision_result", "scoring_result",
    "gt_bboxes_3d", "gt_labels_3d", "gt_attr_labels", "ego_fut_trajs",
    "gt_future_trajectory", "gt_actor_future", "gt_occupancy", "gt_boxes",
}


def assert_no_evaluation_leakage(value: Any) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_AGENT_FIELDS & value.keys()
        if leaked:
            raise ValueError(f"evaluation-only fields in agent input: {sorted(leaked)}")
        for child in value.values():
            assert_no_evaluation_leakage(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_evaluation_leakage(child)


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


def planning_to_unified_ego(points):
    """nuScenes planning [right, forward] -> scorer ego [forward, left]."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"unexpected planning coordinates: {points.shape}")
    return np.column_stack((points[:, 1], -points[:, 0]))


def model_geometry(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """Build the nuScenes lidar/camera transforms consumed by UniAD and VAD."""
    ego = model_input["ego_state"]
    lidar_rotation = quat_rotation(ego["lidar2ego_rotation"])
    lidar_translation = np.asarray(ego["lidar2ego_translation"], dtype=float)
    lidar2img = []
    for camera in UNIAD_CAMERA_ORDER:
        item = model_input["camera_calibration"][camera]
        camera_rotation = quat_rotation(item["sensor2ego_rotation"])
        camera_translation = np.asarray(item["sensor2ego_translation"], dtype=float)
        camera_to_lidar_rotation = lidar_rotation.T @ camera_rotation
        camera_to_lidar_translation = lidar_rotation.T @ (camera_translation - lidar_translation)
        lidar_to_camera_rotation = camera_to_lidar_rotation.T
        lidar_to_camera = np.eye(4)
        lidar_to_camera[:3, :3] = lidar_to_camera_rotation
        lidar_to_camera[:3, 3] = -lidar_to_camera_rotation @ camera_to_lidar_translation
        intrinsic = np.asarray(item["camera_intrinsics"], dtype=float).copy()
        intrinsic[1] *= 896.0 / 900.0
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsic
        lidar2img.append((viewpad @ lidar_to_camera).tolist())
    ego_rotation = quat_rotation(ego["rotation"])
    return {
        "lidar2img": lidar2img,
        "l2g_r_mat": (lidar_rotation.T @ ego_rotation.T).tolist(),
        "l2g_t": (lidar_translation @ ego_rotation.T + np.asarray(ego["position"], dtype=float)).tolist(),
    }

def build_http_request(model_input: Dict[str, Any], scene_id: str = "", frame_id: int = -1) -> Dict[str, Any]:
    """Payload accepted by the bundled UniAD/VAD FastAPI demos.

    The six images are vertically composed in WorldDreamer's real output order;
    each demo owns its internal reorder to UniAD/VAD order.
    """
    assert_no_evaluation_leakage(model_input)
    ego=model_input["ego_state"]
    payload={"img_byte_array":compose_worlddreamer_png(model_input["camera_images"]),
            "timestamp":float(ego["timestamp"])/1e6, "ego_pose":ego_pose_matrix(ego),
            "command":int(model_input["command"]["code"]), "accel":ego["acceleration"],
            "rotation_rate":[0.0,0.0,ego["yaw_rate"]], "vel":ego["velocity"],
            "camera_order":WORLD_DREAMER_CAMERA_ORDER, "scene_id":scene_id,
            "frame_id":frame_id, "model_geometry":model_geometry(model_input)}
    assert_no_evaluation_leakage(payload)
    return payload

def ego_pose_matrix(state: Dict[str, Any]):
    from driveaudit_eval.agents.core import quat_rotation
    import numpy as np
    m=np.eye(4); m[:3,:3]=quat_rotation(state["rotation"]); m[:3,3]=np.asarray(state["position"],float)
    return m.tolist()


def parse_output(agent, body):
    import json
    obj=json.loads(body)
    if agent=="uniad":
        box=obj["bbox_results"][0]; traj=box["planning_traj"]
        if traj and isinstance(traj[0][0],list): traj=traj[0]
        return traj, None, None, obj
    if agent in ("drivor", "diffusiondrivev2", "sparsedrivev2", "goalflow"):
        return obj["planning_traj"], obj.get("mode_index"), obj.get("mode_score"), obj
    # VAD driver-get returns a one-element list.
    sample=obj[0] if isinstance(obj,list) else obj
    return sample["planning_traj"], sample.get("mode_index"), sample.get("mode_score"), sample

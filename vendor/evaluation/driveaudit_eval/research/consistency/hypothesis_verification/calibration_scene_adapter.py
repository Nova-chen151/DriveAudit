from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.shared.common import json_cell, read_csv
from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS

DYNAMIC_CLASSES = {"car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "bicycle", "motorcycle"}


@dataclass
class TCVVCalibrationScene:
    physical_scene_id:str;source_dataset:str;timestamps:list[float];camera_names:list[str];image_paths:dict[str,list[str]]
    intrinsics:dict[str,Any];extrinsics:dict[str,Any];ego_poses:list[dict[str,Any]];agents:dict[str,Any];provenance:dict[str,Any]

    def to_dict(self):return asdict(self)


def adapt_driveaudit_eval_pair(pair:dict[str,Any])->TCVVCalibrationScene:
    conditions=Path(pair["image_sequence"]["conditions_root"]);frames=read_csv(conditions/"frames.csv")
    calibration=read_csv(conditions/"camera_calibration.csv");poses=read_csv(conditions/"ego_pose.csv");boxes=read_csv(conditions/"boxes.csv")
    history_frames = int(pair["timeline"]["image_history_frames"])
    paths={camera:[str(Path(pair["image_sequence"]["history_root"])/camera/f"{frame:06d}.jpg") for frame in range(10)] for camera in CAMERAS}
    digest=hashlib.sha256()
    for path in [conditions/"frames.csv",conditions/"camera_calibration.csv",conditions/"ego_pose.csv",conditions/"boxes.csv"]:
        digest.update(path.read_bytes())
    for camera in CAMERAS:
        for path in paths[camera][:2]:digest.update(Path(path).read_bytes())
    agents={}
    for row in boxes:
        if int(row["frame_index"]) >= history_frames: continue
        category=str(row.get("class_name") or row.get("category") or row.get("class") or "unknown")
        if category not in DYNAMIC_CLASSES: continue
        agent=str(row.get("agent_id") or row.get("track_id") or row.get("instance_token") or row.get("obj_id"))
        agents.setdefault(agent,{"class":category,"boxes_3d":[],"valid_mask":[]})
        agents[agent]["boxes_3d"].append(row);agents[agent]["valid_mask"].append(str(row.get("valid_flag", "true")).lower() == "true")
    intrinsics={row["camera"]:json_cell(row["camera_intrinsics"]) for row in calibration[:6]}
    extrinsics={row["camera"]:{"translation":json_cell(row["sensor2ego_translation"]),"rotation":json_cell(row["sensor2ego_rotation"])} for row in calibration[:6]}
    parsed_poses=[{**row,"ego2global_translation":json_cell(row["ego2global_translation"]),"ego2global_rotation":json_cell(row["ego2global_rotation"])} for row in poses[:history_frames]]
    return TCVVCalibrationScene(digest.hexdigest(),"DriveAudit_challenge",[float(row["timestamp"])/1e6 for row in frames[:history_frames]],list(CAMERAS),paths,
                                intrinsics,extrinsics,parsed_poses,agents,{"pair_id":pair["pair_id"],"conditions_root":str(conditions.resolve()),"content_hash":digest.hexdigest()})

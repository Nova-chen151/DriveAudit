from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.agents.core import quat_rotation
from driveaudit_eval.consistency.shared.visibility import _box_corners_3d
from driveaudit_eval.research.consistency.observed_flow.schema import write_csv, write_json

FACES = {"bottom": (0, 1, 2, 3), "top": (4, 7, 6, 5), "side_01": (0, 4, 5, 1),
         "side_12": (1, 5, 6, 2), "side_23": (2, 6, 7, 3), "side_30": (3, 7, 4, 0)}


def camera_corners(box: dict[str, Any], calibration: dict[str, Any]) -> np.ndarray:
    corners = _box_corners_3d(box); world_rotation = quat_rotation(calibration["ego2global_rotation"])
    ego = (corners - np.asarray(calibration["ego2global_translation"], float)) @ world_rotation
    sensor_rotation = quat_rotation(calibration["sensor2ego_rotation"])
    return (ego - np.asarray(calibration["sensor2ego_translation"], float)) @ sensor_rotation


def visible_faces(box: dict[str, Any], calibration: dict[str, Any], image_size=(1600, 900), native_size=(1600, 900), minimum_depth=.2) -> dict[str, Any]:
    corners = camera_corners(box, calibration); center = corners.mean(axis=0); intrinsic = np.asarray(calibration["camera_intrinsics"], float)
    uvw = corners @ intrinsic.T; uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9); uv *= np.asarray([image_size[0]/native_size[0],image_size[1]/native_size[1]])
    rows=[]
    for face_id, indices in FACES.items():
        points=corners[list(indices)]; normal=np.cross(points[1]-points[0],points[2]-points[0]); face_center=points.mean(axis=0)
        if np.dot(normal,face_center-center)<0: normal=-normal
        facing=float(np.dot(normal/(np.linalg.norm(normal)+1e-9),-face_center/(np.linalg.norm(face_center)+1e-9)))
        polygon=uv[list(indices)]; area=float(abs(cv2.contourArea(polygon.astype(np.float32))))
        visible=bool(np.all(points[:,2]>minimum_depth) and facing>0 and area>1)
        rows.append({"face_id":face_id,"corner_indices":list(indices),"polygon":polygon.astype(float).tolist(),"area":area,"facing_cosine":facing,"visible":visible})
    selected=[row for row in rows if row["visible"]]
    condition=float(max(np.linalg.cond(np.column_stack([np.asarray(row["polygon"],float),np.ones(4)])) for row in selected)) if selected else None
    return {"visible_face_ids":[row["face_id"] for row in selected],"visible_surfaces":selected,"all_surfaces":rows,
            "surface_area":sum(row["area"] for row in selected),"surface_condition_number":condition}


def analyze(aligned_rows: list[dict[str, Any]], projections: list[dict[str, Any]], config: dict[str, Any], output: str | Path) -> list[dict[str, Any]]:
    pmap={(row["agent_id"],row["camera_name"],int(row["frame_index"])):row for row in projections}; rows=[]
    for aligned in aligned_rows:
        key=(aligned["agent_id"],aligned["camera_name"],int(aligned["frame_index"])); projection=pmap.get(key)
        if not projection or not projection.get("projected_valid"):continue
        value=visible_faces(aligned["box_reference"],aligned["calibration"],tuple(config["image"]["formal_size"]),tuple(config["image"]["native_size"]),float(config["projection"]["minimum_depth_m"]))
        rows.append({"agent_id":aligned["agent_id"],"agent_type":aligned["agent_type"],"category":aligned["category"],"camera_name":aligned["camera_name"],
                     "frame_index":int(aligned["frame_index"]),"timestamp":aligned["timestamp"],"occluded_ratio":projection.get("occlusion_ratio"),
                     "truncated_ratio":projection.get("truncated_ratio"),**value})
    grouped=defaultdict(dict)
    for row in rows:grouped[(row["agent_id"],row["camera_name"])][row["frame_index"]]=row
    for frames in grouped.values():
        for frame,row in frames.items():
            next_row=frames.get(frame+1);row["face_switch"]=bool(next_row and set(row["visible_face_ids"])!=set(next_row["visible_face_ids"]))
    output=Path(output);write_json(output/"visible_surfaces/visible_surfaces.json",rows);write_csv(output/"visible_surfaces/per_frame.csv",_flat(rows));return rows


def _flat(rows):return [{key:value for key,value in row.items() if key not in {"visible_surfaces","all_surfaces"}} for row in rows]

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import write_csv, write_json


def build(expectations: list[dict[str, Any]], surfaces: list[dict[str, Any]], config: dict[str, Any], output: str | Path) -> list[dict[str, Any]]:
    smap={(row["agent_id"],row["camera_name"],int(row["frame_index"])):row for row in surfaces};rows=[]
    for expectation in expectations:
        source=smap.get((expectation["agent_id"],expectation["camera_name"],int(expectation["frame_index"])));target=smap.get((expectation["agent_id"],expectation["camera_name"],int(expectation["target_frame_index"])))
        if source is None or target is None:continue
        model=make_model(expectation,source,target,config["geometry"])
        rows.append({"agent_id":expectation["agent_id"],"agent_type":expectation["agent_type"],"category":expectation["category"],"camera_name":expectation["camera_name"],
                     "frame_index":int(expectation["frame_index"]),"target_frame_index":int(expectation["target_frame_index"]),"timestamp":expectation["timestamp"],"target_timestamp":expectation["target_timestamp"],
                     "delta_frames":expectation["delta_frames"],"expected_visible":expectation["expected_visible"],"source_projection":expectation["source_projection"],"target_projection":expectation["target_projection"],
                     "expected_center_flow":expectation["expected_center_flow"],"expected_scale_change":expectation["expected_scale_change"],"expected_rotation_change":expectation["expected_rotation_change"],
                     "face_switch":source.get("face_switch",False),**model})
    output=Path(output);write_json(output/"geometric_warp/warp_models.json",rows);write_csv(output/"geometric_warp/per_window.csv",_flat(rows));return rows


def make_model(expectation:dict[str,Any],source:dict[str,Any],target:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    source_faces={row["face_id"]:row for row in source["visible_surfaces"]};target_faces={row["face_id"]:row for row in target["visible_surfaces"]};common=set(source_faces)&set(target_faces)
    if common and not source.get("face_switch"):
        face=max(common,key=lambda name:source_faces[name]["area"]);a=np.asarray(source_faces[face]["polygon"],np.float32);b=np.asarray(target_faces[face]["polygon"],np.float32)
        matrix=cv2.getPerspectiveTransform(a,b);condition=float(np.linalg.cond(matrix));valid=np.isfinite(matrix).all() and condition<=float(config["maximum_homography_condition_number"])
        if valid:return _result("surface_homography",matrix,condition,True,face)
    a=np.asarray(expectation.get("expected_corner_source",[]),float);b=np.asarray(expectation.get("expected_corner_target",[]),float)
    if len(a)>=3:
        matrix,inliers=cv2.estimateAffine2D(a,b,method=cv2.LMEDS)
        if matrix is not None and np.isfinite(matrix).all():
            homogeneous=np.vstack([matrix,[0,0,1]]);return _result("piecewise_affine",homogeneous,float(np.linalg.cond(homogeneous)),True,None)
    source_center=np.asarray(expectation["source_projection"]["projected_center"],float);target_center=np.asarray(expectation["target_projection"]["projected_center"],float);angle=float(expectation["expected_rotation_change"]);scale=float(expectation["expected_scale_change"])
    affine=cv2.getRotationMatrix2D(tuple(source_center),-np.degrees(angle),scale);affine[:,2]+=target_center-source_center;homogeneous=np.vstack([affine,[0,0,1]])
    return _result("center_scale_rotation",homogeneous,float(np.linalg.cond(homogeneous)),np.isfinite(homogeneous).all(),None)


def _result(model,matrix,condition,valid,face):
    inverse=np.linalg.inv(matrix) if valid else np.eye(3)
    return {"geometry_model":model,"forward_warp":np.asarray(matrix,float).tolist(),"inverse_warp":inverse.astype(float).tolist(),"warp_condition_number":condition,
            "warp_validity":bool(valid),"visible_face_used":face,"expected_flow_is_geometry_only":True}


def transform_points(points:np.ndarray,matrix:list[list[float]]|np.ndarray)->np.ndarray:
    if not len(points):return np.empty((0,2))
    value=cv2.perspectiveTransform(np.asarray(points,np.float32).reshape(-1,1,2),np.asarray(matrix,float)).reshape(-1,2);return value.astype(float)


def expected_flow(points:np.ndarray,model:dict[str,Any])->np.ndarray:return transform_points(points,model["forward_warp"])-points


def warp_patch(image:np.ndarray,model:dict[str,Any],size:tuple[int,int])->np.ndarray:
    return cv2.warpPerspective(image,np.asarray(model["forward_warp"],float),size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def _flat(rows):return [{key:value for key,value in row.items() if key not in {"source_projection","target_projection","forward_warp","inverse_warp"}} for row in rows]


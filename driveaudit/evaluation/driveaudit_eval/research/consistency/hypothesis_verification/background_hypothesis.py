from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.agents.core import quat_rotation


def build(observed: dict[str, Any], source_calibration: dict[str, Any], target_calibration: dict[str, Any],
          config: dict[str, Any], image_size: tuple[int,int]) -> dict[str, Any]:
    points=np.asarray(observed["observed_points"],float); target_depth=max(float(observed.get("projected_depth") or 20),.2)
    candidates=[]
    if config.get("road_plane",True):
        flow=_road_plane_flow(points,source_calibration,target_calibration,image_size,
                              float(observed["source_projection"]["box_reference"].get("z",0))-float(observed["source_projection"]["box_reference"].get("height",0))/2)
        if flow is not None:candidates.append({"depth_source":"road_plane_approximation","expected_flow_per_point":flow.tolist(),"uses_future_true_depth":False})
    depths=[("target_depth_near",target_depth*float(config["target_depth_near_ratio"])),("target_depth",target_depth),
            ("target_depth_far",target_depth*float(config["target_depth_far_ratio"]))]
    depths += [(f"fixed_medium_{int(depth)}m",float(depth)) for depth in config["fixed_medium_depths_m"]]
    depths += [("far_background",float(config["far_background_depth_m"])),("infinite_depth_rotation_only",float(config["infinite_depth_m"]))]
    for name,depth in depths:
        candidates.append({"depth_source":name,"depth_m":depth,
                           "expected_flow_per_point":_fixed_depth_flow(points,source_calibration,target_calibration,depth,image_size).tolist(),
                           "uses_future_true_depth":False})
    return {"background_hypotheses":candidates,"future_gt_depth_used":False,"depth_count":len(candidates)}


def _native_uv(points:np.ndarray,calibration:dict[str,Any],size:tuple[int,int])->np.ndarray:
    intrinsic=np.asarray(calibration["camera_intrinsics"],float);width,height=size
    uv=points.copy();uv[:,0]*=max(2*intrinsic[0,2],width)/width;uv[:,1]*=max(2*intrinsic[1,2],height)/height
    return uv


def _camera_to_world(camera:np.ndarray,calibration:dict[str,Any])->np.ndarray:
    sr=quat_rotation(calibration["sensor2ego_rotation"]);wr=quat_rotation(calibration["ego2global_rotation"])
    ego=camera@sr.T+np.asarray(calibration["sensor2ego_translation"],float)
    return ego@wr.T+np.asarray(calibration["ego2global_translation"],float)


def _world_to_pixel(world:np.ndarray,calibration:dict[str,Any],size:tuple[int,int])->np.ndarray:
    wr=quat_rotation(calibration["ego2global_rotation"]);sr=quat_rotation(calibration["sensor2ego_rotation"])
    ego=(world-np.asarray(calibration["ego2global_translation"],float))@wr
    camera=(ego-np.asarray(calibration["sensor2ego_translation"],float))@sr
    intrinsic=np.asarray(calibration["camera_intrinsics"],float);uvw=camera@intrinsic.T;uv=uvw[:,:2]/np.maximum(uvw[:,2:3],1e-6)
    width,height=size;uv[:,0]*=width/max(2*intrinsic[0,2],width);uv[:,1]*=height/max(2*intrinsic[1,2],height);return uv


def _fixed_depth_flow(points,source,target,depth,size):
    if not len(points):return np.empty((0,2))
    intrinsic=np.asarray(source["camera_intrinsics"],float);native=_native_uv(points,source,size)
    rays=np.column_stack([native,np.ones(len(native))])@np.linalg.inv(intrinsic).T
    camera=rays/rays[:,2:3]*depth;world=_camera_to_world(camera,source);return _world_to_pixel(world,target,size)-points


def _road_plane_flow(points,source,target,size,plane_z):
    if not len(points):return np.empty((0,2))
    intrinsic=np.asarray(source["camera_intrinsics"],float);native=_native_uv(points,source,size)
    rays=np.column_stack([native,np.ones(len(native))])@np.linalg.inv(intrinsic).T
    origin=_camera_to_world(np.zeros((1,3)),source)[0];world_unit=_camera_to_world(rays,source)-origin
    denominator=world_unit[:,2];valid=np.abs(denominator)>1e-6
    scale=np.where(valid,(plane_z-origin[2])/denominator,-1)
    if np.any(scale<=0):return None
    world=origin+world_unit*scale[:,None];return _world_to_pixel(world,target,size)-points


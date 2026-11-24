#!/usr/bin/env python3
"""Pure, side-effect-free trajectory replay scorer.

It never imports or instantiates ``TrafficManager.Model``.  All simulator data
is read from the frozen manifest and converted into local Python values before
scoring, so an Agent trajectory cannot advance SUMO/LimSim or alter its DB.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from driveaudit_eval.agents.core import (COMFORT_LIMITS, EGO_LENGTH, EGO_WIDTH, comfort, corners, dac_from_ratios,
                   ego_to_world, line_progress, obb_intersects, percentile, polygon_mask_ratio,
                   quat_rotation, route_progress, stable_hash, write_csv, write_json)


class MapCache:
    def __init__(self, cache_path: Path, manifest: Dict[str, Any]):
        self.cache_path=cache_path; self.available={}; self.h5=None
        for scene in manifest["scenes"]:
            for frame in scene["frames"]: self.available[frame["evaluation_only"]["map_cache_token"]]=frame["frame_id"]
        try:
            import h5py
            self.h5=h5py.File(cache_path,"r")
            self.keys=set(self.h5["gt_masks_bev_static"].keys())
        except Exception: self.keys=set()
    def mask(self, token: str) -> Optional[np.ndarray]:
        if not self.h5 or not self.keys: return None
        if token not in self.keys:
            target=self.available.get(token, 0); candidates=[(abs(self.available.get(k, 10**9)-target),k) for k in self.keys if k in self.available]
            if not candidates: return None
            token=min(candidates)[1]
        # one_hot_encode stores bit flags; class 0 is drivable_area.
        return (np.asarray(self.h5["gt_masks_bev_static"][token]) & 1).astype(bool)


def driving_score(pdms: float | None, route_completion: float) -> float | None:
    return None if pdms is None else float(pdms) * float(route_completion)


def box_world(row: Dict[str, str], frame: Dict[str, Any]) -> Dict[str, Any]:
    state=frame["model_input"]["ego_state"]
    p=np.array([float(row["x"]),float(row["y"]),float(row.get("z",0.0))])
    r_e2w=quat_rotation(state["rotation"]); t_e2w=np.asarray(state["position"],float)
    if all(row.get(key) not in (None, "") for key in ("global_x", "global_y", "global_yaw")):
        world=np.array([float(row["global_x"]),float(row["global_y"]),float(row.get("global_z",0.0))])
        yaw=float(row["global_yaw"])
    elif str(row.get("box_coordinate_system", "")).startswith("ego_"):
        world=r_e2w @ p + t_e2w
        heading=r_e2w @ np.array([math.cos(float(row["yaw"])),math.sin(float(row["yaw"])),0.0])
        yaw=math.atan2(heading[1],heading[0])
    else:
        r_l2e=quat_rotation(state["lidar2ego_rotation"]); t_l2e=np.asarray(state["lidar2ego_translation"],float)
        world=r_e2w @ (r_l2e @ p + t_l2e) + t_e2w
        heading=r_e2w @ r_l2e @ np.array([math.cos(float(row["yaw"])),math.sin(float(row["yaw"])),0.0])
        yaw=math.atan2(heading[1],heading[0])
    return {"id":row["obj_id"],"x":float(world[0]),"y":float(world[1]),"yaw":yaw,
            "width":float(row["width"]),"length":float(row["length"]),"class":row["class_name"]}


def evaluate_open_loop_trajectory(record: Dict[str, Any], frame: Dict[str, Any], frame_lookup: Dict[int, Dict[str, Any]],
                                  map_cache: Optional[MapCache], dt: float = .5) -> Dict[str, Any]:
    local=np.asarray(record.get("trajectory",[]),float)
    result={"scene_id":record.get("scene_id"),"frame_id":record.get("frame_id"),"agent":record.get("agent"),
            "inference_success":bool(record.get("inference_success")),"metric_type":"open_loop_proxy",
            "inference_latency_s":record.get("latency_s"),
            "inference_error":record.get("error")}
    if not result["inference_success"] or local.ndim!=2 or local.shape[0]<2 or local.shape[1]<2 or not np.isfinite(local).all():
        result.update({"valid_trajectory":False,"reason":"missing_or_invalid_trajectory"}); return result
    state=frame["model_input"]["ego_state"]; world=ego_to_world(local[:,:2],state["position"],float(state["yaw"]))
    offsets=np.asarray(record.get("time_offsets") or [(i+1)*dt for i in range(len(world))],float)
    # A trajectory may be inferred at the tail of the supplied sequence, but
    # cannot be replay-scored if the frozen simulator has no state for every
    # requested future timestamp.  Never treat absent actors as empty traffic.
    required_future_ids=[frame["frame_id"]+int(round(float(t)*12.0)) for t in offsets]
    if any(fid not in frame_lookup for fid in required_future_ids):
        result.update({"valid_trajectory":False,"reason":"insufficient_future_evaluation_state",
                       "evaluation_horizon_available":False})
        return result
    result["evaluation_horizon_available"]=True
    actual=np.asarray([frame_lookup[fid]["model_input"]["ego_state"]["position"][:2] for fid in required_future_ids],float)
    horizon_l2={}
    for horizon in (1,2,3):
        index=int(np.argmin(np.abs(offsets-horizon)))
        horizon_l2[f"planning_L2_{horizon}s"]=float(np.linalg.norm(world[index]-actual[index])) if abs(offsets[index]-horizon)<=.26 else None
    mask=map_cache.mask(frame["evaluation_only"]["map_cache_token"]) if map_cache else None
    ratios=[]; collision_objects=[]; first_collision=None; valid_steps=len(world)
    for i,(point,t) in enumerate(zip(world,offsets)):
        future_id=frame["frame_id"]+int(round(float(t)*12.0)); other_frame=frame_lookup.get(future_id)
        ego_poly=corners(point[0],point[1],float(state["yaw"]),EGO_LENGTH,EGO_WIDTH)
        collided=[]
        if other_frame:
            for row in other_frame["evaluation_only"]["background_boxes"]:
                other=box_world(row,other_frame)
                if obb_intersects(ego_poly,corners(other["x"],other["y"],other["yaw"],other["length"],other["width"])): collided.append(other["id"])
        collision_objects.append(collided)
        if collided and first_collision is None: first_collision=float(t); valid_steps=min(valid_steps,i)
        local_point=local[i,:2]
        ratio=polygon_mask_ratio(mask,local_point,0.0,EGO_LENGTH,EGO_WIDTH) if mask is not None else None
        ratios.append(ratio)
        if ratio is not None and ratio < .2 and valid_steps==len(world): valid_steps=min(valid_steps,i)
    finite_ratios=[x for x in ratios if x is not None]
    dac=dac_from_ratios(finite_ratios) if finite_ratios else None
    collision_any=first_collision is not None
    route=np.asarray(frame["evaluation_only"]["reference_route_positions"],float)
    progress, _=route_progress(route,np.asarray(state["position"][:2]),world,max(valid_steps,0))
    expected=max(1e-3,line_progress(route,route[0],route[min(len(route)-1,int(round(offsets[-1]*12)))])-line_progress(route,route[0],route[0]))
    ep=float(np.clip(progress/expected,0,1)); c=comfort(local[:,:2],dt)
    ttc_score=0.0 if collision_any and first_collision<=1.0 else 1.0
    min_ttc=first_collision
    pdms=None if dac is None else (1.0-float(collision_any))*dac*(5*ttc_score+2*c["comfort_score"]+5*ep)/12.0
    result.update({"valid_trajectory":True,"trajectory_world":world.tolist(),"collision_any":int(collision_any),
                   "collision_count":sum(bool(x) for x in collision_objects),"collision_object_id":collision_objects[int(first_collision/dt)-1][0] if collision_any else None,
                   "first_collision_time":first_collision,"collision_rate":float(collision_any),"NC":1.0-float(collision_any),
                   "DAC":dac,"minimum_drivable_ratio":min(finite_ratios) if finite_ratios else None,"mean_drivable_ratio":float(np.mean(finite_ratios)) if finite_ratios else None,
                   "offroad_any":int(any(x is not None and x<.2 for x in ratios)),"offroad_rate":float(sum(x is not None and x<.2 for x in ratios)/len(finite_ratios)) if finite_ratios else None,
                   "first_offroad_time":next((float(offsets[i]) for i,x in enumerate(ratios) if x is not None and x<.2),None),
                   "TTC_score":ttc_score,"min_ttc":min_ttc,"ttc_1s_risk":int(min_ttc is not None and min_ttc<1),"ttc_2s_risk":int(min_ttc is not None and min_ttc<2),"ttc_3s_risk":int(min_ttc is not None and min_ttc<3),
                   "collision_1s":int(min_ttc is not None and min_ttc<=1),"collision_2s":int(min_ttc is not None and min_ttc<=2),"collision_3s":int(min_ttc is not None and min_ttc<=3),
                   "EP":ep,"valid_progress":progress,"expected_progress":expected,"PDMS_open_loop":pdms,**horizon_l2,**{k:v for k,v in c.items() if k!="kinematics"}})
    result["comfort_violation_rate"]=float(len(c["comfort_violation_types"])/6)
    return result


def aggregate(rows: List[Dict[str,Any]], scene_id: str) -> Dict[str,Any]:
    valid=[r for r in rows if r.get("valid_trajectory")]
    successes=sum(bool(r.get("inference_success")) for r in rows)
    ret={"scene_id":scene_id,"input_frames":len(rows),"inference_success_frames":successes,
         "valid_frames":len(valid),"inference_success_rate":successes/len(rows) if rows else 0.0,
         "metric_type":"open_loop_proxy"}
    if not valid: return ret
    latencies=[float(r["inference_latency_s"]) for r in rows if r.get("inference_latency_s") is not None]
    if latencies:
        ret.update({"inference_latency_s_mean":float(np.mean(latencies)),
                    "inference_latency_s_p50":percentile(latencies,50),
                    "inference_latency_s_p90":percentile(latencies,90),
                    "inference_latency_s_p99":percentile(latencies,99)})
    # Resource samples are intentionally unavailable unless captured by a runner;
    # never substitute a post-run GPU reading for a peak measurement.
    ret["peak_gpu_memory_mb"]="not_available"
    ret["peak_cpu_memory_mb"]="not_available"
    for key in ("collision_rate","NC","DAC","TTC_score","comfort_score","EP","PDMS_open_loop","offroad_rate","comfort_violation_rate"):
        values=[r[key] for r in valid if r.get(key) is not None]
        ret[key+"_mean" if key not in {"NC","DAC","TTC_score","EP"} else key+"_mean"]=float(np.mean(values)) if values else None
    ret["NC_scene_strict"]=float(all(r["NC"]==1 for r in valid)); ret["PDMS_open_loop_median"]=percentile([r["PDMS_open_loop"] for r in valid if r["PDMS_open_loop"] is not None],50); ret["PDMS_open_loop_p10"]=percentile([r["PDMS_open_loop"] for r in valid if r["PDMS_open_loop"] is not None],10)
    for key in ("min_ttc","longitudinal_acceleration_max","lateral_acceleration_max","jerk_max","yaw_rate_max","yaw_acceleration_max","curvature_max"):
        values=[r[key] for r in valid if r.get(key) is not None]
        ret[key if key == "min_ttc" else key+"_dataset_max"]=min(values) if key=="min_ttc" and values else (max(values) if values else None)
    for key in ("ttc_1s_risk","ttc_2s_risk","ttc_3s_risk"):
        ret[key+"_mean"]=float(np.mean([r[key] for r in valid]))
    for key in ("planning_L2_1s","planning_L2_2s","planning_L2_3s","collision_1s","collision_2s","collision_3s"):
        values=[r[key] for r in valid if r.get(key) is not None]
        ret[key+"_mean"]=float(np.mean(values)) if values else None
    # Fixed 3-second stride at the 12 Hz source rate: these windows do not
    # overlap, so route progress is never counted once per adjacent image.
    window_rows=sorted(valid,key=lambda r:int(r["frame_id"]))[::36]
    numerator=sum(r["valid_progress"] for r in window_rows); denominator=sum(r["expected_progress"] for r in window_rows)
    ret.update({"RC_open_loop":float(np.clip(numerator/max(denominator,1e-6),0,1)),
                "rc_evaluation_stride_seconds":3.0,"rc_window_count":len(window_rows)})
    ret["Driving_Score_open_loop"]=driving_score(ret.get("PDMS_open_loop_mean"),ret["RC_open_loop"])
    return ret



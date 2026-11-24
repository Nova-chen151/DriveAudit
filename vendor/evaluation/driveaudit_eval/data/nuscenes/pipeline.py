from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
from pyquaternion import Quaternion

from .schema import CAMERAS, DYNAMIC_CLASSES, category, sha256, write_csv, write_json

_WORKER = {}


def _convert_worker(record):
    try:
        return convert_scene(_WORKER["nusc"],_WORKER["config"],_WORKER["roots"],record,_WORKER["resume"],_WORKER["force"]),None
    except Exception as exc:
        return None,{"scene_id":record["scene_name"],"error":f"{type(exc).__name__}: {exc}"}


def ensure_selection(nusc, config, roots, count=100):
    path=roots["input"]/"scene_selection.json"
    if not path.is_file():
        from .select_scenes import select
        return select(nusc,config,roots["input"])
    value=json.loads(path.read_text());
    if len(value["scenes"]) < count: raise RuntimeError(f"selection has {len(value['scenes'])}, need {count}")
    return value


def _link(source: Path, target: Path, mode="symlink") -> str:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink(): return "existing"
    if mode == "symlink": target.symlink_to(source.resolve()); return "symlink"
    if mode == "hardlink":
        try: os.link(source,target); return "hardlink"
        except OSError: shutil.copy2(source,target); return "copy_fallback"
    shutil.copy2(source,target); return "copy"


def _resize(source: Path, target: Path, size: tuple[int,int]):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file(): return
    image=cv2.imread(str(source));
    if image is None: raise ValueError(f"undecodable image: {source}")
    resized=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    tmp=target.with_suffix(".tmp.jpg")
    if not cv2.imwrite(str(tmp),resized,[cv2.IMWRITE_JPEG_QUALITY,95]): raise IOError(target)
    os.replace(tmp,target)


def _box_provenance(nusc, sd: dict, ann_token: str) -> dict:
    ann=nusc.get("sample_annotation",ann_token); sample=nusc.get("sample",sd["sample_token"])
    if sd["is_key_frame"] or not sample["prev"]:
        return {"annotation_source":"native_keyframe","source_prev_sample_token":sample["token"],"source_next_sample_token":sample["token"],"interpolation_alpha":0.0,"interpolation_confidence":1.0}
    previous=nusc.get("sample",sample["prev"]); span=sample["timestamp"]-previous["timestamp"]
    alpha=(sd["timestamp"]-previous["timestamp"])/span if span else 0.0
    return {"annotation_source":"official_devkit_interpolation","source_prev_sample_token":previous["token"],"source_next_sample_token":sample["token"],
            "interpolation_alpha":float(np.clip(alpha,0,1)),"interpolation_confidence":1.0 if 0 <= alpha <= 1 else 0.0}


def _boxes(nusc, scene_name: str, sd_token: str, frame_index: int, camera: str|None) -> list[dict]:
    sd=nusc.get("sample_data",sd_token); rows=[]
    for index,box in enumerate(nusc.get_boxes(sd_token)):
        ann=nusc.get("sample_annotation",box.token); mapped=category(box.name); velocity=nusc.box_velocity(box.token)
        yaw=float(box.orientation.yaw_pitch_roll[0]); provenance=_box_provenance(nusc,sd,box.token)
        rows.append({"frame_index":frame_index,"token":sd_token,"source_sample_token":sd["sample_token"],"object_index":index,
                     "obj_id":f"{scene_name}:{ann['instance_token']}","agent_id":f"{scene_name}:{ann['instance_token']}","instance_token":ann["instance_token"],
                     "annotation_token":box.token,"camera":camera,"sample_data_token":sd_token,"sample_data_timestamp":int(sd["timestamp"]),
                     "category_raw":box.name,"class_name":mapped,"category_mapped":mapped,"is_dynamic":mapped in DYNAMIC_CLASSES,
                     "x":float(box.center[0]),"y":float(box.center[1]),"z":float(box.center[2]),"width":float(box.wlh[0]),"length":float(box.wlh[1]),"height":float(box.wlh[2]),"yaw":yaw,
                     "vx":None if not np.isfinite(velocity[0]) else float(velocity[0]),"vy":None if not np.isfinite(velocity[1]) else float(velocity[1]),"velocity_source":"official_box_velocity",
                     "num_lidar_pts":ann["num_lidar_pts"],"num_radar_pts":ann["num_radar_pts"],"visibility":ann["visibility_token"],"valid_flag":True,**provenance})
    return rows


def _finite_differences(rows: list[dict]):
    grouped=defaultdict(list)
    for row in rows: grouped[row["agent_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda r:r["frame_index"])
        previous=None; previous_speed=None
        for row in values:
            row["speed_fd"]=row["acceleration_fd"]=row["yaw_rate_fd"]=row["vx_fd"]=row["vy_fd"]=None
            if previous and row["frame_index"] == previous["frame_index"]+1:
                dt=(row["sample_data_timestamp"]-previous["sample_data_timestamp"])/1e6
                if dt>0:
                    dx=(row["x"]-previous["x"])/dt;dy=(row["y"]-previous["y"])/dt;speed=math.hypot(dx,dy); row["speed_fd"]=speed;row["vx_fd"]=dx;row["vy_fd"]=dy
                    dyaw=math.atan2(math.sin(row["yaw"]-previous["yaw"]),math.cos(row["yaw"]-previous["yaw"]));row["yaw_rate_fd"]=dyaw/dt
                    if previous_speed is not None: row["acceleration_fd"]=(speed-previous_speed)/dt
                    previous_speed=speed
            else: previous_speed=None
            row["speed"]=math.hypot(row["vx"] or 0,row["vy"] or 0) if row["vx"] is not None else row["speed_fd"]
            row["velocity_final_source"]="official_box_velocity" if row["vx"] is not None else "adjacent_valid_finite_difference"
            previous=row


def _local_boxes(global_rows, common_ego):
    poses={int(row["frame_index"]):row for row in common_ego};result=[]
    for row in global_rows:
        pose=poses[int(row["frame_index"])];q=Quaternion(json.loads(pose["ego2global_rotation"]) if isinstance(pose["ego2global_rotation"],str) else pose["ego2global_rotation"])
        center=q.inverse.rotate(np.asarray([row["x"],row["y"],row["z"]])-np.asarray(json.loads(pose["ego2global_translation"]) if isinstance(pose["ego2global_translation"],str) else pose["ego2global_translation"]))
        vx=row["vx"] if row["vx"] is not None else row.get("vx_fd");vy=row["vy"] if row["vy"] is not None else row.get("vy_fd")
        velocity_available=vx is not None and vy is not None
        local_v=q.inverse.rotate([vx or 0.0,vy or 0.0,0.0]); ego_yaw=q.yaw_pitch_roll[0]
        result.append({**row,"global_x":row["x"],"global_y":row["y"],"global_z":row["z"],"global_yaw":row["yaw"],"x":float(center[0]),"y":float(center[1]),"z":float(center[2]),
                       "yaw":float(math.atan2(math.sin(row["yaw"]-ego_yaw),math.cos(row["yaw"]-ego_yaw))),"vx":float(local_v[0]),"vy":float(local_v[1]),
                       "velocity_available":velocity_available,"box_coordinate_system":"ego_at_cam_front_reference_timestamp"})
    return result


def _calibration(nusc, timeline, scene_name):
    rows=[]; per_camera={}; common=[]
    for frame in timeline:
        camera_poses={}
        for camera,item in frame["cameras"].items():
            sd=nusc.get("sample_data",item["sample_data_token"]); cs=nusc.get("calibrated_sensor",sd["calibrated_sensor_token"]); ego=nusc.get("ego_pose",sd["ego_pose_token"])
            value={"frame_index":frame["frame_index"],"camera":camera,"sample_data_token":sd["token"],"timestamp":sd["timestamp"],"data_path":sd["filename"],
                   "image_width":sd["width"],"image_height":sd["height"],"sensor2ego_translation":cs["translation"],"sensor2ego_rotation":cs["rotation"],
                   "ego2global_translation":ego["translation"],"ego2global_rotation":ego["rotation"],"camera_intrinsics":cs["camera_intrinsic"],"quaternion_convention":"wxyz"}
            rows.append({"token":sd["sample_token"],"type":camera,"sensor2lidar_rotation":Quaternion(cs["rotation"]).rotation_matrix.tolist(),"sensor2lidar_translation":cs["translation"],**value})
            camera_poses[camera]={"timestamp_us":sd["timestamp"],"ego_pose_token":sd["ego_pose_token"],"translation":ego["translation"],"rotation_wxyz":ego["rotation"]}
            per_camera.setdefault(camera,[]).append(camera_poses[camera])
            if camera=="CAM_FRONT":
                sample=nusc.get("sample",sd["sample_token"]); lidar=nusc.get("sample_data",sample["data"]["LIDAR_TOP"]); lcs=nusc.get("calibrated_sensor",lidar["calibrated_sensor_token"])
                common.append({"frame_index":frame["frame_index"],"token":sd["token"],"source_sample_token":sd["sample_token"],"timestamp_us":sd["timestamp"],"lidar2ego_translation":lcs["translation"],"lidar2ego_rotation":lcs["rotation"],"ego2global_translation":ego["translation"],"ego2global_rotation":ego["rotation"]})
        frame["per_camera_ego_pose"]=camera_poses
    return rows,per_camera,common


def _refresh_video_metadata(scene_root: Path):
    """Write exposure-time camera poses in one global frame for async epipolar checks."""
    metadata_path=scene_root/"metadata.json"; calibration_path=scene_root/"conditions/camera_calibration.csv"
    if not metadata_path.is_file() or not calibration_path.is_file(): return
    metadata=json.loads(metadata_path.read_text()); frames=defaultdict(dict)
    for row in csv.DictReader(calibration_path.open()):
        sensor_q=Quaternion(json.loads(row["sensor2ego_rotation"])); ego_q=Quaternion(json.loads(row["ego2global_rotation"]))
        rotation=ego_q.rotation_matrix@sensor_q.rotation_matrix
        translation=ego_q.rotation_matrix@np.asarray(json.loads(row["sensor2ego_translation"]),float)+np.asarray(json.loads(row["ego2global_translation"]),float)
        frames[int(row["frame_index"])][row["camera"]]={"K":json.loads(row["camera_intrinsics"]),"R":rotation.tolist(),"t":translation.tolist(),
            "image_size":[int(row["image_width"]),int(row["image_height"])],"timestamp_us":int(row["timestamp"]),"extrinsic_convention":"camera_to_global_at_native_exposure"}
    metadata["frames"]=[{"frame_index":index,"cameras":frames[index]} for index in sorted(frames)]
    metadata["video_geometry_extrinsic_convention"]="camera_to_global_at_native_exposure"
    write_json(metadata_path,metadata)


def _refresh_reference_frame_calibration(scene_root: Path):
    """Use the CAM_FRONT exposure-time ego frame as the shared BEV reference."""
    path=scene_root/"conditions/camera_calibration.csv"
    if not path.is_file(): return
    rows=list(csv.DictReader(path.open())); grouped=defaultdict(list)
    for row in rows: grouped[int(row["frame_index"])].append(row)
    common=[]
    for frame,values in grouped.items():
        reference=next(row for row in values if row["camera"]=="CAM_FRONT")
        ref_q=Quaternion(json.loads(reference["ego2global_rotation"])); ref_t=np.asarray(json.loads(reference["ego2global_translation"]),float)
        for row in values:
            ego_q=Quaternion(json.loads(row["ego2global_rotation"])); sensor_q=Quaternion(json.loads(row["sensor2ego_rotation"]))
            cam_r=ego_q.rotation_matrix@sensor_q.rotation_matrix
            cam_t=ego_q.rotation_matrix@np.asarray(json.loads(row["sensor2ego_translation"]),float)+np.asarray(json.loads(row["ego2global_translation"]),float)
            row["sensor2lidar_rotation"]=(ref_q.rotation_matrix.T@cam_r).tolist()
            row["sensor2lidar_translation"]=(ref_q.rotation_matrix.T@(cam_t-ref_t)).tolist()
        common.append({"frame_index":frame,"token":reference["sample_data_token"],"source_sample_token":reference["token"],"timestamp_us":int(reference["timestamp"]),
            "lidar2ego_translation":[0.0,0.0,0.0],"lidar2ego_rotation":[1.0,0.0,0.0,0.0],"ego2global_translation":ref_t.tolist(),"ego2global_rotation":list(ref_q.elements),
            "reference_frame":"ego_at_cam_front_native_exposure"})
    write_csv(path,rows);write_csv(scene_root/"conditions/ego_pose.csv",sorted(common,key=lambda row:row["frame_index"]));write_json(scene_root/"ego_pose/common_ego_pose.json",sorted(common,key=lambda row:row["frame_index"]))


def _map(nusc, scene, timeline, target: Path):
    from nuscenes.map_expansion.map_api import NuScenesMap
    from scipy.ndimage import distance_transform_edt
    location=nusc.get("log",scene["log_token"])["location"]; api=NuScenesMap(dataroot=nusc.dataroot,map_name=location)
    layers=[name for name in ("drivable_area","road_segment","lane","ped_crossing","walkway","stop_line","carpark_area","road_divider") if name in api.non_geometric_layers]
    target.mkdir(parents=True,exist_ok=True); h5path=target/"map_cache.h5"
    with h5py.File(h5path,"w") as h5:
        groups={name:h5.create_group(name) for name in ("gt_masks_bev","gt_masks_bev_static","gt_aux_bev")}
        for frame in timeline:
            pose=frame["per_camera_ego_pose"]["CAM_FRONT"]; x,y=pose["translation"][:2]; yaw=Quaternion(pose["rotation_wxyz"]).yaw_pitch_roll[0]
            masks=api.get_map_mask((x,y,100,100),math.degrees(yaw),layers,(200,200)).astype(np.uint8)
            packed=sum((masks[i].astype(np.int32)<<i) for i in range(len(layers)))
            aux=np.stack([(distance_transform_edt(mask)-distance_transform_edt(1-mask)).astype(np.float32) for mask in masks[:8]])
            key=frame["cameras"]["CAM_FRONT"]["sample_data_token"]
            groups["gt_masks_bev"].create_dataset(key,data=packed,compression="gzip");groups["gt_masks_bev_static"].create_dataset(key,data=packed,compression="gzip");groups["gt_aux_bev"].create_dataset(key,data=aux,compression="gzip")
    write_json(target/"map_metadata.json",{"source":"official_nuscenes_map_api","location":location,"layers":layers,"patch_size_m":[100,100],"canvas_size":[200,200],"traffic_light_state":"not_available_source_dataset"})
    return location,layers


def convert_scene(nusc,config,roots,record,resume=False,force=False):
    scene_name=record["scene_name"]; source_scene=next(s for s in nusc.scene if s["token"]==record["scene_token"])
    scene_input=roots["input"]/"scenes"/scene_name; done=scene_input/".conversion_complete.json"
    if done.is_file() and resume and not force: return json.loads(done.read_text())
    timeline_doc=json.loads((scene_input/"timeline.json").read_text()); timeline=timeline_doc["frames"]
    started=time.time(); size=tuple(config["challenge"]["resolution"]); modes=Counter()
    # Images are resized once into reference, then linked into isolated input/output views.
    for frame in timeline:
        index=frame["frame_index"]
        for camera,item in frame["cameras"].items():
            source=Path(item["source_path"]); ref_native=roots["reference"]/"native/scenes"/scene_name/camera/f"{index:06d}.jpg"
            modes[_link(source,ref_native,"symlink")]+=1
            ref=roots["reference"]/"scenes"/scene_name/camera/f"{index:06d}.jpg";_resize(source,ref,size)
            if index<config["challenge"]["history_frames"]:
                modes[_link(ref,scene_input/"history"/camera/f"{index:06d}.jpg","hardlink")]+=1
                modes[_link(ref,roots["input"]/"history_scenes"/scene_name/camera/f"{index:06d}.jpg","hardlink")]+=1
                modes[_link(source,scene_input/"native/history"/camera/f"{index:06d}.jpg","symlink")]+=1
            else:
                modes[_link(ref,roots["output"]/"scenes"/scene_name/camera/f"{index:06d}.jpg","hardlink")]+=1
                modes[_link(source,roots["output_native"]/"scenes"/scene_name/camera/f"{index:06d}.jpg","symlink")]+=1
                modes[_link(ref,roots["reference"]/"future_scenes"/scene_name/camera/f"{index:06d}.jpg","hardlink")]+=1
    calibration,per_camera_ego,common_ego=_calibration(nusc,timeline,scene_name)
    common_boxes=[]; per_camera_boxes={camera:[] for camera in CAMERAS}
    for frame in timeline:
        common_boxes += _boxes(nusc,scene_name,frame["cameras"]["CAM_FRONT"]["sample_data_token"],frame["frame_index"],None)
        for camera,item in frame["cameras"].items(): per_camera_boxes[camera] += _boxes(nusc,scene_name,item["sample_data_token"],frame["frame_index"],camera)
    _finite_differences(common_boxes);global_boxes=common_boxes;common_boxes=_local_boxes(global_boxes,common_ego)
    conditions=scene_input/"conditions"; conditions.mkdir(parents=True,exist_ok=True)
    frame_rows=[{"frame_index":f["frame_index"],"token":f["cameras"]["CAM_FRONT"]["sample_data_token"],"source_sample_token":f["cameras"]["CAM_FRONT"]["sample_token"],"scene_index":record.get("selection_index"),"scene_frame_index":f["frame_index"],
                 "timestamp":f["reference_timestamp_us"],"location":record["location"],"timeofday":"heuristic_from_description","description":record["description"],
                 "is_key_frame":f["cameras"]["CAM_FRONT"]["is_key_frame"],"lidar_path":"not_required_camera_timeline"} for f in timeline]
    write_csv(conditions/"frames.csv",frame_rows);write_csv(conditions/"boxes.csv",common_boxes);write_csv(conditions/"global_boxes.csv",global_boxes);write_csv(conditions/"camera_calibration.csv",calibration);write_csv(conditions/"ego_pose.csv",common_ego)
    write_json(conditions/"tracks.json",_tracks_json(scene_name,global_boxes,90));write_json(conditions/"per_camera_boxes/index.json",{"cameras":CAMERAS,"coordinate_system":"global_world"})
    for camera,rows in per_camera_boxes.items(): write_json(conditions/"per_camera_boxes"/f"{camera}.json",rows)
    write_json(scene_input/"calibration/cameras.json",{"camera_order":CAMERAS,"quaternion_convention":"wxyz","rows":calibration})
    write_json(scene_input/"ego_pose/common_ego_pose.json",common_ego);write_json(scene_input/"ego_pose/per_camera_ego_pose.json",per_camera_ego)
    write_json(scene_input/"timestamps.json",{"unit":"microseconds","frames":timeline})
    location,layers=_map(nusc,source_scene,timeline,scene_input/"map")
    if not (conditions/"map_cache.h5").exists(): (conditions/"map_cache.h5").symlink_to(Path("../map/map_cache.h5"))
    metadata={"source_dataset":"nuscenes","source_version":"v1.0-trainval","scene_token":record["scene_token"],"scene_name":scene_name,"clip_frame_range":[0,89],
              "history_frame_range":[0,9],"future_frame_range":[10,89],"camera_order":CAMERAS,"image_resolution":list(size),"trajectory_coordinate":"global_world",
              "trajectory_role":"generation_condition","quaternion_convention":"wxyz","box_center_convention":"geometric_center","box_size_order":"width_length_height",
              "interpolation_protocol":"NuScenes.get_boxes(sample_data_token)","challenge_split":record["challenge_split"],"location":location,"map_layers":layers,
              "future_rgb_in_input":False,"oracle_self_reference":True,"annotation_limit":"intermediate boxes are official devkit interpolation, not native human annotations"}
    write_json(scene_input/"metadata.json",metadata);_refresh_video_metadata(scene_input);write_json(conditions/"metadata.json",{"metadata":{"version":"v1.0-trainval"},"view_order":CAMERAS,"num_frames":90,"history_frames":10,"future_frames":80,"format":"nuscenes100_v1"})
    # Compatibility symlink for existing current minimal readers.
    minimal=scene_input/"minimal"; minimal.mkdir(exist_ok=True)
    if not (minimal/"conditions").exists(): (minimal/"conditions").symlink_to(Path("../conditions"))
    reference_scene=roots["reference"]/"scenes"/scene_name
    for name in ("conditions","calibration","ego_pose","timestamps.json","map","metadata.json"):
        source=scene_input/name; target=reference_scene/name
        if not target.exists(): target.symlink_to(os.path.relpath(source,target.parent))
    result={"scene_id":scene_name,"status":"completed","runtime_seconds":time.time()-started,"image_materialization":dict(modes),"common_box_rows":len(common_boxes),
            "per_camera_box_rows":sum(map(len,per_camera_boxes.values())),"dynamic_agents":len({r['agent_id'] for r in common_boxes if r['is_dynamic']}),"source_scene_token":record["scene_token"]}
    write_json(done,result);return result


def _tracks_json(scene_name,rows,total):
    grouped=defaultdict(list)
    for row in rows: grouped[row["agent_id"]].append(row)
    tracks=[]
    for agent,values in grouped.items():
        by={r["frame_index"]:r for r in values}; first=values[0]
        tracks.append({"scene_id":scene_name,"agent_id":agent,"instance_token":first["instance_token"],"category_raw":first["category_raw"],"category_mapped":first["category_mapped"],"is_dynamic":first["is_dynamic"],
                       "timestamps_us":[by[i]["sample_data_timestamp"] if i in by else None for i in range(total)],"valid_mask":[i in by for i in range(total)],
                       "positions_global":[[by[i][k] for k in ("x","y","z")] if i in by else None for i in range(total)],
                       "yaw_rad":[by[i]["yaw"] if i in by else None for i in range(total)],"size_wlh":[[by[i][k] for k in ("width","length","height")] if i in by else None for i in range(total)]})
    return {"scene_id":scene_name,"coordinate_system":"global_world","tracks":tracks}


def _canonical_trajectory_rows(scene_root: Path, scene_name: str):
    rows=list(csv.DictReader((scene_root/"conditions/boxes.csv").open()))
    frame_rows=list(csv.DictReader((scene_root/"conditions/frames.csv").open()));frame_times={int(r["frame_index"]):float(r["timestamp"])/1e6 for r in frame_rows}
    grouped=defaultdict(list)
    for row in rows:
        if row["is_dynamic"].lower() in {"true","1"}: grouped[row["agent_id"]].append(row)
    result=[]
    for values in grouped.values():
      if len(values)<2: continue
      by_frame={int(row["frame_index"]):row for row in values};available=sorted(by_frame)
      for frame in range(90):
        valid=frame in by_frame;nearest=min(available,key=lambda value:abs(value-frame));row=by_frame[frame] if valid else by_frame[nearest]
        mapped={"pedestrian":"pedestrian","bicycle":"bicycle","motorcycle":"bicycle"}.get(row["category_mapped"],"vehicle")
        result.append({"scene_id":scene_name,"agent_id":row["agent_id"],"agent_type":mapped,"timestamp":frame_times[frame],
                       "x":row["global_x"],"y":row["global_y"],"heading":row["global_yaw"],"valid":valid,"length":row["length"],"width":row["width"],"coordinate_system":"world","heading_unit":"rad",
                       "velocity":row.get("speed","") if valid else "","acceleration":row.get("acceleration_fd","") if valid else "","history_frames":10})
    return result


def convert_selected(nusc,config,roots,selection,count,resume=False,force=False,max_workers=1):
    records=selection["scenes"][:count]; results=[]; failures=[]
    indexed=[dict(record,selection_index=index) for index,record in enumerate(records)]
    if max_workers and max_workers>1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _WORKER.update(nusc=nusc,config=config,roots=roots,resume=resume,force=force)
        with ProcessPoolExecutor(max_workers=max_workers,mp_context=mp.get_context("fork")) as executor:
            completed=list(executor.map(_convert_worker,indexed))
    else:
        _WORKER.update(nusc=nusc,config=config,roots=roots,resume=resume,force=force);completed=[_convert_worker(record) for record in indexed]
    results=[value for value,error in completed if value];failures=[error for value,error in completed if error]
    for value in results:
        scene=value["scene_id"]
        _refresh_reference_frame_calibration(roots["input"]/"scenes"/scene)
        _refresh_video_metadata(roots["input"]/"scenes"/scene)
        value["hashes"]={"input_history":sha256(roots["input"]/"scenes"/scene/"history"),"conditions":sha256(roots["input"]/"scenes"/scene/"conditions"),
                         "gt_output_future":sha256(roots["output"]/"scenes"/scene),"reference_full":sha256(roots["reference"]/"scenes"/scene)}
    trajectory=[]
    for record in records:
        scene_root=roots["input"]/"scenes"/record["scene_name"]
        if (scene_root/".conversion_complete.json").is_file(): trajectory += _canonical_trajectory_rows(scene_root,record["scene_name"])
    write_csv(roots["output"]/"trajectories.csv",trajectory);write_csv(roots["reference"]/"trajectories.csv",trajectory)
    manifest={"dataset":config["dataset"]["name"],"status":"completed" if not failures and len(results)==count else "partial","requested":count,"completed":len(results),"failed":failures,
              "camera_order":CAMERAS,"history_frames":10,"future_frames":80,"oracle_self_reference":True,"submission_type":"gt_oracle_upper_bound","scenes":results}
    write_json(roots["input"]/"dataset_manifest.json",manifest);write_json(roots["output"]/"submission_manifest.json",manifest);write_json(roots["reference"]/"reference_manifest.json",manifest)
    write_json(roots["evaluation"]/"output_manifest.json",{"version":config["dataset"]["name"],"roots":{k:str(v.resolve()) for k,v in roots.items()},"compatibility_links":["input/scenes/<scene>/minimal/conditions -> ../conditions","reference/scenes/<scene>/{conditions,calibration,ego_pose,map,metadata} -> input scene"],"manifest":manifest})
    write_json(roots["evaluation"]/"runtime/conversion.json",manifest);return manifest

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from driveaudit_eval.consistency.shared.common import json_cell, quat_rotation, read_csv, write_json


BEVFORMER_CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
BEVFORMER_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def prepare_generated_resize_cache(source_root: str | Path, destination_root: str | Path,
                                   frame_indices: range, size: tuple[int, int] = (1600, 896)) -> dict[str, Any]:
    """Upsample generated pixels for the official 0.5 pipeline; this adds no visual information."""
    source_root, destination_root = Path(source_root), Path(destination_root)
    count = 0
    for camera in BEVFORMER_CAMERA_ORDER:
        target_dir = destination_root / camera; target_dir.mkdir(parents=True, exist_ok=True)
        for frame in frame_indices:
            source, target = source_root / camera / f"{frame:06d}.jpg", target_dir / f"{frame:06d}.jpg"
            if not target.is_file():
                with Image.open(source) as image:
                    image.resize(size, Image.Resampling.BILINEAR).save(target, quality=95)
            count += 1
    return {"source_size": [400, 224], "cache_size": list(size), "interpolation": "bilinear",
            "adds_visual_information": False, "images": count}


def build_bevformer_infos(frames_path: str | Path, calibration_path: str | Path,
                          ego_pose_path: str | Path, history_root: str | Path,
                          future_root: str | Path, history_frames: int,
                          frame_count: int, output_path: str | Path,
                          scene_token: str, frame_independent: bool = False) -> dict[str, Any]:
    """Build image-only nuScenes infos. No box or future-trajectory argument exists."""
    frames = sorted(read_csv(frames_path), key=lambda row: int(row["frame_index"]))[:frame_count]
    poses = {int(row["frame_index"]): row for row in read_csv(ego_pose_path)}
    calibrations = {(int(row["frame_index"]), row["camera"]): row for row in read_csv(calibration_path)}
    history_root, future_root = Path(history_root).resolve(), Path(future_root).resolve()
    infos = []
    for index, frame in enumerate(frames):
        pose = poses[index]
        lidar_to_ego_rotation = quat_rotation(json_cell(pose["lidar2ego_rotation"]))
        lidar_to_ego_translation = np.asarray(json_cell(pose["lidar2ego_translation"]), dtype=np.float64)
        cams = {}
        for camera in BEVFORMER_CAMERA_ORDER:
            calibration = calibrations[(index, camera)]
            image_root = history_root if index < history_frames else future_root
            image_path = image_root / camera / f"{index:06d}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            camera_to_ego_rotation = quat_rotation(json_cell(calibration["sensor2ego_rotation"]))
            camera_to_ego_translation = np.asarray(json_cell(calibration["sensor2ego_translation"]), dtype=np.float64)
            camera_to_lidar_rotation = lidar_to_ego_rotation.T @ camera_to_ego_rotation
            camera_to_lidar_translation = lidar_to_ego_rotation.T @ (camera_to_ego_translation - lidar_to_ego_translation)
            intrinsic = np.asarray(json_cell(calibration["camera_intrinsics"]), dtype=np.float64)
            with Image.open(image_path) as image:
                width, height = image.size
            intrinsic[0, :] *= width / float(calibration.get("image_width") or width)
            intrinsic[1, :] *= height / float(calibration.get("image_height") or height)
            cams[camera] = {
                "data_path": str(image_path),
                "sensor2lidar_rotation": camera_to_lidar_rotation,
                "sensor2lidar_translation": camera_to_lidar_translation,
                "cam_intrinsic": intrinsic,
            }
        infos.append({
            "lidar_path": frame.get("lidar_path", ""), "token": frame["token"], "sweeps": [], "cams": cams,
            "lidar2ego_translation": json_cell(pose["lidar2ego_translation"]),
            "lidar2ego_rotation": json_cell(pose["lidar2ego_rotation"]),
            "ego2global_translation": json_cell(pose["ego2global_translation"]),
            "ego2global_rotation": json_cell(pose["ego2global_rotation"]),
            "timestamp": int(float(frame["timestamp"])), "prev": frames[index - 1]["token"] if index else "",
            "next": frames[index + 1]["token"] if index + 1 < len(frames) else "",
            "scene_token": f"{scene_token}_{index}" if frame_independent else scene_token,
            "frame_idx": index, "can_bus": np.zeros(18, dtype=np.float64),
        })
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"infos": infos, "metadata": {"version": "v1.0-trainval"}}, handle)
    return {"frames": len(infos), "history_frames": history_frames, "future_frames": len(infos) - history_frames,
            "camera_order": BEVFORMER_CAMERA_ORDER, "future_truth_fields_present": False,
            "frame_independent": frame_independent}


def _set_image_scale(pipeline: Any, scale: float) -> None:
    if isinstance(pipeline, dict):
        if pipeline.get("type") == "RandomScaleImageMultiViewImage":
            pipeline["scales"] = [float(scale)]
        for value in pipeline.values():
            _set_image_scale(value, scale)
    elif isinstance(pipeline, (list, tuple)):
        for value in pipeline:
            _set_image_scale(value, scale)


def run_pixel_sequence(repo: str | Path, config_path: str | Path, checkpoint: str | Path,
                       infos_path: str | Path, output_path: str | Path,
                       profile_path: str | Path, resize_scale: float = 2.0) -> dict[str, Any]:
    """Run official BEVFormer sequentially. Evaluation truth is absent by construction."""
    repo = Path(repo).resolve(); sys.path.insert(0, str(repo))
    import mmcv
    import torch
    import projects.mmdet3d_plugin  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(Path(config_path).resolve()))
    cfg.model.pretrained = None; cfg.model.train_cfg = None
    cfg.data.test.ann_file = str(Path(infos_path).resolve()); cfg.data.test.data_root = ""; cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 0; cfg.data.test.pop("samples_per_gpu", None)
    _set_image_scale(cfg.data.test.pipeline, resize_scale)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint_data = load_checkpoint(model, str(Path(checkpoint).resolve()), map_location="cpu", strict=True)
    model.CLASSES = checkpoint_data.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model.cuda(), device_ids=[0]); model.eval()
    torch.cuda.reset_peak_memory_stats()
    outputs, latencies = [], []
    for frame_index, data in enumerate(loader):
        info = dataset.data_infos[frame_index]
        start = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]["pts_bbox"]
        torch.cuda.synchronize(); latencies.append(time.perf_counter() - start)
        boxes = result["boxes_3d"].tensor.detach().cpu().numpy()
        outputs.append({"frame_index": int(info.get("frame_idx", frame_index)),
                        "scene_token": info.get("scene_token"), "sample_token": info.get("token"),
                        "boxes": boxes,
                        "scores": result["scores_3d"].detach().cpu().numpy(),
                        "labels": result["labels_3d"].detach().cpu().numpy()})
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(outputs, str(output_path))
    profile = {"pixel_inference_performed": True, "checkpoint_used": True, "strict_checkpoint_load": True,
               "gpu_used": True, "frames": len(outputs), "resize_scale": resize_scale,
               "mean_inference_seconds": float(np.mean(latencies)), "total_inference_seconds": float(np.sum(latencies)),
               "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated() / 2**20),
               "checkpoint_epoch": checkpoint_data.get("meta", {}).get("epoch")}
    write_json(profile_path, profile)
    return profile





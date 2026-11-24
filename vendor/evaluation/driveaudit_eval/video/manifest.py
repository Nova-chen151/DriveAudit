from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from driveaudit_eval.video.io import SceneSample

BENCHMARK_PROTOCOL = "driveaudit_eval_nuscenes_dreamforge_dataset_level_engine_v1"

def _resolve_path(value: Optional[Any], base_dir: Path) -> Optional[Path]:
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path_string(value: Optional[Any], base_dir: Path) -> Optional[str]:
    path = _resolve_path(value, base_dir)
    return str(path) if path is not None else None


def resolve_config_paths(cfg: Dict[str, Any], config_dir: Path) -> None:
    weights = cfg.get("weights", {})
    for key in ("root", "dino_repo", "dino_vits14", "loftr_outdoor", "yolo", "raft_dir", "fvd_i3d_torchscript", "fid_inception_v3"):
        if weights.get(key) is not None:
            weights[key] = _resolve_optional_path_string(weights[key], config_dir)
    metrics = cfg.get("metrics", {})
    if metrics.get("fid", {}).get("weights_path") is not None:
        metrics["fid"]["weights_path"] = _resolve_optional_path_string(metrics["fid"]["weights_path"], config_dir)
    if metrics.get("fvd", {}).get("weights_path") is not None:
        metrics["fvd"]["weights_path"] = _resolve_optional_path_string(metrics["fvd"]["weights_path"], config_dir)
    if metrics.get("loftr", {}).get("weights_path") is not None:
        metrics["loftr"]["weights_path"] = _resolve_optional_path_string(metrics["loftr"]["weights_path"], config_dir)
    if metrics.get("boundary", {}).get("dino", {}).get("weights_path") is not None:
        metrics["boundary"]["dino"]["weights_path"] = _resolve_optional_path_string(
            metrics["boundary"]["dino"]["weights_path"], config_dir
        )
    if metrics.get("tracking", {}).get("yolo_weights_path") is not None:
        metrics["tracking"]["yolo_weights_path"] = _resolve_optional_path_string(
            metrics["tracking"]["yolo_weights_path"], config_dir
        )


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix.lower(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _path_list_signature(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    return [_file_signature(path) for path in paths]


def build_dataset_index(samples: Sequence[SceneSample], cameras: Sequence[str]) -> Dict[str, Any]:
    scenes = []
    for sample in sorted(samples, key=lambda item: item.scene_id):
        scene_cameras = [camera for camera in cameras if camera in sample.cameras]
        scenes.append(
            {
                "scene_id": sample.scene_id,
                "cameras": scene_cameras,
                "metadata": _file_signature(sample.metadata_path) if sample.metadata_path else None,
                "history": {
                    camera: _path_list_signature(sample.history_images.get(camera, []))
                    for camera in scene_cameras
                },
                "generated_future": {
                    camera: _path_list_signature(sample.future_images.get(camera, []))
                    for camera in scene_cameras
                },
                "real_reference": {
                    camera: _path_list_signature(sample.real_reference_images.get(camera, []))
                    for camera in scene_cameras
                },
            }
        )
    scene_order = [scene["scene_id"] for scene in scenes]
    return {
        "scene_order": scene_order,
        "scene_order_hash": _json_hash(scene_order),
        "scenes": scenes,
    }


def enrich_manifest_with_locks(manifest: Dict[str, Any], dataset_index: Dict[str, Any]) -> Dict[str, Any]:
    manifest = deepcopy(manifest)
    manifest["version"] = 2
    manifest["benchmark_protocol"] = BENCHMARK_PROTOCOL
    manifest["dataset_version_hash"] = _json_hash(dataset_index)
    manifest["scene_order_hash"] = dataset_index["scene_order_hash"]
    manifest["dataset_index"] = dataset_index
    manifest["frame_index_map"] = {
        entry["scene_id"]: list(entry.get("frame_positions", entry.get("frame_indices", [])))
        for entry in manifest.get("scenes", [])
    }
    manifest["clip_index_map"] = {
        entry["scene_id"]: {
            camera: list(starts)
            for camera, starts in entry.get("clip_starts", {}).items()
        }
        for entry in manifest.get("scenes", [])
    }
    return manifest


def enforce_manifest_lock(manifest: Dict[str, Any], dataset_index: Dict[str, Any]) -> None:
    expected_dataset_hash = _json_hash(dataset_index)
    expected_scene_hash = dataset_index["scene_order_hash"]
    observed_dataset_hash = manifest.get("dataset_version_hash")
    observed_scene_hash = manifest.get("scene_order_hash")
    if not observed_dataset_hash:
        raise ValueError("evaluation_manifest is missing dataset_version_hash; regenerate a locked manifest.")
    if not observed_scene_hash:
        raise ValueError("evaluation_manifest is missing scene_order_hash; regenerate a locked manifest.")
    if observed_dataset_hash != expected_dataset_hash:
        raise ValueError(
            "evaluation_manifest dataset_version_hash mismatch: "
            f"manifest={observed_dataset_hash}, current={expected_dataset_hash}"
        )
    if observed_scene_hash != expected_scene_hash:
        raise ValueError(
            "evaluation_manifest scene_order_hash mismatch: "
            f"manifest={observed_scene_hash}, current={expected_scene_hash}"
        )

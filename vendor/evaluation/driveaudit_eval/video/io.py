from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from driveaudit_eval.video.utils import ensure_dir


@dataclass
class SceneSample:
    scene_id: str
    cameras: List[str]
    history_images: Dict[str, List[Path]]
    future_images: Dict[str, List[Path]]
    real_reference_images: Dict[str, List[Path]] = field(default_factory=dict)
    metadata_path: Optional[Path] = None
    camera_metadata: Dict[int, Dict[str, dict]] = field(default_factory=dict)
    image_sizes: Dict[str, tuple[int, int]] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def has_real_reference(self) -> bool:
        return any(self.real_reference_images.get(camera) for camera in self.cameras)


def sorted_images(camera_dir: Path) -> List[Path]:
    if not camera_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in camera_dir.iterdir() if path.suffix.lower() in exts)


def samples_from_sources(sources, cameras):
    """Use existing files directly, retaining the adapter's exposure filtering."""
    samples = []
    for row in sources:
        groups = {key: {camera: [Path(path) for path in row[key].get(camera, [])] for camera in cameras}
                  for key in ('history_images', 'future_images', 'real_reference_images')}
        metadata = Path(row['metadata_path'])
        samples.append(SceneSample(scene_id=row['scene_id'], cameras=list(cameras), **groups,
            metadata_path=metadata, camera_metadata=parse_json_metadata(metadata),
            image_sizes=infer_camera_image_sizes(groups['history_images']),
            extra={'adapter': 'input_gt_direct_sources'}))
    return samples


def camera_dirs(root: Path, cameras: Sequence[str]) -> Dict[str, List[Path]]:
    return {camera: sorted_images(root / camera) for camera in cameras}


def resolve_history_images(
    history_root: Optional[Path],
    scene_id: str,
    cameras: Sequence[str],
) -> Dict[str, List[Path]]:
    empty = {camera: [] for camera in cameras}
    if history_root is None:
        return empty
    root = Path(history_root)
    if not root.exists():
        return empty
    candidates = [root / scene_id, root]
    for candidate in candidates:
        if all((candidate / camera).exists() for camera in cameras):
            return camera_dirs(candidate, cameras)
    return empty


def infer_camera_image_sizes(image_groups: Dict[str, List[Path]]) -> Dict[str, tuple[int, int]]:
    sizes: Dict[str, tuple[int, int]] = {}
    for camera, paths in image_groups.items():
        if not paths:
            continue
        try:
            with Image.open(paths[0]) as image:
                sizes[camera] = image.size
        except Exception:
            continue
    return sizes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_cell(value: str):
    if value is None or value == "":
        return None
    return json.loads(value)


def parse_camera_csv(path: Path) -> Dict[int, Dict[str, dict]]:
    if not path.exists():
        return {}
    ret: Dict[int, Dict[str, dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_index = int(row["frame_index"])
            camera = row["camera"]
            k = parse_json_cell(row.get("camera_intrinsics", ""))
            r = parse_json_cell(row.get("sensor2lidar_rotation", ""))
            t = parse_json_cell(row.get("sensor2lidar_translation", ""))
            if k is None or r is None or t is None:
                continue
            ret.setdefault(frame_index, {})[camera] = {
                "K": np.asarray(k, dtype=np.float64),
                "R": np.asarray(r, dtype=np.float64),
                "t": np.asarray(t, dtype=np.float64).reshape(3),
                "data_path": row.get("data_path", ""),
                "extrinsic_convention_hint": "sensor_to_ego",
            }
    return ret


def load_metadata(metadata_root: Optional[Path], scene_id: str) -> tuple[Optional[Path], Dict[int, Dict[str, dict]]]:
    if metadata_root is None:
        return None, {}
    metadata_root = Path(metadata_root)
    if metadata_root.is_file():
        if metadata_root.suffix.lower() == ".json":
            return metadata_root, parse_json_metadata(metadata_root)
        if metadata_root.name.endswith(".csv"):
            return metadata_root, parse_camera_csv(metadata_root)
    candidates = [
        metadata_root / f"{scene_id}.json",
        metadata_root / scene_id / "metadata.json",
        metadata_root / "camera_calibration.csv",
        metadata_root / "conditions" / "camera_calibration.csv",
        metadata_root / "input" / "minimal" / "conditions" / "camera_calibration.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() == ".json":
                return candidate, parse_json_metadata(candidate)
            return candidate, parse_camera_csv(candidate)
    return None, {}


def parse_json_metadata(path: Path) -> Dict[int, Dict[str, dict]]:
    data = read_json(path)
    frames = data.get("frames", data)
    ret: Dict[int, Dict[str, dict]] = {}
    if isinstance(frames, list):
        for frame in frames:
            frame_index = int(frame.get("frame_index", frame.get("frame_id", 0)))
            cameras = frame.get("cameras", {})
            for camera, item in cameras.items():
                if {"K", "R", "t"}.issubset(item):
                    ret.setdefault(frame_index, {})[camera] = {
                        "K": np.asarray(item["K"], dtype=np.float64),
                        "R": np.asarray(item["R"], dtype=np.float64),
                        "t": np.asarray(item["t"], dtype=np.float64),
                        "image_size": tuple(item["image_size"])
                        if item.get("image_size")
                        else None,
                    }
    elif isinstance(frames, dict):
        for frame_key, cameras in frames.items():
            try:
                frame_index = int(frame_key)
            except ValueError:
                continue
            for camera, item in cameras.items():
                if {"K", "R", "t"}.issubset(item):
                    ret.setdefault(frame_index, {})[camera] = {
                        "K": np.asarray(item["K"], dtype=np.float64),
                        "R": np.asarray(item["R"], dtype=np.float64),
                        "t": np.asarray(item["t"], dtype=np.float64),
                        "image_size": tuple(item["image_size"])
                        if item.get("image_size")
                        else None,
                    }
    return ret


def find_worlddreamer_bundle(run_dir: Path) -> Optional[Path]:
    for parent in [run_dir, *run_dir.parents]:
        if (parent / "input" / "history").exists() and (parent / "input" / "minimal").exists():
            return parent
    return None


def scene_from_worlddreamer_run(
    run_dir: Path,
    cameras: Sequence[str],
    metadata_root: Optional[Path],
    real_reference_root: Optional[Path],
    history_root: Optional[Path] = None,
) -> Optional[SceneSample]:
    future_root = run_dir / "future_generated"
    if not future_root.exists():
        return None
    bundle = find_worlddreamer_bundle(run_dir)
    resolved_history_root = (
        Path(history_root)
        if history_root is not None
        else bundle / "input" / "history"
        if bundle is not None
        else run_dir / "history"
    )
    if not resolved_history_root.exists():
        return None
    scene_id = run_dir.name
    if metadata_root is None and bundle is not None:
        metadata_root = bundle / "input" / "minimal"
    metadata_path, camera_metadata = load_metadata(metadata_root, scene_id)
    real_images = {}
    if real_reference_root is not None:
        real_images = discover_real_reference(Path(real_reference_root), scene_id, cameras)
    history_images = resolve_history_images(resolved_history_root, scene_id, cameras)
    return SceneSample(
        scene_id=scene_id,
        cameras=list(cameras),
        history_images=history_images,
        future_images=camera_dirs(future_root, cameras),
        real_reference_images=real_images,
        metadata_path=metadata_path,
        camera_metadata=camera_metadata,
        image_sizes=infer_camera_image_sizes(history_images),
        extra={"adapter": "worlddreamer_run", "bundle": str(bundle) if bundle else None},
    )


def discover_real_reference(
    real_root: Path, scene_id: str, cameras: Sequence[str]
) -> Dict[str, List[Path]]:
    if not real_root.exists():
        return {}
    candidates = [real_root / scene_id, real_root]
    for candidate in candidates:
        if all((candidate / camera).exists() for camera in cameras):
            return camera_dirs(candidate, cameras)
    scenes = [path for path in sorted(real_root.iterdir()) if path.is_dir()]
    collected: Dict[str, List[Path]] = {camera: [] for camera in cameras}
    for scene in scenes:
        for camera in cameras:
            collected[camera].extend(sorted_images(scene / camera))
    return collected


def scene_from_generic_root(
    root: Path,
    cameras: Sequence[str],
    metadata_root: Optional[Path],
    real_reference_root: Optional[Path],
    history_root: Optional[Path] = None,
) -> Optional[SceneSample]:
    if all((root / camera).exists() for camera in cameras):
        scene_id = root.name
        metadata_path, camera_metadata = load_metadata(metadata_root, scene_id)
        real_images = (
            discover_real_reference(Path(real_reference_root), scene_id, cameras)
            if real_reference_root
            else {}
        )
        future_images = camera_dirs(root, cameras)
        history_images = resolve_history_images(history_root, scene_id, cameras)
        image_sizes = infer_camera_image_sizes(history_images)
        image_sizes.update(
            {
                camera: size
                for camera, size in infer_camera_image_sizes(future_images).items()
                if camera not in image_sizes
            }
        )
        return SceneSample(
            scene_id=scene_id,
            cameras=list(cameras),
            history_images=history_images,
            future_images=future_images,
            real_reference_images=real_images,
            metadata_path=metadata_path,
            camera_metadata=camera_metadata,
            image_sizes=image_sizes,
            extra={"adapter": "future_camera_tree"},
        )
    if (root / "history").exists() and (root / "future").exists():
        scene_id = root.name
        metadata_path, camera_metadata = load_metadata(metadata_root, scene_id)
        real_images = (
            discover_real_reference(Path(real_reference_root), scene_id, cameras)
            if real_reference_root
            else {}
        )
        history_images = camera_dirs(root / "history", cameras)
        return SceneSample(
            scene_id=scene_id,
            cameras=list(cameras),
            history_images=history_images,
            future_images=camera_dirs(root / "future", cameras),
            real_reference_images=real_images,
            metadata_path=metadata_path,
            camera_metadata=camera_metadata,
            image_sizes=infer_camera_image_sizes(history_images),
            extra={"adapter": "generic_scene"},
        )
    return None


def discover_samples(
    dataset_root: Optional[Path],
    generated_root: Path,
    cameras: Sequence[str],
    metadata_root: Optional[Path] = None,
    real_reference_root: Optional[Path] = None,
    history_root: Optional[Path] = None,
    max_scenes: Optional[int] = None,
) -> List[SceneSample]:
    generated_root = Path(generated_root).resolve()
    samples: List[SceneSample] = []

    direct = scene_from_worlddreamer_run(
        generated_root, cameras, metadata_root, real_reference_root, history_root=history_root
    )
    if direct is None:
        direct = scene_from_generic_root(
            generated_root, cameras, metadata_root, real_reference_root, history_root=history_root
        )
    if direct is not None:
        samples.append(direct)
    else:
        for scene_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
            sample = scene_from_generic_root(
                scene_dir, cameras, metadata_root, real_reference_root, history_root=history_root
            )
            if sample is None:
                sample = scene_from_worlddreamer_run(
                    scene_dir, cameras, metadata_root, real_reference_root, history_root=history_root
                )
            if sample is not None:
                samples.append(sample)
            if max_scenes is not None and len(samples) >= max_scenes:
                break

    if not samples and dataset_root is not None:
        dataset_root = Path(dataset_root)
        candidate = dataset_root / "generated"
        if candidate.exists() and candidate != generated_root:
            return discover_samples(
                dataset_root=None,
                generated_root=candidate,
                cameras=cameras,
                metadata_root=metadata_root or dataset_root / "metadata",
                real_reference_root=real_reference_root or dataset_root / "real_reference",
                history_root=history_root or dataset_root / "input" / "history",
                max_scenes=max_scenes,
            )
    return samples[:max_scenes] if max_scenes is not None else samples


def validate_sample(sample: SceneSample) -> List[str]:
    warnings = []
    for camera in sample.cameras:
        if not sample.history_images.get(camera):
            warnings.append(f"missing history images for {camera}")
        if not sample.future_images.get(camera):
            warnings.append(f"missing future images for {camera}")
    return warnings

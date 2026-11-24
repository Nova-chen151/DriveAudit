from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from driveaudit_eval.video.config import CAMERAS


CAMERA_PRESETS = {
    "front": ["CAM_FRONT"],
    "front_triple": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    "surround": CAMERAS,
}


@dataclass(frozen=True)
class ImageSample:
    domain: str
    scene_id: str
    camera: str
    frame_position: int
    frame_id: int
    path: Path


@dataclass(frozen=True)
class ClipSample:
    domain: str
    scene_id: str
    camera: str
    start_position: int
    frame_positions: Tuple[int, ...]
    frame_ids: Tuple[int, ...]
    paths: Tuple[Path, ...]


@dataclass
class SamplingPlan:
    seed: int
    cameras: List[str]
    frames_per_scene: int
    clip_length: int
    stride: int
    source_fps: float
    target_fps: float
    real_images: List[ImageSample]
    fake_images: List[ImageSample]
    real_clips: List[ClipSample]
    fake_clips: List[ClipSample]
    scene_frame_positions: Dict[str, List[int]]
    scene_clip_starts: Dict[str, Dict[str, List[int]]]
    manifest: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "camera_set": self.cameras,
            "frames_per_scene": self.frames_per_scene,
            "clip_length": self.clip_length,
            "stride": self.stride,
            "time_policy": self.manifest.get("time_policy", "nominal_fps_normalization"),
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "num_real_images": len(self.real_images),
            "num_fake_images": len(self.fake_images),
            "num_real_clips": len(self.real_clips),
            "num_fake_clips": len(self.fake_clips),
            "scene_frame_positions": self.scene_frame_positions,
            "scene_clip_starts": self.scene_clip_starts,
            "manifest_version": self.manifest.get("version"),
        }


def resolve_sampling_cameras(cfg: Dict[str, Any]) -> List[str]:
    sampling = cfg.get("sampling", {})
    explicit = sampling.get("cameras")
    if explicit:
        cameras = list(explicit)
    else:
        preset = sampling.get("camera_set", "surround")
        cameras = CAMERA_PRESETS.get(preset, cfg.get("dataset", {}).get("cameras", CAMERAS))
    allowed = set(cfg.get("dataset", {}).get("cameras", CAMERAS))
    return [camera for camera in cameras if camera in allowed]


def _frame_id(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def _stable_seed(seed: int, parts: Iterable[Any]) -> int:
    text = "\0".join(str(part) for part in parts)
    value = seed & 0xFFFFFFFF
    for byte in text.encode("utf-8"):
        value = ((value * 131) + byte) & 0xFFFFFFFF
    return value


def _sample_positions(count: int, requested: int, seed: int, scene_id: str, tag: str) -> List[int]:
    if count <= 0:
        return []
    if requested <= 0 or requested >= count:
        return list(range(count))
    rng = random.Random(_stable_seed(seed, [scene_id, tag]))
    return sorted(rng.sample(range(count), requested))


def _sample_starts(starts: List[int], max_count: Optional[int], seed: int, scene_id: str, camera: str) -> List[int]:
    if max_count is None or max_count <= 0 or max_count >= len(starts):
        return starts
    rng = random.Random(_stable_seed(seed, [scene_id, camera, "clips"]))
    return sorted(rng.sample(starts, max_count))


def _dedupe_images(samples: Sequence[ImageSample]) -> List[ImageSample]:
    seen = set()
    ret: List[ImageSample] = []
    for sample in samples:
        key = (sample.domain, str(sample.path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        ret.append(sample)
    return ret


def _dedupe_clips(samples: Sequence[ClipSample]) -> List[ClipSample]:
    seen = set()
    ret: List[ClipSample] = []
    for sample in samples:
        key = (sample.domain, tuple(str(path.resolve()) for path in sample.paths))
        if key in seen:
            continue
        seen.add(key)
        ret.append(sample)
    return ret


def _fps_step(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0 or target_fps >= source_fps:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def _normalize_fps(paths: Sequence[Path], source_fps: float, target_fps: float) -> List[Path]:
    step = _fps_step(source_fps, target_fps)
    return list(paths)[::step]


def aligned_real_future_paths(scene_sample, camera: str, cfg: Dict[str, Any]) -> List[Path]:
    real_paths = list(scene_sample.real_reference_images.get(camera, []))
    fake_paths = list(scene_sample.future_images.get(camera, []))
    if not real_paths:
        return []
    if not fake_paths:
        return real_paths
    real_by_stem = {path.stem: path for path in real_paths}
    if all(path.stem in real_by_stem for path in fake_paths):
        return [real_by_stem[path.stem] for path in fake_paths]
    history_frames = int(cfg.get("dataset", {}).get("history_frames", 0))
    if len(real_paths) >= len(fake_paths) + history_frames:
        return real_paths[history_frames : history_frames + len(fake_paths)]
    return real_paths[: len(fake_paths)]


def _camera_sequences(scene_sample, camera: str, cfg: Dict[str, Any]) -> Tuple[List[Path], List[Path]]:
    sampling = cfg.get("sampling", {})
    source_fps = float(sampling.get("source_fps", 12))
    target_fps = float(sampling.get("target_fps", source_fps))
    normalize = (lambda paths: list(paths)) if sampling.get('time_policy') == 'native_order' else (lambda paths: _normalize_fps(paths, source_fps, target_fps))
    fake = normalize(scene_sample.future_images.get(camera, []))
    real = normalize(aligned_real_future_paths(scene_sample, camera, cfg))
    count = min(len(fake), len(real))
    return real[:count], fake[:count] if real else fake


def build_evaluation_manifest(samples: Sequence[Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    sampling = cfg.get("sampling", {})
    seed = int(sampling.get("seed", 42))
    cameras = resolve_sampling_cameras(cfg)
    frames_per_scene = int(sampling.get("frames_per_scene", 50))
    clip_length = int(sampling.get("clip_length", sampling.get("num_frames", 16)))
    stride = int(sampling.get("stride", sampling.get("clip_stride", 2)))
    source_fps = float(sampling.get("source_fps", 12))
    target_fps = float(sampling.get("target_fps", source_fps))
    max_clips = sampling.get("max_clips_per_scene_camera")
    max_clips = None if max_clips is None else int(max_clips)

    scene_entries = []
    for scene_sample in sorted(samples, key=lambda item: item.scene_id):
        usable_cameras = [camera for camera in cameras if scene_sample.future_images.get(camera)]
        lengths = []
        for camera in usable_cameras:
            real, fake = _camera_sequences(scene_sample, camera, cfg)
            if real and fake:
                lengths.append(min(len(real), len(fake)))
            elif fake and not any(cfg.get("metrics", {}).get(name, {}).get("enabled", True) for name in ("fid", "fvd")):
                # Reference-free diagnostics still need a locked frame selection.
                lengths.append(len(fake))
        if not lengths:
            continue
        common_count = min(lengths)
        frame_positions = _sample_positions(common_count, frames_per_scene, seed, scene_sample.scene_id, "images")
        clip_starts = {}
        for camera in usable_cameras:
            starts = list(range(0, common_count - clip_length + 1, stride))
            clip_starts[camera] = _sample_starts(starts, max_clips, seed, scene_sample.scene_id, camera)
        scene_entries.append(
            {
                "scene_id": scene_sample.scene_id,
                "num_aligned_frames": common_count,
                "cameras": list(usable_cameras),
                "frame_positions": frame_positions,
                "frame_indices": frame_positions,
                "clip_starts": clip_starts,
            }
        )

    return {
        "version": 1,
        "time_policy": sampling.get("time_policy", "nominal_fps_normalization"),
        "seed": seed,
        "scene_order": [entry["scene_id"] for entry in scene_entries],
        "camera_selection": {
            "preset": sampling.get("camera_set", "surround"),
            "cameras": cameras,
        },
        "frame_sampling": {
            "frames_per_scene": frames_per_scene,
            "source_fps": source_fps,
            "target_fps": target_fps,
        },
        "clip_sampling": {
            "clip_length": clip_length,
            "stride": stride,
            "source_fps": source_fps,
            "target_fps": target_fps,
            "max_clips_per_scene_camera": max_clips,
        },
        "scenes": scene_entries,
    }


def build_sampling_plan(
    samples: Sequence[Any],
    cfg: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
) -> SamplingPlan:
    if manifest is None:
        manifest = build_evaluation_manifest(samples, cfg)
    sampling = cfg.get("sampling", {})
    seed = int(manifest.get("seed", sampling.get("seed", 42)))
    cameras = list(manifest.get("camera_selection", {}).get("cameras") or resolve_sampling_cameras(cfg))
    frames_per_scene = int(manifest.get("frame_sampling", {}).get("frames_per_scene", sampling.get("frames_per_scene", 50)))
    clip_cfg = manifest.get("clip_sampling", {})
    clip_length = int(clip_cfg.get("clip_length", sampling.get("clip_length", sampling.get("num_frames", 16))))
    stride = int(clip_cfg.get("stride", sampling.get("stride", sampling.get("clip_stride", 2))))
    source_fps = float(clip_cfg.get("source_fps", sampling.get("source_fps", 12)))
    target_fps = float(clip_cfg.get("target_fps", sampling.get("target_fps", source_fps)))

    real_images: List[ImageSample] = []
    fake_images: List[ImageSample] = []
    real_clips: List[ClipSample] = []
    fake_clips: List[ClipSample] = []
    scene_frame_positions: Dict[str, List[int]] = {}
    scene_clip_starts: Dict[str, Dict[str, List[int]]] = {}
    samples_by_id = {sample.scene_id: sample for sample in samples}

    for scene_entry in manifest.get("scenes", []):
        scene_id = scene_entry["scene_id"]
        if scene_id not in samples_by_id:
            raise ValueError(f"evaluation_manifest references missing scene: {scene_id}")
        scene_sample = samples_by_id[scene_id]
        usable_cameras = [camera for camera in scene_entry.get("cameras", cameras) if camera in cameras]
        sequences: Dict[str, Tuple[List[Path], List[Path]]] = {}
        for camera in usable_cameras:
            real, fake = _camera_sequences(scene_sample, camera, cfg)
            sequences[camera] = (real, fake)
        positions = [int(pos) for pos in scene_entry.get("frame_positions", scene_entry.get("frame_indices", []))]
        scene_frame_positions[scene_sample.scene_id] = positions
        scene_clip_starts[scene_sample.scene_id] = {}
        for camera in usable_cameras:
            real, fake = sequences[camera]
            if not real or not fake:
                continue
            for pos in positions:
                if pos >= min(len(real), len(fake)):
                    raise ValueError(f"evaluation_manifest frame position {pos} exceeds scene={scene_id} camera={camera}")
                fake_path = fake[pos]
                real_path = real[pos]
                fake_images.append(
                    ImageSample("fake", scene_sample.scene_id, camera, pos, _frame_id(fake_path, pos), fake_path)
                )
                real_images.append(
                    ImageSample("real", scene_sample.scene_id, camera, pos, _frame_id(real_path, pos), real_path)
                )
            starts = [int(start) for start in scene_entry.get("clip_starts", {}).get(camera, [])]
            scene_clip_starts[scene_sample.scene_id][camera] = starts
            for start in starts:
                if start + clip_length > min(len(real), len(fake)):
                    raise ValueError(f"evaluation_manifest clip start {start} exceeds scene={scene_id} camera={camera}")
                frame_positions = tuple(range(start, start + clip_length))
                real_paths = tuple(real[start : start + clip_length])
                fake_paths = tuple(fake[start : start + clip_length])
                real_clips.append(
                    ClipSample(
                        "real",
                        scene_sample.scene_id,
                        camera,
                        start,
                        frame_positions,
                        tuple(_frame_id(path, pos) for path, pos in zip(real_paths, frame_positions)),
                        real_paths,
                    )
                )
                fake_clips.append(
                    ClipSample(
                        "fake",
                        scene_sample.scene_id,
                        camera,
                        start,
                        frame_positions,
                        tuple(_frame_id(path, pos) for path, pos in zip(fake_paths, frame_positions)),
                        fake_paths,
                    )
                )

    return SamplingPlan(
        seed=seed,
        cameras=cameras,
        frames_per_scene=frames_per_scene,
        clip_length=clip_length,
        stride=stride,
        source_fps=source_fps,
        target_fps=target_fps,
        real_images=_dedupe_images(real_images),
        fake_images=_dedupe_images(fake_images),
        real_clips=_dedupe_clips(real_clips),
        fake_clips=_dedupe_clips(fake_clips),
        scene_frame_positions=scene_frame_positions,
        scene_clip_starts=scene_clip_starts,
        manifest=manifest,
    )

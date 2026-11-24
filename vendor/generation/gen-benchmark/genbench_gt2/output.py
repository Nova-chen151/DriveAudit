"""Safe, transactional publication of GenBench GT2 model predictions."""

import hashlib
import json
import os
import shutil
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .constants import CAMERAS, FUTURE_FRAMES, IMAGE_SIZE, MODEL_SLUGS, SCHEMA_VERSION
from .schema import SceneRef


_RESAMPLING = getattr(Image, "Resampling", Image)
_IMAGE_COUNT = len(CAMERAS) * len(FUTURE_FRAMES)


class OutputSafetyError(ValueError):
    """Raised when an output operation would leave its approved root."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    image_count: int
    reason: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_component(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise OutputSafetyError("invalid {} component".format(name))
    return value


def _assert_contained(root: Path, path: Path) -> Path:
    """Reject paths whose existing symlink components resolve outside *root*."""
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise OutputSafetyError("output path escapes approved prediction root: {}".format(path))
    return Path(path)


def _approved_root(prediction_root: Path, scene: SceneRef) -> Path:
    root = Path(prediction_root).resolve()
    expected = (Path(scene.root).resolve() / "predictions").resolve()
    if root != expected:
        raise OutputSafetyError("prediction root must equal scene.root / predictions")
    _safe_component(scene.dataset, "dataset")
    _safe_component(scene.scene_id, "scene")
    return root


def _model_root(prediction_root: Path, model: str, scene: SceneRef) -> Path:
    if model not in MODEL_SLUGS:
        raise OutputSafetyError("unknown model slug: {}".format(model))
    _safe_component(model, "model")
    root = _approved_root(prediction_root, scene)
    return _assert_contained(root, root / model)


def _atomic_json(path: Path, value: Mapping[str, object], root: Path) -> None:
    _assert_contained(root, path)
    _assert_contained(root, path.parent)
    temporary = path.with_name("{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    _assert_contained(root, temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_names():
    return {"{:06d}.jpg".format(frame) for frame in FUTURE_FRAMES}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_checkpoint(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    digest, files = value.get("digest"), value.get("files")
    if (not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:") or
            any(character not in "0123456789abcdef" for character in digest[7:])):
        return False
    if not isinstance(files, list):
        return False
    for item in files:
        if (not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not item["path"] or
                not _is_int(item.get("size")) or item["size"] < 0 or
                not _is_int(item.get("mtime_ns")) or item["mtime_ns"] < 0):
            return False
    return True


def _required_metadata_error(status: Mapping[str, object]) -> Optional[str]:
    required = (
        "checkpoint", "seed", "inference_settings", "history_frames_used", "repeated_reference_slots",
        "used_future_ego_trajectory", "used_future_agent_trajectories", "available_map_channels",
        "unavailable_map_channels", "temporal_strategy",
    )
    for name in required:
        if name not in status:
            return name
    if not _valid_checkpoint(status["checkpoint"]):
        return "checkpoint"
    if not _is_int(status["seed"]):
        return "seed"
    if not isinstance(status["inference_settings"], Mapping):
        return "inference_settings"
    for name in ("history_frames_used", "repeated_reference_slots"):
        if not isinstance(status[name], list) or any(not _is_int(item) for item in status[name]):
            return name
    for name in ("used_future_ego_trajectory", "used_future_agent_trajectories"):
        if not isinstance(status[name], bool):
            return name
    for name in ("available_map_channels", "unavailable_map_channels"):
        if not isinstance(status[name], list) or any(not isinstance(item, str) for item in status[name]):
            return name
    if not isinstance(status["temporal_strategy"], Mapping):
        return "temporal_strategy"
    return None


def _status_error(status: object, expected_model: str, expected_scene: SceneRef) -> Optional[str]:
    if not isinstance(status, Mapping):
        return "status is not an object"
    metadata_error = _required_metadata_error(status)
    if metadata_error is not None:
        return "status has invalid {}".format(metadata_error)
    if (status.get("schema_version") != SCHEMA_VERSION or status.get("state") != "complete" or
            status.get("model") != expected_model or status.get("dataset") != expected_scene.dataset or
            status.get("scene_id") != expected_scene.scene_id or status.get("input_relative_path") != expected_scene.relative_path or
            status.get("expected_image_count") != _IMAGE_COUNT or status.get("observed_image_count") != _IMAGE_COUNT):
        return "status identity is invalid"
    if (not isinstance(status.get("started_at"), str) or not isinstance(status.get("finished_at"), str) or
            not isinstance(status.get("elapsed_seconds"), (int, float)) or isinstance(status.get("elapsed_seconds"), bool) or
            status["elapsed_seconds"] < 0):
        return "status timing is invalid"
    return None


def _validate_content(path: Path, expected_model: str, expected_scene: SceneRef, root: Path) -> ValidationResult:
    if not path.is_dir():
        return ValidationResult(False, 0, "scene directory does not exist")
    expected_top = {"future", "status.json"}
    try:
        _assert_contained(root, path)
        if {item.name for item in path.iterdir()} != expected_top:
            return ValidationResult(False, 0, "scene has unexpected top-level entries")
        future = path / "future"
        _assert_contained(root, future)
        if not future.is_dir() or {item.name for item in future.iterdir()} != {"rgb"}:
            return ValidationResult(False, 0, "scene has invalid future layout")
        rgb = future / "rgb"
        _assert_contained(root, rgb)
        if not rgb.is_dir() or {item.name for item in rgb.iterdir()} != set(CAMERAS):
            return ValidationResult(False, 0, "scene has invalid camera directories")
        image_count = 0
        for camera in CAMERAS:
            directory = rgb / camera
            _assert_contained(root, directory)
            if not directory.is_dir() or {item.name for item in directory.iterdir()} != _expected_names():
                return ValidationResult(False, image_count, "scene has missing or extra JPEG names")
            for frame in FUTURE_FRAMES:
                image_path = directory / "{:06d}.jpg".format(frame)
                _assert_contained(root, image_path)
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != IMAGE_SIZE:
                        return ValidationResult(False, image_count, "scene has noncanonical image")
                image_count += 1
        status_path = path / "status.json"
        _assert_contained(root, status_path)
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ValidationResult(False, 0, "cannot validate scene: {}".format(exc))
    error = _status_error(status, expected_model, expected_scene)
    if error is not None:
        return ValidationResult(False, image_count, error)
    return ValidationResult(True, image_count)


def validate_completed_scene(path: Path, expected_model: str, expected_scene: SceneRef) -> ValidationResult:
    """Return whether *path* is a complete, resumable scene for this exact identity."""
    try:
        model_root = _model_root(Path(expected_scene.root) / "predictions", expected_model, expected_scene)
        expected_path = model_root / expected_scene.dataset / "scenes" / expected_scene.scene_id
        _assert_contained(model_root.parent, expected_path)
        if Path(path).resolve() != expected_path.resolve():
            return ValidationResult(False, 0, "scene path is not the approved final path")
    except OutputSafetyError as exc:
        return ValidationResult(False, 0, str(exc))
    return _validate_content(Path(path), expected_model, expected_scene, model_root.parent)


def _image_array(image: object, value_range: Optional[str]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    candidate = image
    for method in ("detach", "float", "cpu"):
        operation = getattr(candidate, method, None)
        if callable(operation):
            candidate = operation()
    as_numpy = getattr(candidate, "numpy", None)
    if callable(as_numpy):
        candidate = as_numpy()
    array = np.asarray(candidate)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have HxWx3 shape")
    if np.issubdtype(array.dtype, np.floating):
        if value_range not in ("zero_one", "minus_one_one"):
            raise ValueError("floating image requires an explicit value_range")
        if not np.isfinite(array).all():
            raise ValueError("floating image contains nonfinite values")
        lower, upper = (0.0, 1.0) if value_range == "zero_one" else (-1.0, 1.0)
        if (array < lower - 1e-4).any() or (array > upper + 1e-4).any():
            raise ValueError("floating image is outside declared value_range")
        array = np.clip(array, lower, upper)
        if value_range == "minus_one_one":
            array = (array + 1.0) * 0.5
        return np.rint(array * 255.0).astype(np.uint8)
    if array.dtype != np.uint8:
        raise ValueError("non-floating image arrays must be uint8")
    return array


class ScenePublisher:
    """Stage one scene in a unique work directory and atomically publish it."""

    def __init__(self, prediction_root: Path, model: str, scene: SceneRef, job_id: str, worker_id: str):
        self.scene = scene
        self.model = model
        self.root = _model_root(prediction_root, model, scene)
        self.work_root = self.root / ".work"
        job = _safe_component(str(job_id), "job id")
        worker = _safe_component(str(worker_id), "worker id")
        name = "{}__{}__job-{}__{}__{}".format(scene.dataset, scene.scene_id, job, worker, uuid.uuid4().hex)
        self.work_scene = self.work_root / name
        self.final_scene = self.root / scene.dataset / "scenes" / scene.scene_id
        _assert_contained(self.root.parent, self.work_root)
        _assert_contained(self.root.parent, self.work_scene)
        _assert_contained(self.root.parent, self.final_scene)
        self._started_at = None
        self._entered = False

    def __enter__(self):
        _assert_contained(self.root.parent, self.work_root)
        _assert_contained(self.root.parent, self.work_scene)
        self.work_scene.mkdir(parents=True, exist_ok=False)
        self._started_at = datetime.now(timezone.utc)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _assert_contained(self.root.parent, self.work_scene)
        if self.work_scene.exists() and self.work_scene.parent.resolve() == self.work_root.resolve():
            shutil.rmtree(str(self.work_scene))

    def _require_entered(self) -> None:
        if not self._entered or not self.work_scene.is_dir():
            raise RuntimeError("ScenePublisher must be used inside its context manager")

    def write_frame(self, camera: str, frame_index: int, image: object, value_range: Optional[str] = None) -> Path:
        """Normalize and stage one canonical target image."""
        self._require_entered()
        if camera not in CAMERAS:
            raise ValueError("unknown canonical camera: {}".format(camera))
        if frame_index not in FUTURE_FRAMES:
            raise ValueError("frame_index must be a target frame")
        output = self.work_scene / "future" / "rgb" / camera / "{:06d}.jpg".format(frame_index)
        _assert_contained(self.root.parent, output)
        if output.exists():
            raise ValueError("duplicate staged filename: {}".format(output.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        normalized = _image_array(image, value_range)
        rendered = Image.fromarray(normalized).convert("RGB").resize(IMAGE_SIZE, _RESAMPLING.LANCZOS)
        rendered.save(output, format="JPEG", quality=95, subsampling=0, optimize=False)
        return output

    def write_frames(self, frames: Mapping[str, Mapping[int, object]], value_range: Optional[str] = None) -> None:
        """Write exactly the canonical six-camera by 80-frame prediction grid."""
        if set(frames) != set(CAMERAS):
            raise ValueError("frames must contain exactly the canonical cameras")
        for camera in CAMERAS:
            if set(frames[camera]) != set(FUTURE_FRAMES):
                raise ValueError("frames for {} must contain exactly the target frames".format(camera))
        for camera in CAMERAS:
            for frame in FUTURE_FRAMES:
                self.write_frame(camera, frame, frames[camera][frame], value_range)

    def _archive_invalid_final(self) -> None:
        _assert_contained(self.root.parent, self.final_scene)
        if not self.final_scene.exists():
            return
        if _validate_content(self.final_scene, self.model, self.scene, self.root.parent).valid:
            raise FileExistsError("a valid completed scene already exists: {}".format(self.final_scene))
        archived = self.root / "_failures" / self.scene.dataset / "{}.incomplete.{}".format(self.scene.scene_id, _timestamp())
        _assert_contained(self.root.parent, archived)
        archived.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(self.final_scene), str(archived))

    def _resolve_failure(self) -> None:
        failure = self.root / "_failures" / self.scene.dataset / "{}.json".format(self.scene.scene_id)
        _assert_contained(self.root.parent, failure)
        if not failure.exists():
            return
        resolved = self.root / "_failures" / "_resolved" / self.scene.dataset / "{}.{}.json".format(self.scene.scene_id, _timestamp())
        _assert_contained(self.root.parent, resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(failure), str(resolved))

    def publish(self, metadata: Mapping[str, object]) -> Path:
        """Validate the staged scene, write status, and atomically replace the final path."""
        self._require_entered()
        _assert_contained(self.root.parent, self.work_scene)
        _assert_contained(self.root.parent, self.final_scene)
        metadata_error = _required_metadata_error(metadata)
        if metadata_error is not None:
            raise ValueError("missing or invalid required status metadata: {}".format(metadata_error))
        preliminary = _validate_content_without_status(self.work_scene, self.root.parent)
        if not preliminary.valid:
            raise ValueError("cannot publish incomplete scene: {}".format(preliminary.reason))
        finished_at = datetime.now(timezone.utc)
        started_at = self._started_at or finished_at
        status = dict(metadata)
        status.update({
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "model": self.model,
            "input_relative_path": self.scene.relative_path,
            "dataset": self.scene.dataset,
            "scene_id": self.scene.scene_id,
            "expected_image_count": _IMAGE_COUNT,
            "observed_image_count": preliminary.image_count,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
        })
        _atomic_json(self.work_scene / "status.json", status, self.root.parent)
        validated = _validate_content(self.work_scene, self.model, self.scene, self.root.parent)
        if not validated.valid:
            raise ValueError("cannot publish invalid scene: {}".format(validated.reason))
        self._archive_invalid_final()
        self.final_scene.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(self.work_scene), str(self.final_scene))
        self._resolve_failure()
        return self.final_scene


def _validate_content_without_status(path: Path, root: Path) -> ValidationResult:
    try:
        _assert_contained(root, path)
        if {item.name for item in path.iterdir()} != {"future"}:
            return ValidationResult(False, 0, "scene has unexpected staged entries")
        future = path / "future"
        rgb = future / "rgb"
        _assert_contained(root, future)
        if not future.is_dir() or {item.name for item in future.iterdir()} != {"rgb"}:
            return ValidationResult(False, 0, "scene has invalid future layout")
        _assert_contained(root, rgb)
        if not rgb.is_dir() or {item.name for item in rgb.iterdir()} != set(CAMERAS):
            return ValidationResult(False, 0, "scene has invalid camera directories")
        count = 0
        for camera in CAMERAS:
            directory = rgb / camera
            _assert_contained(root, directory)
            if not directory.is_dir() or {item.name for item in directory.iterdir()} != _expected_names():
                return ValidationResult(False, count, "scene has missing or extra JPEG names")
            for frame in FUTURE_FRAMES:
                image_path = directory / "{:06d}.jpg".format(frame)
                _assert_contained(root, image_path)
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != IMAGE_SIZE:
                        return ValidationResult(False, count, "scene has noncanonical image")
                count += 1
    except (OSError, ValueError):
        return ValidationResult(False, 0, "cannot validate staged images")
    return ValidationResult(count == _IMAGE_COUNT, count, "" if count == _IMAGE_COUNT else "wrong image count")


def checkpoint_identity(paths: Sequence[Path]) -> Mapping[str, object]:
    """Return a reproducible checkpoint identity from resolved file metadata."""
    records = []
    for path in paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        records.append({"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    records.sort(key=lambda record: record["path"])
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"digest": "sha256:" + hashlib.sha256(encoded).hexdigest(), "files": records}


def write_failure(prediction_root: Path, model: str, scene: SceneRef, phase: str, error: BaseException, rank: int, job_id: str) -> Path:
    """Persist an atomic diagnostic record for a failed scene attempt."""
    root = _model_root(prediction_root, model, scene)
    target = root / "_failures" / scene.dataset / "{}.json".format(scene.scene_id)
    _assert_contained(root.parent, target)
    record = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "dataset": scene.dataset,
        "scene_id": scene.scene_id,
        "input_relative_path": scene.relative_path,
        "phase": str(phase),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "rank": rank,
        "job_id": str(job_id),
        "timestamp": _utc_now(),
    }
    _atomic_json(target, record, root.parent)
    return target

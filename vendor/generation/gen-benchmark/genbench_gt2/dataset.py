"""Strict, RGB-safe loading for the GenBench GT2 prediction contract."""

import csv
import json
import logging
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .constants import CAMERAS, FUTURE_FRAMES, HISTORY_FRAMES, IMAGE_SIZE, MAP_CHANNELS
from .schema import AgentBox, AuditReport, CameraCalibration, MapCondition, SceneInputs, SceneRef


class DataBoundaryError(ValueError):
    pass


_PACKAGE_DATASETS = {"nuscenes300_challenger_v1": "nuscenes", "av2_challenger_v1": "av2", "nuplan_challenger_v1": "nuplan", "pandaset_challenger_v1": "pandaset"}
_POSE_FRAMES = ("history_frame_9_ego", "history_last_ego")
_VECTOR_LABELS = ("divider", "ped_crossing", "boundary", "drivable_area")


def _fail(message: str) -> None:
    raise DataBoundaryError(message)


def _json(path: Path) -> Mapping[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail("cannot read {}: {}".format(path, exc))
    if not isinstance(value, dict):
        _fail("{} must be an object".format(path.name))
    return value


def _items(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        _fail("{} must be a list".format(name))
    return value


def _num(value: object, name: str) -> float:
    if isinstance(value, bool):
        _fail("{} must be numeric".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail("{} must be numeric".format(name))
    if not np.isfinite(result):
        _fail("{} must be finite".format(name))
    return result


def _matrix(value: object, shape: Tuple[int, int], name: str, invertible: bool = False, homogeneous: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail("{} must be numeric".format(name))
    if result.shape != shape or not np.isfinite(result).all():
        _fail("{} must be a finite {}x{} matrix".format(name, *shape))
    if invertible:
        try:
            if abs(np.linalg.det(result)) < 1e-12:
                _fail("{} must be invertible".format(name))
        except np.linalg.LinAlgError:
            _fail("{} must be invertible".format(name))
    if homogeneous and not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        _fail("{} must have homogeneous last row".format(name))
    return result


def _dataset_slug(relative_path: str) -> str:
    prefix = relative_path.split("/", 1)[0]
    if prefix not in _PACKAGE_DATASETS:
        raise ValueError("unsupported GenBench package: {}".format(prefix))
    return _PACKAGE_DATASETS[prefix]


def history_rgb_path(scene: SceneRef, camera: str, frame_index: int) -> Path:
    if camera not in CAMERAS:
        raise ValueError("unknown canonical camera: {}".format(camera))
    if frame_index not in HISTORY_FRAMES:
        raise DataBoundaryError("future RGB is evaluation-only; requested frame {}".format(frame_index))
    asset_root = scene.asset_root or scene.root
    return asset_root / scene.relative_path / "history" / "rgb" / camera / "{:06d}.jpg".format(frame_index)


def _root_layout(root: Path) -> Tuple[Path, Path, Path]:
    """Return the scene root, static asset root, and trajectory root.

    CAT-K is distributed as a sparse overlay containing only replacement
    trajectories.  Its sibling GenBench root retains the static assets and
    generation-set metadata needed by the video generators.
    """
    marker = root / "CATK_INFERENCE_MANIFEST.csv"
    if marker.is_file():
        # CAT-K is a sparse trajectory overlay.  Generation may use a sibling
        # static tree explicitly, so that camera calibration, history and ego
        # future controls remain from input-gt.
        asset_root = Path(os.environ.get("GENBENCH_STATIC_ASSET_ROOT", root.parent)).resolve()
        return root, asset_root, root
    return root, root, root


def load_generation_set(root: Path, set_name: str) -> List[SceneRef]:
    root = Path(root).resolve()
    scene_root, asset_root, trajectory_root = _root_layout(root)
    # The explicit agent-only contract takes all scene identity/static inputs
    # from input-gt; overlay metadata must never change the selection mapping.
    metadata_root = asset_root if os.environ.get("GENBENCH_CATK_AGENT_ONLY") == "1" else scene_root
    metadata_path = metadata_root / "metadata" / "generation_sets.json"
    if not metadata_path.is_file():
        metadata_path = asset_root / "metadata" / "generation_sets.json"
    document = _json(metadata_path)
    if set_name not in document:
        _fail("unknown generation set: {}".format(set_name))
    scenes, paths, identities = [], set(), set()
    for index, entry in enumerate(_items(document[set_name], "generation set")):
        if not isinstance(entry, str):
            _fail("generation-set entry {} must be a string".format(index))
        path = PurePosixPath(entry)
        if path.is_absolute() or "." in path.parts or ".." in path.parts or len(path.parts) != 3 or path.parts[1] != "scenes" or not all(path.parts):
            _fail("invalid generation-set path: {}".format(entry))
        relative = path.as_posix()
        dataset = _dataset_slug(relative)
        resolved = (scene_root / relative).resolve()
        try:
            resolved.relative_to(scene_root)
        except ValueError:
            _fail("generation-set path escapes root: {}".format(entry))
        identity = (dataset, path.parts[2])
        if resolved in paths or identity in identities:
            _fail("duplicate generation-set scene: {}".format(entry))
        if not resolved.is_dir():
            _fail("scene directory does not exist: {}".format(resolved))
        paths.add(resolved)
        identities.add(identity)
        static_scene = (asset_root / relative).resolve()
        if not static_scene.is_dir():
            _fail("static scene directory does not exist: {}".format(static_scene))
        scenes.append(SceneRef(index, relative, dataset, path.parts[2], scene_root, asset_root, trajectory_root))
    return scenes


def _calibration(scene_dir: Path) -> Tuple[Tuple[str, ...], Mapping[str, CameraCalibration]]:
    data = _json(scene_dir / "calibration" / "cameras.json")
    order = tuple(_items(data.get("camera_order"), "camera_order"))
    if order != CAMERAS:
        _fail("camera_order must equal canonical CAMERAS")
    rows = _items(data.get("cameras"), "cameras")
    if len(rows) != len(CAMERAS):
        _fail("cameras must contain six rows")
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail("camera calibration must be an object")
        camera = row.get("camera")
        if camera not in CAMERAS or camera in output:
            _fail("cameras must contain each canonical camera once")
        if row.get("challenge_image_size") != list(IMAGE_SIZE):
            _fail("{} has noncanonical challenge_image_size".format(camera))
        output[camera] = CameraCalibration(camera, _matrix(row.get("challenge_intrinsics"), (3, 3), "intrinsics"), _matrix(row.get("camera_to_ego_4x4"), (4, 4), "camera_to_ego", invertible=True))
    if tuple(output) != CAMERAS:
        _fail("cameras must be listed in canonical order")
    return order, output


def _poses(scene_dir: Path) -> np.ndarray:
    data = _json(scene_dir / "calibration" / "reference_ego_poses.json")
    if data.get("coordinate_frame") != "history_frame_9_ego":
        _fail("reference ego poses must use history_frame_9_ego")
    rows = _items(data.get("ego_to_anchor"), "ego_to_anchor")
    if len(rows) != 90:
        _fail("ego_to_anchor must contain 90 matrices")
    return np.stack([_matrix(row, (4, 4), "ego_to_anchor", invertible=True) for row in rows])


def _history_poses(scene_dir: Path) -> None:
    data = _json(scene_dir / "calibration" / "history_camera_poses.json")
    if data.get("coordinate_frame") not in _POSE_FRAMES:
        _fail("history camera poses have unsupported coordinate frame")
    seen = set()
    for row in _items(data.get("rows"), "history camera poses"):
        if not isinstance(row, dict):
            _fail("history camera pose must be an object")
        frame, camera = row.get("frame_index"), row.get("camera")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame not in range(90) or camera not in CAMERAS:
            _fail("history camera pose has invalid frame or camera")
        if (frame, camera) in seen:
            _fail("duplicate history camera pose")
        seen.add((frame, camera))
        _matrix(row.get("camera_to_anchor_4x4"), (4, 4), "camera_to_anchor", invertible=True, homogeneous=True)
    if not {(f, c) for f in HISTORY_FRAMES for c in CAMERAS}.issubset(seen):
        _fail("history camera poses must cover all history frame-camera pairs")


def _timeline(scene_dir: Path) -> np.ndarray:
    rows = _items(_json(scene_dir / "timeline.json").get("frames"), "timeline frames")
    if len(rows) != 90:
        _fail("timeline must contain 90 frames")
    result = []
    for frame, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("frame_index") != frame:
            _fail("timeline frame indices must be 0 through 89")
        result.append(_num(row.get("reference_time_us"), "reference_time_us"))
        cameras = row.get("cameras")
        if not isinstance(cameras, dict):
            _fail("timeline cameras must be an object")
        if frame in HISTORY_FRAMES and set(cameras) != set(CAMERAS):
            _fail("history timeline frames must contain all canonical cameras")
        if frame in FUTURE_FRAMES and cameras and set(cameras) != set(CAMERAS):
            _fail("future timeline frames must contain either zero or all canonical cameras")
        for camera in CAMERAS:
            if camera in cameras:
                item = cameras[camera]
                if not isinstance(item, dict):
                    _fail("timeline camera entry must be an object")
                _num(item.get("exposure_time_us"), "exposure_time_us")
                _num(item.get("offset_from_reference_us"), "offset_from_reference_us")
    result = np.asarray(result)
    if not np.all(np.diff(result) > 0):
        _fail("timeline reference times must be strictly increasing")
    return result


def _csv(path: Path, fields: Iterable[str]) -> List[Mapping[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not set(fields).issubset(reader.fieldnames):
                _fail("{} has invalid CSV headers".format(path))
            return list(reader)
    except OSError as exc:
        _fail("cannot read {}: {}".format(path, exc))


def _integer(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail("{} must be an integer".format(name))


def _valid(row: Mapping[str, str], name: str) -> bool:
    value = row.get("valid")
    if value not in ("True", "False", "true", "false", "1", "0"):
        _fail("{} has invalid valid flag".format(name))
    return value in ("True", "true", "1")


def _trajectory(path: Path, expected: Sequence[int], name: str) -> np.ndarray:
    output = {}
    for row in _csv(path, ("frame_index", "x", "y", "yaw", "valid")):
        if not _valid(row, name):
            continue
        frame = _integer(row.get("frame_index"), "frame_index")
        if frame not in expected or frame in output:
            _fail("{} must contain one valid row for every expected frame".format(name))
        output[frame] = [_num(row.get(key), "{}.{}".format(name, key)) for key in ("x", "y", "yaw")]
    if tuple(sorted(output)) != tuple(expected):
        _fail("{} is missing valid frames".format(name))
    return np.asarray([output[frame] for frame in expected], dtype=np.float64)


def _agents(scene_dir: Path, future_path: Optional[Path] = None, strict_future: bool = False) -> Mapping[int, Tuple[AgentBox, ...]]:
    fields = ("frame_index", "agent_id", "category", "x", "y", "z", "width", "length", "height", "yaw", "valid")
    groups = defaultdict(list)
    future_path = future_path or (scene_dir / "future" / "future_trajectories.csv")
    future_rows = _csv(future_path, fields)
    if strict_future:
        for row in future_rows:
            if _integer(row.get("frame_index"), "agent frame_index") not in FUTURE_FRAMES:
                _fail("future agent overlay contains a non-future frame")
    for row in _csv(scene_dir / "history" / "agents.csv", fields) + future_rows:
        if not _valid(row, "agent"):
            continue
        frame = _integer(row.get("frame_index"), "agent frame_index")
        agent_id, category = row.get("agent_id"), row.get("category")
        if frame not in range(90) or not agent_id or not category:
            _fail("agent row has invalid identity or frame")
        center = np.asarray([_num(row.get(k), "agent {}".format(k)) for k in ("x", "y", "z")])
        size = np.asarray([_num(row.get(k), "agent {}".format(k)) for k in ("width", "length", "height")])
        if not np.all(size > 0):
            _fail("agent dimensions must be positive")
        groups[frame].append(AgentBox(frame, agent_id, category, center, size, _num(row.get("yaw"), "agent yaw")))
    return {frame: tuple(groups[frame]) for frame in range(90)}


def _map(scene_dir: Path) -> MapCondition:
    try:
        with np.load(scene_dir / "map" / "semantic_map.npz", allow_pickle=False) as archive:
            if not {"mask", "ground_height_m"}.issubset(archive.files):
                _fail("semantic map archive is missing required arrays")
            mask, height = archive["mask"], archive["ground_height_m"]
    except DataBoundaryError:
        raise
    except (OSError, ValueError) as exc:
        _fail("cannot read semantic_map.npz: {}".format(exc))
    if mask.shape != (8, 1024, 1024) or mask.dtype != np.uint8 or not np.logical_or(mask == 0, mask == 1).all():
        _fail("semantic mask must be binary uint8 8x1024x1024")
    if height.shape != (1024, 1024) or not np.issubdtype(height.dtype, np.number) or np.isinf(height).any():
        _fail("ground_height_m must be 1024x1024 without infinite values")
    meta = _json(scene_dir / "map" / "map_meta.json")
    if tuple(_items(meta.get("channels"), "map channels")) != MAP_CHANNELS:
        _fail("map channels must equal MAP_CHANNELS in order")
    if meta.get("array_axes") != "channel,x_forward,y_left" or meta.get("coordinate_frame") != "history_last_ego_x_forward_y_left_meters":
        _fail("map metadata has invalid coordinate semantics")
    available, unavailable = tuple(_items(meta.get("available_channels"), "available channels")), tuple(_items(meta.get("unavailable_channels"), "unavailable channels"))
    if any(not isinstance(item, str) for item in available + unavailable) or set(available) & set(unavailable) or set(available) | set(unavailable) != set(MAP_CHANNELS) or any(item not in MAP_CHANNELS for item in available + unavailable):
        _fail("map availability sets must be disjoint and exhaustive")
    for channel in unavailable:
        if np.any(mask[MAP_CHANNELS.index(channel)]):
            _fail("unavailable map channel must be all zero: {}".format(channel))
    extent = _items(meta.get("extent_m"), "extent_m")
    if len(extent) != 4:
        _fail("extent_m must contain four values")
    resolution = _num(meta.get("resolution_m"), "resolution_m")
    if resolution <= 0:
        _fail("resolution_m must be positive")
    vectors = _json(scene_dir / "map" / "generation_vectors.json")
    classes = _items(vectors.get("classes"), "vector classes")
    if not classes or any(not isinstance(item, str) or item not in _VECTOR_LABELS for item in classes) or len(set(classes)) != len(classes):
        _fail("vector classes must be unique known labels")
    for line in _items(vectors.get("lines"), "vector lines"):
        if not isinstance(line, dict):
            _fail("vector line must be an object")
        label = line.get("label")
        if isinstance(label, int) and not isinstance(label, bool):
            if label < 0 or label >= len(classes):
                _fail("vector integer label is outside classes")
        elif label not in classes:
            _fail("vector label is unknown")
        for point in _items(line.get("points"), "vector points"):
            point = _items(point, "vector point")
            if len(point) != 3:
                _fail("vector points must be 3D")
            for value in point:
                _num(value, "vector coordinate")
    return MapCondition(mask, height, tuple(_num(value, "extent_m") for value in extent), resolution, available, unavailable, vectors)


def _history_rgb(scene: SceneRef) -> Mapping[str, Tuple[Path, ...]]:
    result = {}
    for camera in CAMERAS:
        paths = []
        for frame in HISTORY_FRAMES:
            path = history_rgb_path(scene, camera, frame)
            try:
                with Image.open(path) as image:
                    if image.size != IMAGE_SIZE:
                        _fail("history RGB has incorrect image size: {}".format(path))
                    image.convert("RGB")
            except (OSError, ValueError) as exc:
                _fail("cannot decode history RGB {}: {}".format(path, exc))
            paths.append(path)
        result[camera] = tuple(paths)
    return result


def _catk_ego_trajectory(path: Path) -> np.ndarray:
    data = _json(path)
    if data.get("coordinate_frame") != "history_frame_9_ego":
        _fail("CAT-K ego prediction must use history_frame_9_ego")
    positions = _items(data.get("position"), "CAT-K ego positions")
    headings = _items(data.get("heading"), "CAT-K ego headings")
    valid = _items(data.get("valid"), "CAT-K ego valid flags")
    expected_count = len(FUTURE_FRAMES)
    if len(positions) != expected_count or len(headings) != expected_count or len(valid) != expected_count:
        _fail("CAT-K ego prediction must contain exactly {} future frames".format(expected_count))
    result = []
    for index, (position, heading, is_valid) in enumerate(zip(positions, headings, valid)):
        if is_valid is not True:
            _fail("CAT-K ego prediction has an invalid future frame: {}".format(FUTURE_FRAMES[index]))
        values = _items(position, "CAT-K ego position")
        if len(values) < 2:
            _fail("CAT-K ego position must contain x and y")
        result.append([_num(values[0], "CAT-K ego x"), _num(values[1], "CAT-K ego y"), _num(heading, "CAT-K ego heading")])
    return np.asarray(result, dtype=np.float64)


def load_scene_inputs(scene: SceneRef) -> SceneInputs:
    asset_root = scene.asset_root or scene.root
    trajectory_root = scene.trajectory_root or scene.root
    directory = (asset_root / scene.relative_path).resolve()
    try:
        directory.relative_to(asset_root.resolve())
    except ValueError:
        _fail("scene path escapes root")
    generation, context = _json(directory / "generation_input.json"), _json(directory / "scene_context.json")
    if generation.get("reference_frame") != 9 or generation.get("target_frames") != [0, 89] or generation.get("future_rgb") is not False:
        _fail("generation_input.json violates the future-RGB boundary")
    prompt = context.get("generation_prompt")
    if context.get("dataset") != scene.dataset or not isinstance(prompt, str) or not prompt.strip():
        _fail("scene context has wrong dataset or empty generation_prompt")
    order, calibrations = _calibration(directory)
    ego = _poses(directory)
    _history_poses(directory)
    timeline = _timeline(directory)
    _trajectory(directory / "history" / "ego.csv", HISTORY_FRAMES, "history ego")
    trajectory_directory = (trajectory_root / scene.relative_path).resolve()
    agent_only = os.environ.get("GENBENCH_CATK_AGENT_ONLY") == "1" and trajectory_root.resolve() != asset_root.resolve()
    if agent_only and os.environ.get("GENBENCH_USE_CATK_EGO_PREDICTION") != "0":
        _fail("CATK agent-only mode requires GENBENCH_USE_CATK_EGO_PREDICTION=0")
    catk_ego_path = trajectory_directory / "catk_ego_prediction.json"
    if os.environ.get("GENBENCH_USE_CATK_EGO_PREDICTION", "1") != "0" and catk_ego_path.is_file():
        future_ego = _catk_ego_trajectory(catk_ego_path)
    else:
        future_ego = _trajectory(directory / "future" / "ego_trajectory.csv", FUTURE_FRAMES, "future ego trajectory")
    future_agents_path = trajectory_directory / "future" / "future_trajectories.csv"
    if not future_agents_path.is_file():
        if agent_only:
            _fail("required CATK future agent overlay is missing: {}".format(future_agents_path))
        future_agents_path = directory / "future" / "future_trajectories.csv"
    return SceneInputs(scene, order, _history_rgb(scene), calibrations, ego, future_ego, _agents(directory, future_agents_path, strict_future=agent_only), _map(directory), prompt, timeline)


def audit_generation_set(root: Path, set_name: str, audit_reference_layout: bool = True) -> AuditReport:
    scenes = load_generation_set(root, set_name)
    def audit_scene(scene):
        errors, history_count, reference_count = [], 0, 0
        try:
            history_count = sum(len(paths) for paths in load_scene_inputs(scene).history_rgb.values())
        except DataBoundaryError as exc:
            errors.append("{}: {}".format(scene.relative_path, exc))
        if audit_reference_layout:
            asset_root = scene.asset_root or scene.root
            for camera in CAMERAS:
                for frame in FUTURE_FRAMES:
                    path = asset_root / scene.relative_path / "future" / "rgb" / camera / "{:06d}.jpg".format(frame)
                    try:
                        info = path.stat()
                        if stat.S_ISREG(info.st_mode) and info.st_size > 0:
                            reference_count += 1
                        else:
                            errors.append("{}: missing or empty evaluation reference {}".format(scene.relative_path, path))
                    except FileNotFoundError:
                        errors.append("{}: missing or empty evaluation reference {}".format(scene.relative_path, path))
                    except OSError as exc:
                        errors.append("{}: cannot stat evaluation reference {}: {}".format(scene.relative_path, path, exc))
        return history_count, reference_count, errors
    errors, history_count, reference_count = [], 0, 0
    # Network metadata latency dominates the audit. Bound concurrency and retain
    # generation-set order for deterministic reports and error lists.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for completed, result in enumerate(executor.map(audit_scene, scenes), 1):
            history_count += result[0]
            reference_count += result[1]
            errors.extend(result[2])
            if completed % 25 == 0 or completed == len(scenes):
                logging.getLogger(__name__).info("Input audit: %d/%d scenes, %d errors", completed, len(scenes), len(errors))
    return AuditReport(len(scenes), history_count, reference_count, dict(Counter(scene.dataset for scene in scenes)), tuple(errors))

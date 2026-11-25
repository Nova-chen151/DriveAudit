"""Geometry and RGB helpers used by the participant MagicDrive-V2 adapter.

Selected from the native challenge_generation/infer.py in the 2026-11-23 snapshot.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__:
    from .v2_contract import crop_map as crop_v2_map
else:
    from v2_contract import crop_map as crop_v2_map


REFERENCE_HISTORY_FRAMES = tuple(range(1, 10))


FUTURE_FRAMES = tuple(range(10, 90))


RENDER_FRAMES = tuple(range(1, 242))


RENDER_START_FRAME = RENDER_FRAMES[0]


SUBMISSION_START_INDEX = FUTURE_FRAMES[0] - RENDER_START_FRAME


SUBMISSION_END_INDEX = FUTURE_FRAMES[-1] - RENDER_START_FRAME + 1


CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)


RENDERER_TO_EGO_ROTATION = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


CATEGORY_TO_CLASS = {
    "car": 0,
    "truck": 1,
    "construction_vehicle": 2,
    "bus": 3,
    "trailer": 4,
    "barrier": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
    # Static nuScenes objects outside the ten renderer classes are exposed by
    # the challenge as ``other``; barrier is the closest visual token.
    "other": 5,
}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _quaternion_matrix(wxyz: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(wxyz), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"Invalid WXYZ quaternion: {q}")
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _box_corners(center: np.ndarray, width: float, length: float, height: float, yaw: float) -> np.ndarray:
    # The native converter stores [width, length, height, -yaw-pi/2], then
    # LiDARInstance3DBoxes rotates clockwise. Preserve its ORDERED corners:
    # the box encoder flattens all eight and is not permutation invariant.
    # Native training also treats the supplied center z as bottom z; retain
    # that checkpoint convention instead of applying a physical h/2 shift.
    local = np.asarray(
        [
            [length / 2, -width / 2, 0.0],
            [length / 2, -width / 2, height],
            [-length / 2, -width / 2, height],
            [-length / 2, -width / 2, 0.0],
            [length / 2, width / 2, 0.0],
            [length / 2, width / 2, height],
            [-length / 2, width / 2, height],
            [-length / 2, width / 2, 0.0],
        ],
        dtype=np.float64,
    )
    local[:, :2] = local[:, :2] @ _rotation(yaw).T
    return local + center[None]


def _renderer_origin(calibration: dict[str, Any]) -> np.ndarray:
    """Use the published LIDAR_TOP origin in v2; retain the v1 estimate."""
    renderer = calibration.get("renderer")
    if renderer is not None:
        origin = np.asarray(renderer["sensor2ego_translation_m"], dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid v2 renderer origin")
        return origin
    entries = {entry["camera"]: entry for entry in calibration["cameras"]}
    front = np.asarray(entries["CAM_FRONT"]["sensor2ego_translation_m"], dtype=np.float64)
    back = np.asarray(entries["CAM_BACK"]["sensor2ego_translation_m"], dtype=np.float64)
    camera_heights = [
        float(entry["sensor2ego_translation_m"][2])
        for entry in calibration["cameras"]
    ]
    origin = 0.5 * (front + back)
    origin[2] = max(camera_heights) + 0.25
    return origin


def _renderer_to_ego_rotation(calibration: dict[str, Any]) -> np.ndarray:
    renderer = calibration.get("renderer")
    if renderer is None:
        return RENDERER_TO_EGO_ROTATION
    return _quaternion_matrix(renderer["sensor2ego_rotation_wxyz"])


def _planar_rotation(rotation: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return _rotation(yaw)


def _camera_matrices(
    calibration: dict[str, Any], renderer_origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera_params = []
    renderer_to_cameras = []
    entries = {entry["camera"]: entry for entry in calibration["cameras"]}
    if calibration.get("renderer") is not None:
        for camera in CAMERAS:
            entry = entries[camera]
            camera_to_renderer = np.asarray(
                entry["camera_to_renderer_4x4"], dtype=np.float64
            )
            if camera_to_renderer.shape != (4, 4) or not np.isfinite(camera_to_renderer).all():
                raise ValueError(f"Invalid exact camera transform for {camera}")
            # nuScenes keeps the native training K. Other GT2 datasets may
            # provide the GPU-smoke-supported cross-dataset model override;
            # this changes only camera embedding, not physical projection.
            intrinsic = np.asarray(
                entry.get("model_intrinsics", entry["native_intrinsics"]),
                dtype=np.float64,
            )
            camera_params.append(
                np.concatenate([intrinsic, camera_to_renderer[:3]], axis=1)
            )
            renderer_to_cameras.append(np.linalg.inv(camera_to_renderer))
        return np.asarray(camera_params), np.asarray(renderer_to_cameras)

    renderer_to_ego = np.eye(4, dtype=np.float64)
    renderer_to_ego[:3, :3] = RENDERER_TO_EGO_ROTATION
    renderer_to_ego[:3, 3] = renderer_origin
    ego_to_renderer = np.linalg.inv(renderer_to_ego)
    for camera in CAMERAS:
        entry = entries[camera]
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = _quaternion_matrix(entry["sensor2ego_rotation_wxyz"])
        camera_to_ego[:3, 3] = np.asarray(entry["sensor2ego_translation_m"], dtype=np.float64)
        camera_to_renderer = ego_to_renderer @ camera_to_ego
        # Honor the explicit compatibility override; otherwise retain native K.
        # Image projection uses challenge_intrinsics separately; visibility
        # below depends only on camera-space depth, not either intrinsic matrix.
        intrinsic = np.asarray(
            entry.get("model_intrinsics", entry["native_intrinsics"]),
            dtype=np.float64,
        )
        camera_params.append(np.concatenate([intrinsic, camera_to_renderer[:3]], axis=1))
        renderer_to_cameras.append(np.linalg.inv(camera_to_renderer))
    return np.asarray(camera_params), np.asarray(renderer_to_cameras)


def _warp_maps(
    semantic_mask: np.ndarray,
    ego_positions: np.ndarray,
    ego_yaws: np.ndarray,
    renderer_origin: np.ndarray,
    map_metadata=None,
    renderer_to_ego_rotation: np.ndarray = RENDERER_TO_EGO_ROTATION,
) -> np.ndarray:
    if map_metadata is not None:
        local_to_ego = _planar_rotation(renderer_to_ego_rotation)
        return np.stack(
            [
                crop_v2_map(
                    semantic_mask,
                    map_metadata,
                    position,
                    float(yaw),
                    renderer_origin,
                    local_to_ego,
                )
                for position, yaw in zip(ego_positions, ego_yaws)
            ],
            axis=0,
        )
    # MagicDriveDiT expects: drivable, crossing, walkway, stop, carpark,
    # road divider, lane divider, road block. input-op provides all but road block.
    bits = (0, 3, 4, 5, 6, 7, 8)
    source = np.stack([((semantic_mask >> bit) & 1).astype(np.float32) for bit in bits], axis=0)
    # NuScenesMultiResDataset feeds a 200x200, 0.5 m/pixel raster into the
    # renderer.  MapControlEmbedding then brings it to the image-latent grid.
    # Supplying a pre-upsampled 400x400 map makes the model interpolate a
    # mismatched 53x100 feature grid and severely degrades generation.
    axis = (np.arange(200, dtype=np.float64) - 100.0) * 0.5
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    local_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    warped = np.zeros((len(RENDER_FRAMES), 8, 200, 200), dtype=np.float32)
    for frame_idx, (position, yaw) in enumerate(zip(ego_positions, ego_yaws)):
        ego_local_points = (
            local_points @ renderer_to_ego_rotation[:2, :2].T
            + renderer_origin[None, :2]
        )
        reference_points = ego_local_points @ _rotation(float(yaw)).T + position
        ix = np.rint(reference_points[:, 0] / 0.5 + 100.0).astype(np.int64)
        iy = np.rint(reference_points[:, 1] / 0.5 + 100.0).astype(np.int64)
        # input-op has no future global map. Keep the map control dense and
        # temporally continuous with nearest-border extension; zero-filling the
        # unknown region makes the map condition collapse as Ego leaves the
        # 100 m crop and causes block artifacts across the full video.
        ix = np.clip(ix, 0, 199)
        iy = np.clip(iy, 0, 199)
        for out_channel, src_channel in enumerate(range(7)):
            flat = warped[frame_idx, out_channel].reshape(-1)
            flat[:] = source[src_channel, ix, iy]
    return warped


def _infer_prompt(scene_dir: Path) -> str:
    from PIL import Image

    context_path = scene_dir / "scene_context.json"
    if context_path.is_file():
        with context_path.open("r", encoding="utf-8") as handle:
            context = json.load(handle)
        if "location" not in context:
            return context["generation_prompt"]
        location = str(context["location"])
        description = " ".join(str(context.get("description", "")).split())
        return f"A driving scene image at {location}. {description}."

    luminance = []
    for camera in CAMERAS:
        with Image.open(scene_dir / "history" / "rgb" / camera / "000009.jpg") as image:
            luminance.append(float(np.asarray(image.convert("L").resize((32, 18)), dtype=np.float32).mean()))
    location = os.environ.get("CHALLENGE_LOCATION_PROMPT", "").strip().lower()
    allowed_locations = {
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage",
    }
    if location and location not in allowed_locations:
        raise ValueError(
            "CHALLENGE_LOCATION_PROMPT must be one of "
            f"{sorted(allowed_locations)}, got {location!r}"
        )
    is_night = float(np.mean(luminance)) < 62.0
    if not location:
        location = "singapore-hollandvillage" if is_night else "singapore-onenorth"
    prefix = f"A driving scene image at {location}."
    if is_night:
        return f"{prefix} Night, urban traffic."
    return f"{prefix} Daytime, urban traffic."


def _load_reference_images(scene_dir: Path) -> np.ndarray:
    from PIL import Image

    camera_sequences = []
    for camera in CAMERAS:
        frames = []
        for frame_index in REFERENCE_HISTORY_FRAMES:
            path = scene_dir / "history" / "rgb" / camera / f"{frame_index:06d}.jpg"
            with Image.open(path) as image:
                if image.size != (400, 224):
                    raise ValueError(f"Unexpected image size {image.size}: {path}")
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            frames.append(np.transpose(array, (2, 0, 1)))
        camera_sequences.append(np.stack(frames, axis=1))
    # B, NC, C, T, H, W
    return np.stack(camera_sequences, axis=0)[None]


def _atomic_pickle(value: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_path)


def _normalise_samples(samples: Any) -> Any:
    import torch

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor renderer result, got {type(samples)}")
    samples = samples.detach().to(device="cpu", dtype=torch.float32)
    if samples.ndim == 6:
        if samples.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(samples.shape)}")
        samples = samples[0]
    if samples.ndim != 5 or tuple(samples.shape[:2]) != (6, 3):
        raise ValueError(f"Expected samples shaped [6,3,T,H,W], got {tuple(samples.shape)}")
    if samples.shape[2] < SUBMISSION_END_INDEX:
        raise ValueError(
            f"Renderer returned {samples.shape[2]} frames; need at least {SUBMISSION_END_INDEX}"
        )
    if tuple(samples.shape[-2:]) != (224, 400):
        raise ValueError(f"Renderer returned wrong image size: {tuple(samples.shape[-2:])}")
    return samples[:, :, SUBMISSION_START_INDEX:SUBMISSION_END_INDEX]

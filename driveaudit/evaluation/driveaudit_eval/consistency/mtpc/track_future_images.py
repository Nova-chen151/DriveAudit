from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.mtpc.common import CAMERAS, write_json
from driveaudit_eval.video.utils import read_image_bgr


def track_future_images(backend: Any, future_image_root: str | Path, calibrations: list[dict[str, Any]],
                        ego_poses: list[dict[str, Any]], timestamps: list[float], frame_start: int,
                        frame_end: int, output: str | Path) -> dict[str, Any]:
    """Stage A. Signature intentionally has no submitted/future reference argument."""
    output = Path(output); by_frame = {int(row["frame_index"]): [] for row in ego_poses}
    for row in calibrations: by_frame.setdefault(int(row["frame_index"]), []).append(row)
    pose_map = {int(row["frame_index"]): row for row in ego_poses}; root = Path(future_image_root)
    for frame in range(frame_start, frame_end):
        images = {camera: read_image_bgr(root / camera / f"{frame:06d}.jpg") for camera in CAMERAS}
        backend.frame_index = frame - 1
        backend.track_future_frame(images, by_frame[frame], pose_map[frame], timestamps[frame])
    result = backend.finalize(); observations = result.pop("observations"); history = result.pop("history_observations")
    per_camera = {camera: [row for row in observations if row["camera_name"] == camera] for camera in CAMERAS}
    agents = sorted({row["agent_id"] for row in observations}); per_agent = {agent: [row for row in observations if row["agent_id"] == agent] for agent in agents}
    write_json(output / "per_camera_tracks.json", per_camera); write_json(output / "per_agent_tracks.json", per_agent)
    write_json(output / "history_visual_tracks.json", history); write_json(output / "masks/index.json", [{"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"], "polygon": row["mask_polygon"]} for row in observations])
    write_json(output / "tracked_points/index.json", [{"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"], "points": row["tracked_points"]} for row in observations])
    write_json(output / "optical_flow/index.json", [{"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"], "vectors": row["flow_vectors"]} for row in observations])
    write_json(output / "embeddings/index.json", [{"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"], "embedding": row["appearance_embedding"]} for row in observations])
    write_json(output / "inference_profile.json", result); return {**result, "observations": observations, "history_observations": history}

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

from .schema import CAMERAS


def chain(nusc, first_token: str) -> list[dict]:
    rows = []
    token = first_token
    while token:
        row = nusc.get("sample_data", token)
        rows.append(row); token = row["next"]
    return rows


def _nearest(rows: list[dict], timestamp: int) -> dict:
    stamps = [row["timestamp"] for row in rows]
    index = bisect_left(stamps, timestamp)
    options = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return rows[min(options, key=lambda i: (abs(stamps[i] - timestamp), stamps[i]))]


def build(nusc, scene: dict, total_frames: int = 90, max_offset_us: int = 60_000) -> tuple[list[dict] | None, dict]:
    first_sample = nusc.get("sample", scene["first_sample_token"])
    streams = {camera: chain(nusc, first_sample["data"][camera]) for camera in CAMERAS}
    front = streams["CAM_FRONT"]
    best = None
    rejection = {"too_short": {c: len(v) for c, v in streams.items() if len(v) < total_frames}}
    if rejection["too_short"]:
        return None, rejection
    # nuScenes camera streams are asynchronous and occasionally contain an extra
    # 20 Hz exposure in only some cameras.  Build the native common subsequence;
    # never duplicate a slower camera frame to fill such an exposure.
    synchronized = []
    previous = {camera: None for camera in CAMERAS}
    for ref in front:
        cameras = {}
        for camera in CAMERAS:
            row = ref if camera == "CAM_FRONT" else _nearest(streams[camera], ref["timestamp"])
            if row["token"] == previous[camera] or abs(row["timestamp"] - ref["timestamp"]) > max_offset_us:
                cameras = {}; break
            cameras[camera] = row
        if cameras:
            synchronized.append((ref, cameras)); previous = {camera: row["token"] for camera, row in cameras.items()}
    for start in range(len(synchronized) - total_frames + 1):
        candidate = []
        valid = True
        offsets = []
        seen = {camera: set() for camera in CAMERAS}
        for frame_index, (ref, matched) in enumerate(synchronized[start:start + total_frames]):
            cameras = {}
            for camera in CAMERAS:
                row = matched[camera]
                offset = int(row["timestamp"] - ref["timestamp"])
                if abs(offset) > max_offset_us or row["token"] in seen[camera]:
                    valid = False; break
                seen[camera].add(row["token"]); offsets.append(abs(offset))
                cameras[camera] = {"sample_data_token": row["token"], "timestamp_us": int(row["timestamp"]),
                                   "offset_us": offset, "source_path": str(Path(nusc.dataroot) / row["filename"]),
                                   "is_key_frame": bool(row["is_key_frame"]), "sample_token": row["sample_token"]}
            if not valid: break
            candidate.append({"frame_index": frame_index, "reference_timestamp_us": int(ref["timestamp"]), "cameras": cameras})
        if valid:
            score = (max(offsets), sum(offsets) / len(offsets), start)
            if best is None or score < best[0]:
                best = (score, candidate)
    if best is None:
        return None, {**rejection, "reason": "no_90_frame_native_common_subsequence_with_unique_tokens_and_offset_limit",
                      "native_common_subsequence_length": len(synchronized)}
    score, timeline = best
    return timeline, {"maximum_offset_us": score[0], "mean_absolute_offset_us": score[1], "start_index": score[2],
                      "per_camera_frame_count": {camera: total_frames for camera in CAMERAS},
                      "native_common_subsequence_length": len(synchronized),
                      "reference_chain_frames_skipped_to_avoid_cross_camera_duplication": len(front) - len(synchronized)}

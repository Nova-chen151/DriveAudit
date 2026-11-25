from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def compute_presence_metrics(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                             mapping: list[dict[str, Any]]) -> dict[str, Any]:
    submitted = {track["agent_id"]: track for track in submission_tracks}
    extracted = {track["image_track_id"]: track for track in image_tracks}
    expected, found = 0, 0
    appearance_errors, disappearance_errors = [], []
    matched_ids, recovered_ids = set(), set()
    for match in mapping:
        sid, iid = match["submission_agent_id"], match["image_track_id"]
        if sid is None:
            continue
        visible_frames = {row["frame_index"] for row in submitted[sid]["observations"] if row.get("expected_visible", True)}
        expected += len(visible_frames)
        if iid:
            image_frames = {row["frame_index"] for row in extracted[iid]["observations"] if row.get("source") == "image_detection"}
            common = visible_frames & image_frames
            found += len(common); matched_ids.add(sid)
            if match["match_status"] == "matched_history_anchor": recovered_ids.add(sid)
            if visible_frames and image_frames:
                appearance_errors.append(min(image_frames) - min(visible_frames))
                disappearance_errors.append(max(image_frames) - max(visible_frames))
    persistent = [track for track in submission_tracks if track.get("history_present", False)]
    entrants = [track for track in submission_tracks if not track.get("history_present", False)]
    observable_agents = [track for track in submission_tracks if any(row.get("expected_visible", True) for row in track["observations"])]
    unmatched_image = sum(row["match_status"] == "unmatched_image_track" for row in mapping)
    result = {"expected_visible_agent_frames": expected,
            "visible_agent_recall": found / expected if expected else NOT_AVAILABLE,
            "track_recovery_rate": sum(track["agent_id"] in recovered_ids for track in persistent) / len(persistent) if persistent else NOT_AVAILABLE,
            "future_entry_recall": sum(track["agent_id"] in matched_ids for track in entrants) / len(entrants) if entrants else NOT_AVAILABLE,
            "unmatched_submission_ratio": sum(track["agent_id"] not in matched_ids for track in observable_agents) / len(observable_agents) if observable_agents else NOT_AVAILABLE,
            "unmatched_image_track_ratio": unmatched_image / len(image_tracks) if image_tracks else NOT_AVAILABLE,
            "appearance_time_error_frames": sum(abs(value) for value in appearance_errors) / len(appearance_errors) if appearance_errors else NOT_AVAILABLE,
            "disappearance_time_error_frames": sum(abs(value) for value in disappearance_errors) / len(disappearance_errors) if disappearance_errors else NOT_AVAILABLE,
            "object_survival_consistency": found / expected if expected else NOT_AVAILABLE}
    groups = {"persistent_agents": persistent, "future_entrants": entrants}
    for category in ("vehicle", "car", "pedestrian", "bicycle"):
        groups[category] = [track for track in submission_tracks if track.get("agent_type") == category or track.get("category") == category]
    result["by_group"] = {name: {"agent_count": len(values),
                                  "matched_agent_ratio": sum(track["agent_id"] in matched_ids for track in values) / len(values) if values else NOT_AVAILABLE}
                          for name, values in groups.items()}
    return result

from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import mean


def compute_identity_metrics(projections: list[dict[str, Any]], visual: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(row["agent_id"], row["camera_name"], row["frame_index"]) for row in projections if row["history_anchored"] and row["expected_visible"]}
    groups = defaultdict(list)
    for row in visual: groups[(row["agent_id"], row["camera_name"])].append(row)
    supported = sum(row["visible"] and (row["agent_id"], row["camera_name"], row["frame_index"]) in expected for row in visual)
    fragmentation, continuous, lost, reidentified, appearance = 0, [], [], 0, []
    for rows in groups.values():
        rows.sort(key=lambda row: row["frame_index"]); run = 0; previous_visible = False; seen_visible = False
        for row in rows:
            if row["visible"]:
                if not previous_visible and seen_visible: fragmentation += 1
                run += 1; appearance.append(row.get("appearance_similarity", NOT_AVAILABLE))
                reidentified += row["track_status"] == "reidentified"
                seen_visible = True
            else:
                if run: continuous.append(run); run = 0
                lost.append(1)
            previous_visible = row["visible"]
        if run: continuous.append(run)
    identity = supported / len(expected) if expected else NOT_AVAILABLE
    return {"identity_persistence": identity, "track_survival": identity, "id_switch_count": 0,
            "id_switch_semantics": "history_agent_id_bound_state_no_global_reassignment", "fragmentation": fragmentation,
            "reidentification_success": reidentified / max(1, sum(row["track_status"] == "temporarily_lost" for row in visual) + reidentified),
            "lost_duration_frames": mean(lost), "average_continuous_tracking_length": mean(continuous),
            "appearance_embedding_stability": mean(appearance), "cross_view_identity_consistency": identity,
            "camera_handoff_success": NOT_AVAILABLE, "tracked_state_count": len(groups)}

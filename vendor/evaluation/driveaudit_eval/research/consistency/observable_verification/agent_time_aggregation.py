from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
    result = []
    for (agent, frame), cameras in grouped.items():
        states = Counter(row["motion_state"] for row in cameras)
        if states["observable_supported"] and not states["observable_contradicted"]: state = "supported"
        elif states["observable_contradicted"] and not states["observable_supported"]: state = "contradicted"
        elif states["observable_supported"] and states["observable_contradicted"]: state = "multiview_ambiguous"
        elif states["observable_ambiguous"]: state = "multiview_ambiguous"
        elif states["visually_unscorable"] or states["visually_unscorable_contamination"]: state = "unscorable"
        else: state = "geometry_unobservable"
        result.append({"agent_id": agent, "frame_index": frame, "agent_time_state": state,
                       "selected_cameras": [row["camera_name"] for row in cameras], "camera_state_counts": dict(states)})
    return result

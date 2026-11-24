from __future__ import annotations

from typing import Any


REQUIRED_EVIDENCE_FIELDS = {
    "pair_id", "agent_id", "camera_name", "frame_index", "target_frame_index",
    "window_length", "metric_type", "expected_visible", "support_confidence",
}


def validate_evidence(row: dict[str, Any]) -> None:
    missing = REQUIRED_EVIDENCE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"TCVV evidence missing fields: {sorted(missing)}")
    if row["metric_type"] != "conditional_visual_verification":
        raise ValueError("TCVV metrics must declare conditional_visual_verification")
    if int(row["window_length"]) not in range(2, 7):
        raise ValueError("TCVV window length must be between 2 and 6")

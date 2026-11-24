from __future__ import annotations

from typing import Any


def diagnose(observed: dict[str, Any] | None) -> dict[str, Any]:
    if not observed: state = "local_structure_unavailable"
    elif observed.get("image_quality_status", "").startswith("unscorable"): state = "local_structure_uncertain"
    elif int(observed.get("core_point_count", 0)) > 0: state = "local_structure_available"
    else: state = "local_structure_unavailable"
    return {"weak_local_visual_structure_diagnostic": state,
            "local_texture_score": (observed or {}).get("core_support_ratio"),
            "local_edge_score": (observed or {}).get("foreground_contrast_score"),
            "local_trackable_point_score": (observed or {}).get("valid_point_ratio"),
            "local_temporal_continuity": (observed or {}).get("feature_support_score"),
            "presence_claimed": False, "include_in_primary_gate": False}

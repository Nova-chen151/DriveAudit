from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import groups, mean, summary, weighted_mean


def _bucket(row: dict[str, Any]) -> dict[str, str]:
    depth = float(row.get("projected_depth") or 0)
    area = float(row.get("projected_area") or 0)
    occlusion = float(row.get("occlusion_ratio") or 0)
    return {"distance_bucket": "0-20" if depth < 20 else "20-40" if depth < 40 else "40-60" if depth < 60 else ">60",
            "area_bucket": "64-256" if area < 256 else "256-1024" if area < 1024 else ">1024",
            "occlusion_bucket": "0-0.2" if occlusion < .2 else "0.2-0.5" if occlusion < .5 else "0.5-0.8"}


def presence_metrics(evidence: list[dict[str, Any]], projections: list[dict[str, Any]], primary_window: int,
                     camera_weights: dict[str, float] | None = None) -> dict[str, Any]:
    rows = [{**row, **_bucket(row)} for row in evidence if int(row["window_length"]) == int(primary_window)]
    supported = [bool(row["visual_supported"]) for row in rows]
    weights = [float((camera_weights or {}).get(row["camera_name"], 1.0)) for row in rows]
    raw = sum(supported) / len(rows) if rows else NOT_AVAILABLE
    weighted = weighted_mean([float(v) for v in supported], weights)
    grouped_results: dict[str, Any] = {}
    for label, key in (("agent_scope", "history_anchored"), ("agent_type", "agent_type"), ("camera", "camera_name"),
                       ("distance", "distance_bucket"), ("projected_area", "area_bucket"),
                       ("occlusion", "occlusion_bucket")):
        grouped_results[label] = {str(group[0]): {"samples": len(items),
                                  "support_rate": sum(bool(item["visual_supported"]) for item in items) / len(items)}
                                  for group, items in groups(rows, key).items()}
    expected_by_agent_camera: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in projections:
        if row.get("expected_visible"):
            expected_by_agent_camera[(row["agent_id"], row["camera_name"])].append(int(row["frame_index"]))
    support_by_agent_camera: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        if row["visual_supported"]:
            support_by_agent_camera[(row["agent_id"], row["camera_name"])].append(int(row["frame_index"]))
    entry_errors, exit_errors, duration_ratios = [], [], []
    for key, expected_frames in expected_by_agent_camera.items():
        actual = support_by_agent_camera.get(key, [])
        if actual:
            entry_errors.append(abs(min(actual) - min(expected_frames)))
            exit_errors.append(abs(max(actual) - max(expected_frames)))
            duration_ratios.append(len(set(actual)) / max(1, len(set(expected_frames))))
    return {"metric_type": "conditional_visual_verification", "projection_support_rate": raw,
            "reliability_weighted_projection_support_rate": weighted,
            "missing_visual_evidence_ratio": 1.0 - raw if isinstance(raw, (int, float)) else NOT_AVAILABLE,
            "support_confidence": summary(row["support_confidence"] for row in rows),
            "persistent_agent_support_rate": mean(float(row["visual_supported"]) for row in rows if row["history_anchored"]),
            "future_entrant_support_rate": mean(float(row["visual_supported"]) for row in rows if not row["history_anchored"]),
            "support_duration_ratio": mean(duration_ratios),
            "premature_disappearance_ratio": mean(float(value > 0) for value in exit_errors),
            "over_persistence_ratio": NOT_AVAILABLE,
            "over_persistence_unavailable_reason": "verification is bounded to expected-visible projection regions",
            "appearance_time_error_frames": mean(entry_errors), "disappearance_time_error_frames": mean(exit_errors),
            "expected_agent_camera_frames": len(rows), "supported_agent_camera_frames": sum(supported),
            "groups": grouped_results, "per_frame": rows}


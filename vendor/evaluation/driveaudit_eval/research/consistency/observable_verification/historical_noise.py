from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import write_json


def robust_sigma(values: list[float], floor: float) -> dict[str, float | int | None]:
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(x): return {"sample_count": 0, "median_residual": None, "mad_residual": None, "p75_residual": None, "p90_residual": None, "selected_sigma": floor}
    median = float(np.median(x)); mad = float(np.median(np.abs(x - median)))
    return {"sample_count": len(x), "median_residual": median, "mad_residual": mad,
            "p75_residual": float(np.percentile(x, 75)), "p90_residual": float(np.percentile(x, 90)),
            "selected_sigma": max(float(floor), 1.4826 * mad)}


def fit(history_rows: list[dict[str, Any]], config: dict[str, Any], output: str | Path | None = None) -> dict[str, Any]:
    cfg = config["historical_noise"]; floor = float(cfg["noise_floor_px"]); minimum = int(cfg["min_bucket_samples"])
    area_edges, distance_edges = list(map(float, cfg["area_buckets"])), list(map(float, cfg["distance_buckets_m"]))
    rows = []
    for row in history_rows:
        observed = np.asarray(row.get("observed_flow_vectors", []), float)
        expected = np.asarray(row.get("target_expected_flow_per_point", []), float)
        count = min(len(observed), len(expected))
        value = (float(np.median(np.linalg.norm(observed[:count]-expected[:count], axis=1))) if count
                 else row.get("target_stats", {}).get("robust_residual"))
        if not isinstance(value, (int, float)) or not np.isfinite(value): continue
        rows.append({"camera": row["camera_name"], "area_bucket": _bucket(float(row.get("projected_area") or 0), area_edges),
                     "distance_bucket": _bucket(float(row.get("projected_depth") or 1e9), distance_edges), "residual": float(value)})
    global_stats = robust_sigma([r["residual"] for r in rows], floor)
    camera_stats = {camera: robust_sigma([r["residual"] for r in rows if r["camera"] == camera], floor) for camera in sorted({r["camera"] for r in rows})}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows: grouped[(row["camera"], row["area_bucket"], row["distance_bucket"])].append(row["residual"])
    buckets = []
    for (camera, area, distance), values in sorted(grouped.items()):
        local = robust_sigma(values, floor); camera_value = camera_stats[camera]
        if local["sample_count"] >= minimum: selected, fallback = local["selected_sigma"], "per_camera_area_distance_bucket"
        elif camera_value["sample_count"] >= minimum: selected, fallback = camera_value["selected_sigma"], "per_camera"
        elif global_stats["sample_count"]: selected, fallback = global_stats["selected_sigma"], "global_pooled"
        else: selected, fallback = floor, "configured_floor"
        buckets.append({"camera": camera, "area_bucket": area, "distance_bucket": distance, **local,
                        "selected_sigma": float(max(floor, selected)), "fallback_level": fallback})
    model = {"estimator": "mad_robust_sigma", "history_only": True, "future_images_read": False,
             "noise_floor_px": floor, "min_bucket_samples": minimum, "global": global_stats,
             "per_camera": camera_stats, "buckets": buckets,
             "fallback_counts": dict(__import__("collections").Counter(r["fallback_level"] for r in buckets))}
    if output is not None: write_json(Path(output) / "noise_model.json", model)
    return model


def select_sigma(model: dict[str, Any], camera: str, area: float, distance: float, config: dict[str, Any]) -> tuple[float, str]:
    cfg = config["historical_noise"]
    area_bucket = _bucket(area, list(map(float, cfg["area_buckets"])))
    distance_bucket = _bucket(distance, list(map(float, cfg["distance_buckets_m"])))
    match = next((r for r in model["buckets"] if r["camera"] == camera and r["area_bucket"] == area_bucket and r["distance_bucket"] == distance_bucket), None)
    if match: return float(match["selected_sigma"]), str(match["fallback_level"])
    camera_stats = model["per_camera"].get(camera)
    if camera_stats and camera_stats["sample_count"] >= int(cfg["min_bucket_samples"]): return float(camera_stats["selected_sigma"]), "per_camera"
    if model["global"]["sample_count"]: return float(model["global"]["selected_sigma"]), "global_pooled"
    return float(cfg["noise_floor_px"]), "configured_floor"


def _bucket(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi: return f"[{lo:g},{hi:g})"
    return f"[{edges[-2]:g},{edges[-1]:g})"

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import write_csv, write_json
from driveaudit_eval.research.consistency.observed_flow.visibility import polygon_mask


def region_mask(row: dict[str, Any], peers: list[dict[str, Any]], size: tuple[int, int], config: dict[str, Any], mode: str = "core") -> np.ndarray:
    if mode == "rectangle":
        mask = np.zeros((size[1], size[0]), np.uint8); x1, y1, x2, y2 = map(int, row["projected_box_2d"]); mask[max(0,y1):min(size[1],y2+1), max(0,x1):min(size[0],x2+1)] = 1
    else: mask = polygon_mask(row["projected_polygon"], size)
    if mode == "core" and mask.any():
        x, y, w, h = cv2.boundingRect(np.asarray(row["projected_polygon"], np.float32)); radius = max(1, int(round(min(w, h) * float(config["instance_region"]["erosion_ratio"]))))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        if config["instance_region"]["exclude_overlap"]:
            other = np.zeros_like(mask)
            for peer in peers:
                if peer["agent_id"] == row["agent_id"] or not peer.get("projected_valid") or not _boxes_overlap(row["projected_box_2d"], peer["projected_box_2d"]): continue
                other |= polygon_mask(peer["projected_polygon"], size)
            mask[other > 0] = 0
    return mask


def build(projections: list[dict[str, Any]], config: dict[str, Any], output: str | Path,
          image_size: tuple[int, int] | None = None, mode: str = "core") -> list[dict[str, Any]]:
    size = image_size or tuple(config["image"]["formal_size"]); grouped = defaultdict(list)
    for row in projections: grouped[(int(row["frame_index"]), row["camera_name"])].append(row)
    regions = []
    for row in projections:
        mask = region_mask(row, grouped[(int(row["frame_index"]), row["camera_name"])], size, config, mode) if row.get("expected_visible") else np.zeros((size[1], size[0]), np.uint8)
        core_pixels = int(mask.sum()); scorable = bool(row.get("expected_visible") and core_pixels >= int(config["instance_region"]["minimum_core_pixels"]))
        regions.append({"agent_id": row["agent_id"], "agent_type": row["agent_type"], "category": row["category"],
                        "frame_index": int(row["frame_index"]), "camera_name": row["camera_name"], "timestamp": row["timestamp"],
                        "instance_region_source": "projected_visible_face_core" if mode == "core" else f"projected_{mode}",
                        "gt_instance_mask_available": False, "gt_depth_available": False, "region_mode": mode,
                        "projected_valid": bool(row.get("projected_valid")),
                        "core_pixel_count": core_pixels, "foreground_core_coverage": core_pixels / max(float(row.get("projected_area", 0)), 1),
                        "occlusion_ratio": row.get("occlusion_ratio"), "expected_visible": row.get("expected_visible", False),
                        "region_scorable": scorable, "region_reason": "scorable" if scorable else row.get("visibility_reason", "insufficient_core"),
                        "projected_area": row.get("projected_area", 0)})
    output = Path(output); write_csv(output / "gt_instance_regions/per_frame.csv", regions)
    write_json(output / "gt_instance_regions/summary.json", {"instance_region_source": "projected_visible_face_core",
        "gt_instance_mask": "not_available", "gt_depth": "not_available", "no_independent_2d_gt": True,
        "region_count": len(regions), "scorable_region_count": sum(row["region_scorable"] for row in regions)})
    return regions


def _boxes_overlap(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

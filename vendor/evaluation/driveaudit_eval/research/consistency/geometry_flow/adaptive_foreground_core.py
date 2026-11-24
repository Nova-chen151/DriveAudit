from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.visibility import polygon_mask
from driveaudit_eval.research.consistency.observed_flow.gt_instance_region import _boxes_overlap


def erosion_ratio(area: float, config: dict[str, Any]) -> float:
    low, high = map(float, config["area_thresholds"]); small, medium, large = map(float, config["erosion_ratios"])
    return small if area < low else (medium if area < high else large)


def build(row: dict[str, Any], surface: dict[str, Any] | None, peers: list[dict[str, Any]], size: tuple[int, int], config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    polygons = [face["polygon"] for face in (surface or {}).get("visible_surfaces", [])]
    if not polygons: polygons = [row["projected_polygon"]]
    mask=np.zeros((size[1],size[0]),np.uint8)
    for polygon in polygons:mask|=polygon_mask(polygon,size)
    original=int(mask.sum());ratio=erosion_ratio(original,config);radius=max(1,int(round(np.sqrt(max(original,1))*ratio/4)))
    core=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    before_overlap=int(core.sum());other=np.zeros_like(core)
    if config["exclude_overlap"]:
        for peer in peers:
            if peer["agent_id"]==row["agent_id"] or not peer.get("projected_valid") or not _boxes_overlap(row["projected_box_2d"],peer["projected_box_2d"]):continue
            if float(peer.get("projected_depth",float("inf"))) >= float(row.get("projected_depth",float("inf"))):continue
            other|=polygon_mask(peer["projected_polygon"],size)
        core[other>0]=0
    final=int(core.sum());stats={"original_polygon_area":original,"core_area_before_overlap":before_overlap,"erosion_ratio":ratio,
        "overlap_removed_area":before_overlap-final,"occlusion_removed_area":int(round(original*float(row.get("occlusion_ratio",0)))),
        "final_core_area":final,"core_valid":final>=int(config["minimum_core_pixels"]),"core_source":"visible_surface_union" if surface else "projected_polygon_fallback"}
    return core,stats

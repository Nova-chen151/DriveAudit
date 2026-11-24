from __future__ import annotations

from typing import Any
import numpy as np


def detect(observed:dict[str,Any],neighbor_overlap:float,config:dict[str,Any])->dict[str,Any]:
    core=float(observed.get("core_point_fraction",0));boundary=float(observed.get("boundary_support_ratio",0));roi=float(observed.get("roi_support_ratio",0))
    edge=max(0.0,roi-float(observed.get("core_support_ratio",0)));clusters=int(observed.get("motion_cluster_count",0))
    probability=float(np.clip(.35*(1-core)+.25*edge+.25*neighbor_overlap+.15*max(0,clusters-1),0,1))
    reasons=[]
    if core<float(config["minimum_core_point_fraction"]):reasons.append("core_support_insufficient")
    if edge>float(config["maximum_edge_dominance"]):reasons.append("roi_edge_motion_dominates")
    if neighbor_overlap>float(config["neighbor_overlap_threshold"]):reasons.append("neighbor_agent_contamination")
    if clusters>1:reasons.append("multiple_motion_clusters")
    location="core" if core>=.5 else "boundary_or_expanded_roi"
    return {"contamination_probability":probability,"neighbor_agent_contamination":neighbor_overlap,
            "background_edge_contamination":edge,"core_point_fraction":core,"dominant_cluster_location":location,
            "contamination_rejection_reason":reasons or ["none"],"motion_cluster_count":clusters}


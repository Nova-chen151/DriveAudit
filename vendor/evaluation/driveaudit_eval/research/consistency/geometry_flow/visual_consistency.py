from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import finite_mean, rate, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.common import cosine


def metrics(result:dict[str,Any],model:dict[str,Any],patch:dict[str,Any],thresholds:dict[str,Any],support_config:dict[str,Any])->dict[str,Any]:
    expected=np.asarray(result.get("expected_flow_per_point",[]),float).reshape(-1,2);observed=np.asarray(result.get("observed_flow_per_point",[]),float).reshape(-1,2)
    expected_median=np.median(expected,axis=0) if len(expected) else np.zeros(2);observed_median=np.median(observed,axis=0) if len(observed) else np.zeros(2)
    residual=float(result["residual_epe"]) if isinstance(result.get("residual_epe"),(int,float)) else None;flow_cosine=cosine(observed_median,expected_median) if len(observed) else None
    expected_magnitude=float(np.linalg.norm(expected_median));observed_magnitude=float(np.linalg.norm(observed_median));diagonal=math.hypot(model["source_projection"]["projected_box_2d"][2]-model["source_projection"]["projected_box_2d"][0],model["source_projection"]["projected_box_2d"][3]-model["source_projection"]["projected_box_2d"][1])
    affine=np.asarray(result.get("residual_affine"),float) if result.get("residual_affine") is not None else None
    residual_scale=math.sqrt(abs(np.linalg.det(affine[:,:2]))) if affine is not None else None;residual_rotation=math.atan2(affine[1,0],affine[0,0]) if affine is not None else None
    scale_agreement=max(0.,1-abs((residual_scale or 1.)-1));rotation_agreement=max(0.,1-abs(residual_rotation or 0)/math.pi)
    evidence=bool(result.get("valid_count",0)>=int(thresholds["minimum_valid_points"]) or result.get("fallback_confidence",0)>.2)
    strong=evidence and residual is not None and residual<=float(thresholds["maximum_flow_epe_px"]) and flow_cosine is not None and flow_cosine>=float(thresholds["minimum_flow_cosine"]) and result.get("valid_point_ratio",0)>=float(thresholds["minimum_valid_point_ratio"]) and result.get("ransac_inlier_ratio",0)>=float(support_config["minimum_ransac_inlier_ratio"]) and (patch.get("photometric_warp_error") is None or patch["photometric_warp_error"]<=float(support_config["maximum_photometric_error"]))
    weak=evidence and residual is not None and residual<=2*float(thresholds["maximum_flow_epe_px"]) and flow_cosine is not None and flow_cosine>=float(support_config["weak_minimum_cosine"])
    state="supported" if strong else ("weakly_supported" if weak else ("contradicted" if evidence and flow_cosine is not None and flow_cosine<0 else "unscorable"))
    return {"residual_epe":residual,"normalized_residual_epe_box_diagonal":residual/max(diagonal,1e-6) if residual is not None else None,
        "normalized_residual_epe_expected_motion":residual/max(expected_magnitude,1.) if residual is not None else None,
        "flow_cosine":flow_cosine,"motion_direction_agreement":float(flow_cosine>=.75) if flow_cosine is not None else None,
        "motion_magnitude_relative_error":abs(observed_magnitude-expected_magnitude)/max(expected_magnitude,1.),"scale_change_agreement":scale_agreement,
        "rotation_change_agreement":rotation_agreement,"photometric_warp_error":patch.get("photometric_warp_error"),"gradient_warp_error":patch.get("gradient_warp_error"),
        "valid_point_ratio":float(result.get("valid_point_ratio",0)),"tracking_success":evidence,"support_state":state,
        "expected_flow_median":expected_median.tolist(),"observed_flow_median":observed_median.tolist(),"residual_flow_is_rgb_measured":True,
        "gt_prediction_used_as_observation":False}


def summarize(rows:list[dict[str,Any]],prefix:str="guided")->dict[str,Any]:
    usable=[row for row in rows if row.get(f"{prefix}_support_state")!="unscorable"];total=len(rows)
    return {"window_count":total,"scorable_window_count":len(usable),"scorable_coverage":len(usable)/max(1,total),
        "residual_epe":finite_mean(row.get(f"{prefix}_residual_epe") for row in usable),"flow_cosine":finite_mean(row.get(f"{prefix}_flow_cosine") for row in usable),
        "direction_agreement":finite_mean(row.get(f"{prefix}_motion_direction_agreement") for row in usable),"magnitude_error":finite_mean(row.get(f"{prefix}_motion_magnitude_relative_error") for row in usable),
        "scale_agreement":finite_mean(row.get(f"{prefix}_scale_change_agreement") for row in usable),"rotation_agreement":finite_mean(row.get(f"{prefix}_rotation_change_agreement") for row in usable),
        "valid_point_ratio":finite_mean(row.get(f"{prefix}_valid_point_ratio") for row in rows),"strong_support_rate":rate(row.get(f"{prefix}_support_state")=="supported" for row in usable),
        "strong_or_weak_support_rate":rate(row.get(f"{prefix}_support_state") in {"supported","weakly_supported"} for row in usable),
        "weakly_supported_rate":rate(row.get(f"{prefix}_support_state")=="weakly_supported" for row in usable),"unscorable_rate":sum(row.get(f"{prefix}_support_state")=="unscorable" for row in rows)/max(1,total),
        "contradicted_rate":sum(row.get(f"{prefix}_support_state")=="contradicted" for row in rows)/max(1,total)}


def compare(rows:list[dict[str,Any]])->dict[str,Any]:
    ordinary,guided=summarize(rows,"ordinary"),summarize(rows,"guided")
    return {"ordinary_lk":ordinary,"geometry_guided_lk":guided,"valid_point_gain":_delta(guided["valid_point_ratio"],ordinary["valid_point_ratio"]),
        "epe_reduction":_delta(ordinary["residual_epe"],guided["residual_epe"]),"cosine_gain":_delta(guided["flow_cosine"],ordinary["flow_cosine"]),
        "support_gain":_delta(guided["strong_support_rate"],ordinary["strong_support_rate"]),"unscorable_reduction":ordinary["unscorable_rate"]-guided["unscorable_rate"]}


def _delta(a,b):return float(a)-float(b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None

from __future__ import annotations

from typing import Any


def evaluate(annotation_ok:bool,projection_ok:bool,metrics:dict[str,Any],multiview:dict[str,Any],config:dict[str,Any],input_ok=True,time_ok=True)->dict[str,Any]:
    t=config["gate"]
    quality_checks={"flow_cosine":_at_least(metrics.get("flow_cosine"),t["minimum_flow_cosine"]),"direction_agreement":_at_least(metrics.get("direction_agreement"),t["minimum_direction_agreement"]),
        "magnitude_error":_at_most(metrics.get("magnitude_error"),t["maximum_magnitude_error"]),"scale_agreement":_at_least(metrics.get("scale_agreement"),t["minimum_scale_agreement"]),
        "rotation_agreement":_at_least(metrics.get("rotation_agreement"),t["minimum_rotation_agreement"]),"residual_epe":_at_most(metrics.get("residual_epe"),t["maximum_residual_epe_px"])}
    coverage_checks={"valid_point_ratio":_at_least(metrics.get("valid_point_ratio"),t["minimum_valid_point_ratio"]),"scorable_coverage":_at_least(metrics.get("scorable_coverage"),t["minimum_scorable_coverage"]),
        "strong_support":_at_least(metrics.get("strong_support_rate"),t["minimum_strong_support"]),"strong_or_weak_support":_at_least(metrics.get("strong_or_weak_support_rate"),t["minimum_strong_or_weak_support"]),
        "unscorable_rate":_at_most(metrics.get("unscorable_rate"),t["maximum_unscorable_rate"])}
    multiview_checks={"camera_support":_at_least(multiview.get("gt_camera_support_recall"),t["minimum_camera_support_recall"]),"contradiction":_at_most(multiview.get("gt_multiview_contradiction_rate"),t["maximum_multiview_contradiction"])}
    quality,coverage,multi=all(quality_checks.values()),all(coverage_checks.values()),all(multiview_checks.values())
    if input_ok and time_ok and annotation_ok and projection_ok and quality and coverage and multi:status="valid_gt_consistency"
    elif input_ok and time_ok and annotation_ok and projection_ok and quality and multi:status="valid_gt_consistency_limited_coverage"
    elif not input_ok:status="unscorable_missing_gt_fields"
    elif not time_ok:status="invalid_time_alignment"
    elif not projection_ok:status="invalid_calibration"
    elif not annotation_ok:status="invalid_gt_pairing"
    else:status="invalid_gt_visual_alignment"
    return {"gate_1_input":{"passed":input_ok},"gate_2_time":{"passed":time_ok},"gate_3_projection":{"passed":projection_ok},"gate_4_annotation":{"passed":annotation_ok},"gate_5_instance_region":{"passed":metrics.get("window_count",0)>0},
        "gate_6a_motion_quality":{"passed":quality,"checks":quality_checks},"gate_6b_evidence_coverage":{"passed":coverage,"checks":coverage_checks},"gate_7_multiview":{"passed":multi,"checks":multiview_checks},
        "gate_8_upper_bound":{"passed":status=="valid_gt_consistency","status":status},"thresholds_lowered_from_v1":False}


def _at_least(value,threshold):return isinstance(value,(int,float)) and float(value)>=float(threshold)
def _at_most(value,threshold):return isinstance(value,(int,float)) and float(value)<=float(threshold)


from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


LEVELS = {"DIRECT_EXACT", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "CONCEPTUAL_ADAPTATION", "RELATED_NOT_EQUIVALENT",
          "NOT_IMPLEMENTED", "NOT_AVAILABLE_DATA", "NOT_AVAILABLE_DEPENDENCY", "INSUFFICIENT_SAMPLE"}


METRICS = [
    ("submission_ref_center_distance_mAP", "BEVFormer/nuScenes", "mAP", "score-ranked center-distance AP at 0.5/1/2/4m", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_mATE", "BEVFormer/nuScenes", "mATE", "mean matched xy translation error", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_mASE", "BEVFormer/nuScenes", "mASE", "1 minus aligned 3D scale IoU", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_mAOE", "BEVFormer/nuScenes", "mAOE", "periodic yaw error", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_mAVE", "BEVFormer/nuScenes", "mAVE", "xy velocity L2 error", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_mAAE", "BEVFormer/nuScenes", "mAAE", "attribute classification error", "NOT_AVAILABLE_DATA", "bevformer_metrics.json"),
    ("submission_ref_NDS_adapted", "BEVFormer/nuScenes", "NDS", "official weighted detection composite", "NOT_AVAILABLE_DATA", "bevformer_metrics.json"),
    ("submission_ref_3d_AP_iou_0_5", "BEVFormer/Waymo", "3D AP@0.5", "score-ranked 3D IoU AP", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_3d_APH_iou_0_5", "BEVFormer/Waymo", "3D APH@0.5", "3D IoU AP with heading weighting", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_3d_AP_iou_0_7", "BEVFormer/Waymo", "3D AP@0.7", "score-ranked 3D IoU AP", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_3d_APH_iou_0_7", "BEVFormer/Waymo", "3D APH@0.7", "3D IoU AP with heading weighting", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("submission_ref_dynamic_occupancy_iou", "BEVFormer", "BEV segmentation IoU", "dynamic-box occupancy intersection over union", "CONCEPTUAL_ADAPTATION", "bevformer_metrics.json"),
    ("visibility_stratified", "BEVFormer", "Visibility analysis", "metrics grouped by estimated visible proportion", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "bevformer_metrics.json"),
    ("temporal_recall_gain", "BEVFormer", "BEVFormer vs BEVFormer-S", "prev_bev temporal minus reset-prev-bev recall", "CONCEPTUAL_ADAPTATION", "bevformer_metrics.json"),
    ("act_agent_motion_accuracy_adapted", "ACT-Bench", "IEC Accuracy", "instruction motion class equals image-executed class", "CONCEPTUAL_ADAPTATION", "actbench_metrics.json"),
    ("act_agent_ade_adapted", "ACT-Bench", "ADE", "mean instruction/executed displacement", "CONCEPTUAL_ADAPTATION", "actbench_metrics.json"),
    ("act_agent_fde_adapted", "ACT-Bench", "FDE", "last common visible instruction/executed displacement", "CONCEPTUAL_ADAPTATION", "actbench_metrics.json"),
    ("drivinggen_alignment_dtw", "DrivingGen", "Trajectory Alignment DTW", "DTW on common visible world trajectories", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "drivinggen_metrics.json"),
    ("agent_missing_ratio", "DrivingGen", "Agent Missing", "unmatched expected-visible agent-frame ratio", "CONCEPTUAL_ADAPTATION", "drivinggen_metrics.json"),
    ("agent_consistency", "DrivingGen", "Agent Consistency", "identity/class/size/direction/survival consistency", "CONCEPTUAL_ADAPTATION", "drivinggen_metrics.json"),
    ("trajectory_consistency", "DrivingGen", "Trajectory Consistency", "speed/acceleration/jerk/yaw/curvature consistency", "CONCEPTUAL_ADAPTATION", "drivinggen_metrics.json"),
    ("official_ftd", "DrivingGen", "FTD", "MTR-encoder Frechet trajectory distance", "NOT_AVAILABLE_DEPENDENCY", "drivinggen_metrics.json"),
    ("worldlens_temporal_object_consistency_adapted", "WorldLens", "Temporal Consistency", "track-kinematic temporal consistency", "CONCEPTUAL_ADAPTATION", "worldlens_metrics.json"),
    ("worldlens_cross_view_object_support_adapted", "WorldLens", "Cross-view Consistency", "projection-based multi-camera object support", "CONCEPTUAL_ADAPTATION", "worldlens_metrics.json"),
    ("worldlens_object_persistence_adapted", "WorldLens", "Object Persistence", "matched track survival", "CONCEPTUAL_ADAPTATION", "worldlens_metrics.json"),
    ("submission_ref_precision", "Task fidelity", "Precision", "visible dynamic image detections matched to submitted instruction boxes", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "task_fidelity_metrics.json"),
    ("submission_ref_recall", "Task fidelity", "Recall", "expected-visible submitted instruction boxes recovered from images", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "task_fidelity_metrics.json"),
    ("submission_ref_idf1", "Task fidelity", "IDF1", "identity F1 using submitted agent identity as instruction reference", "DIRECT_WITH_PROTOCOL_DIFFERENCE", "task_fidelity_metrics.json"),
    ("submission_ref_hota_adapted", "Task fidelity", "HOTA", "geometric mean of adapted detection and association accuracy", "CONCEPTUAL_ADAPTATION", "task_fidelity_metrics.json"),
    ("trajectory_distribution_js", "DrivingGen related", "not FTD", "common-bin Jensen-Shannon divergence", "RELATED_NOT_EQUIVALENT", "protocol_d_distribution_reference/metrics.json"),
    ("trajectory_distribution_wasserstein", "DrivingGen related", "not FTD", "one-dimensional Wasserstein distance", "RELATED_NOT_EQUIVALENT", "protocol_d_distribution_reference/metrics.json"),
    ("trajectory_distribution_ks", "DrivingGen related", "not FTD", "two-sample KS statistic", "RELATED_NOT_EQUIVALENT", "protocol_d_distribution_reference/metrics.json"),
    ("trajectory_distribution_mmd", "DrivingGen related", "not FTD", "RBF maximum mean discrepancy", "CONCEPTUAL_ADAPTATION", "protocol_d_distribution_reference/metrics.json"),
    ("standard_nuscenes_mAP", "BEVFormer/nuScenes", "standard mAP", "official paired-GT nuScenes evaluator", "NOT_AVAILABLE_DATA", "protocol_g_paired_gt/metrics.json"),
    ("standard_NDS", "BEVFormer/nuScenes", "standard NDS", "official paired-GT nuScenes evaluator", "NOT_AVAILABLE_DATA", "protocol_g_paired_gt/metrics.json"),
    ("standard_waymo_AP", "BEVFormer/Waymo", "standard Waymo AP/APH", "official Waymo evaluator with LEVEL labels", "NOT_AVAILABLE_DEPENDENCY", "protocol_g_paired_gt/metrics.json"),
    ("tcvv_motion_direction_accuracy_adapted", "ACT-Bench", "Motion Accuracy", "local projected instruction motion direction agrees with bounded RGB evidence", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_stop_turn_completion_adapted", "ACT-Bench", "Action Completion", "short-window stop and turn event completion from local visual evidence", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_local_projection_alignment_adapted", "ACT-Bench", "Trajectory Alignment", "pixel-space local projection flow alignment rather than Ego world-trajectory ADE", "RELATED_NOT_EQUIVALENT", "tcvv_metrics.json"),
    ("tcvv_agent_missing_adapted", "DrivingGen", "Agent Missing", "missing conditional local visual evidence in expected-visible projected regions", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_local_trajectory_consistency_adapted", "DrivingGen", "Trajectory Consistency", "bounded-window projected flow magnitude and direction consistency", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_temporal_consistency_adapted", "DrivingGen", "Temporal Consistency", "entry exit and short-window visual support timing", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_agent_consistency_adapted", "DrivingGen", "Agent Consistency", "conditional persistent and entrant support without open-world re-identification", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_entry_exit_consistency_adapted", "DrivingGen", "Entry/Exit Consistency", "projected camera appearance and disappearance timing", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_cross_view_support_adapted", "WorldLens", "Cross-view Consistency", "simultaneous conditional support across expected cameras", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_temporal_object_consistency_adapted", "WorldLens", "Temporal Consistency", "local-window flow and support consistency without long-track propagation", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_object_persistence_adapted", "WorldLens", "Object Persistence", "persistent-agent conditional support duration", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
    ("tcvv_geometric_consistency_adapted", "WorldLens", "Geometric Consistency", "normalized pixel projection and flow error", "CONCEPTUAL_ADAPTATION", "tcvv_metrics.json"),
]


def build_registry(values: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    values = values or {}; records = []
    for name, paper, original, definition, level, path in METRICS:
        assert level in LEVELS
        value = values.get(name, NOT_AVAILABLE)
        paired = name.startswith("standard_"); distribution = name.startswith("trajectory_distribution") or name == "official_ftd"
        tcvv = name.startswith("tcvv_")
        unavailable_reason = values.get(f"{name}_unavailable_reason")
        if value == NOT_AVAILABLE:
            if unavailable_reason is None:
                if name == "submission_ref_mAAE": unavailable_reason = "missing_attributes"
                elif name == "submission_ref_NDS_adapted": unavailable_reason = "missing_mAAE_no_reweighting_allowed"
                elif name == "official_ftd": unavailable_reason = "missing_official_checkpoint_or_insufficient_sample"
                elif paired: unavailable_reason = "requires_paired_gt"
                elif distribution: unavailable_reason = "requires_distribution_reference_or_dependency"
                else: unavailable_reason = "not_run"
        if tcvv: code_path = "driveaudit_eval/research/consistency/trajectory_verification/metrics/"
        elif distribution: code_path = "driveaudit_eval/consistency/image_trajectory_consistency/distribution_reference_protocol.py"
        elif paired: code_path = "driveaudit_eval/consistency/image_trajectory_consistency/protocol_runner.py"
        else: code_path = "driveaudit_eval/consistency/image_trajectory_consistency/metrics/" + path.replace("_metrics.json", "_metrics.py")
        records.append({"metric": name, "metric_name": name, "current_name": name, "paper": paper,
                        "original_definition": definition, "paper_original_name": original,
                        "paper_definition": definition, "reference_semantics": "paired_dataset_ground_truth" if paired else
                        ("real_reference_distribution" if distribution else
                         ("submitted_trajectory_as_visual_expectation" if tcvv else "submitted_trajectory_as_instruction_reference")),
                        "implementation_level": level, "protocol_s_available": not paired and not distribution and not tcvv,
                        "protocol_d_available": distribution, "protocol_g_available": paired,
                        "protocol_s_status": "available" if not paired and not distribution and not tcvv and value != NOT_AVAILABLE else "not_available",
                        "protocol_d_status": "available" if distribution and value != NOT_AVAILABLE else "not_available",
                        "protocol_g_status": "available" if paired and value != NOT_AVAILABLE else "not_available",
                        "requires_paired_gt": paired, "requires_reference_distribution": distribution,
                        "requires_official_checkpoint": name in {"official_ftd"},
                        "protocol_tcvv_available": tcvv,
                        "protocol_tcvv_status": "available" if tcvv and value != NOT_AVAILABLE else "not_available",
                        "code_path": code_path,
                        "result_key": name, "current_status": "available" if value != NOT_AVAILABLE else "not_available",
                        "current_value": value, "sample_count": values.get(f"{name}_sample_count", NOT_AVAILABLE),
                        "unavailable_reason": unavailable_reason})
    return records

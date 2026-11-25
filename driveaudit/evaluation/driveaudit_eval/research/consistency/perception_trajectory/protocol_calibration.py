from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE


def _ratio(a: Any, b: Any) -> Any:
    return float(a) / float(b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else NOT_AVAILABLE


def _difference(a: Any, b: Any) -> Any:
    return float(a) - float(b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else NOT_AVAILABLE


def coverage_diagnostics(metrics: dict[str, Any], detector_range_m: float = 51.2,
                         miss_penalties: tuple[float, ...] = (2.0, 4.0, 8.0)) -> dict[str, Any]:
    presence, geometry = metrics["presence"], metrics["geometry"]
    expected = int(presence["expected_visible_agent_frames"])
    per_track = geometry.get("per_track", [])
    matched = sum(int(row["common_visible_frames"]) for row in per_track)
    error_sum = sum(float(row["bev_center_ade"]) * int(row["common_visible_frames"]) for row in per_track)
    visible_tracks = sum(row.get("match_status") != "not_observable" and row.get("submission_agent_id") is not None
                         for row in metrics["mapping"])
    matched_tracks = sum(row.get("match_status", "").startswith("matched") for row in metrics["mapping"])
    missed = max(0, expected - matched)
    penalties = [*miss_penalties, detector_range_m]
    ratio = matched / expected if expected else NOT_AVAILABLE
    return {"matched_subset_ade": geometry["bev_center_ade"],
            "matched_coverage": ratio, "matched_agent_frame_coverage": ratio,
            "matched_visible_agent_frames": matched, "expected_visible_agent_frames": expected,
            "missed_visible_agent_frames": missed,
            "track_match_coverage": matched_tracks / visible_tracks if visible_tracks else NOT_AVAILABLE,
            "matched_submitted_tracks": matched_tracks, "visible_submitted_tracks": visible_tracks,
            "coverage_aware_error": {str(float(penalty)): (error_sum + penalty * missed) / expected
                                     if expected else NOT_AVAILABLE for penalty in penalties},
            "recall_adjusted_ade": float(geometry["bev_center_ade"]) / max(float(presence["visible_agent_recall"]), 1e-9)
            if isinstance(geometry["bev_center_ade"], (int, float)) and isinstance(presence["visible_agent_recall"], (int, float))
            else NOT_AVAILABLE,
            "standard_metric": False, "diagnostic_only": True}


def relative_metrics(gt: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    gt_coverage, generated_coverage = coverage_diagnostics(gt), coverage_diagnostics(generated)
    gd, rd = generated["detection"], gt["detection"]
    gi, ri = generated["identity"], gt["identity"]
    return {"recall_retention": _ratio(gd["visible_detection_recall"], rd["visible_detection_recall"]),
            "precision_retention": _ratio(gd["visible_detection_precision"], rd["visible_detection_precision"]),
            "idf1_retention": _ratio(gi["idf1_like"], ri["idf1_like"]),
            "track_completeness_retention": _ratio(gi["track_completeness"], ri["track_completeness"]),
            "recall_drop": _difference(rd["visible_detection_recall"], gd["visible_detection_recall"]),
            "precision_drop": _difference(rd["visible_detection_precision"], gd["visible_detection_precision"]),
            "idf1_drop": _difference(ri["idf1_like"], gi["idf1_like"]),
            "fragmentation_increase": _difference(gi["fragmentation_count"], ri["fragmentation_count"]),
            "center_error_increase": _difference(gd["center_error"]["mean"], rd["center_error"]["mean"]),
            "heading_error_increase": _difference(generated["geometry"]["heading_mae"], gt["geometry"]["heading_mae"]),
            "speed_error_increase": _difference(generated["motion"]["speed_mae"], gt["motion"]["speed_mae"]),
            "matched_coverage_drop": _difference(gt_coverage["matched_coverage"], generated_coverage["matched_coverage"]),
            "track_match_coverage_drop": _difference(gt_coverage["track_match_coverage"], generated_coverage["track_match_coverage"]),
            "gt_coverage": gt_coverage, "generated_coverage": generated_coverage,
            "ade_difference": _difference(generated["geometry"]["bev_center_ade"], gt["geometry"]["bev_center_ade"])}


def gate4(gt: dict[str, Any], generated: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    relative = relative_metrics(gt, generated)
    absolute_limits = config["extractor_absolute_quality_warning"]
    absolute_checks = {
        "gt_recall": (gt["detection"]["visible_detection_recall"], absolute_limits["minimum_gt_recall"], "minimum"),
        "gt_precision": (gt["detection"]["visible_detection_precision"], absolute_limits["minimum_gt_precision"], "minimum"),
        "gt_idf1": (gt["identity"]["idf1_like"], absolute_limits["minimum_gt_idf1"], "minimum"),
        "gt_fragmentation": (gt["identity"]["fragmentation_count"], absolute_limits["maximum_gt_fragmentation"], "maximum"),
        "gt_speed_mae": (gt["motion"]["speed_mae"], absolute_limits["maximum_gt_speed_mae_mps"], "maximum"),
    }
    def checked(values: dict[str, tuple[Any, float, str]]) -> tuple[dict[str, Any], list[str]]:
        details = {}; failures = []
        for name, (value, threshold, direction) in values.items():
            passed = isinstance(value, (int, float)) and (value >= threshold if direction == "minimum" else value <= threshold)
            details[name] = {"value": value, direction: threshold, "passed": passed}
            if not passed: failures.append(name)
        return details, failures
    absolute_details, absolute_failures = checked(absolute_checks)
    limits = config["generated_relative_consistency"]
    relative_checks = {
        "recall_retention": (relative["recall_retention"], limits["minimum_recall_retention"], "minimum"),
        "precision_retention": (relative["precision_retention"], limits["minimum_precision_retention"], "minimum"),
        "idf1_retention": (relative["idf1_retention"], limits["minimum_idf1_retention"], "minimum"),
        "track_completeness_retention": (relative["track_completeness_retention"], limits["minimum_track_completeness_retention"], "minimum"),
        "fragmentation_increase": (relative["fragmentation_increase"], limits["maximum_fragmentation_increase"], "maximum"),
        "speed_error_increase": (relative["speed_error_increase"], limits["maximum_speed_error_increase_mps"], "maximum"),
        "center_error_increase": (relative["center_error_increase"], limits["maximum_center_error_increase_m"], "maximum"),
        "heading_error_increase": (relative["heading_error_increase"], limits["maximum_heading_error_increase_rad"], "maximum"),
        "matched_coverage_drop": (relative["matched_coverage_drop"], limits["maximum_matched_coverage_drop"], "maximum"),
    }
    relative_details, relative_failures = checked(relative_checks)
    warning_limits = config["coverage_aware_warning"]
    low_error = isinstance(relative["ade_difference"], (int, float)) and relative["ade_difference"] <= warning_limits["maximum_ade_improvement_m"]
    low_coverage = ((isinstance(relative["recall_drop"], (int, float)) and relative["recall_drop"] >= warning_limits["minimum_recall_drop"])
                    or (isinstance(relative["track_match_coverage_drop"], (int, float)) and
                        relative["track_match_coverage_drop"] >= warning_limits["minimum_track_coverage_drop"]))
    return {"relative_metrics": relative,
            "gate_4a_extractor_absolute_quality_warning": {
                "status": "warning" if absolute_failures else "passed", "failures": absolute_failures,
                "checks": absolute_details, "penalizes_generated_submission": False,
                "provisional_thresholds": absolute_limits["provisional_thresholds"]},
            "gate_4b_generated_relative_consistency": {
                "status": "failed_relative_degradation" if relative_failures else "passed",
                "failures": relative_failures, "checks": relative_details,
                "provisional_thresholds": limits["provisional_thresholds"]},
            "gate_4c_coverage_aware_warning": {
                "status": "low_error_low_coverage_warning" if low_error and low_coverage else "passed",
                "matched_subset_ade_improved": low_error, "coverage_degraded": low_coverage,
                "provisional_thresholds": warning_limits["provisional_thresholds"]}}

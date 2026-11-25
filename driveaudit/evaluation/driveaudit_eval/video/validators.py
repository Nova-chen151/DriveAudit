from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from driveaudit_eval.video.utils import FORBIDDEN_OUTPUT_KEYS


METRIC_CONFIDENCE_VALUES = {"normal", "medium", "low", "low_sample", "skipped", "disabled", None}
METRIC_VALIDITY_VALUES = {
    "valid",
    "valid_low_sample",
    "invalid_missing_dependency",
    "invalid_missing_weight",
    "invalid_calibration_or_matching",
    "invalid_feature_dim",
    "disabled",
    "skipped",
    None,
}
COMPARISON_VALIDITY_VALUES = {
    "distribution_level",
    "condition_consistency",
    "history_future_condition_consistency",
    "temporal_self_consistency",
    "multiview_self_consistency",
    "object_stability",
    "invalid",
    None,
}


def _walk_keys(value: Any) -> List[str]:
    keys = []
    if isinstance(value, dict):
        keys.extend(value.keys())
        for item in value.values():
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def validate_summary(summary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    forbidden = set(_walk_keys(summary)).intersection(FORBIDDEN_OUTPUT_KEYS)
    if forbidden:
        errors.append(f"forbidden score keys present: {sorted(forbidden)}")
    for metric_name, metric in summary.items():
        if not isinstance(metric, dict) or metric_name in {"config", "dataset_statistics"}:
            continue
        if metric.get("enabled") is False and metric.get("skipped") is not True:
            errors.append(f"{metric_name}: disabled metric must have skipped=true")
        if metric.get("metric_confidence") not in METRIC_CONFIDENCE_VALUES:
            errors.append(f"{metric_name}: invalid metric_confidence={metric.get('metric_confidence')}")
        if metric.get("metric_validity") not in METRIC_VALIDITY_VALUES:
            errors.append(f"{metric_name}: invalid metric_validity={metric.get('metric_validity')}")
        if metric.get("comparison_validity") not in COMPARISON_VALIDITY_VALUES:
            errors.append(f"{metric_name}: invalid comparison_validity={metric.get('comparison_validity')}")
        if metric_name in {"fid", "fvd"} and not metric.get("skipped") and not summary.get("config", {}).get("real_reference"):
            errors.append(f"{metric_name}: missing real_reference config")
        if metric_name == "fvd" and metric.get("fvd_standard") is True:
            from driveaudit_eval.video.distribution import FVD_FEATURE_DIM, FVD_PROTOCOL, FVD_OUTPUT_KIND, FVD_REFERENCE_WEIGHTS_SHA256
            if metric.get("i3d_feature_dim") != FVD_FEATURE_DIM:
                errors.append("fvd: standard FVD requires 400-dimensional Kinetics logits")
            if metric.get("feature_kind") != FVD_OUTPUT_KIND:
                errors.append("fvd: standard FVD requires pre-softmax logits")
            if metric.get("protocol") != FVD_PROTOCOL:
                errors.append("fvd: standard FVD protocol mismatch")
            if metric.get("weights_sha256") != FVD_REFERENCE_WEIGHTS_SHA256:
                errors.append("fvd: standard FVD reference checkpoint SHA256 mismatch")
            for key in ("i3d_feature_dim", "clip_len", "num_real_clips", "num_gen_clips"):
                if key not in metric:
                    errors.append(f"fvd: fvd_standard=true requires {key}")
        if metric_name == "epipolar" and metric.get("metric_validity") == "valid":
            status = metric.get("epipolar_metric_status")
            if status == "invalid_due_to_calibration_or_matching":
                errors.append("epipolar: valid metric cannot have bad history sanity status")
    return errors


def validate_output_dir(output_dir: Path) -> List[str]:
    errors: List[str] = []
    summary_path = output_dir / "summary_by_metric.json"
    if not summary_path.exists() and (output_dir / "results.json").exists():
        summary_path = output_dir / "results.json"
    if not summary_path.exists():
        return [f"missing {summary_path}"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{summary_path.name} is not readable: {exc}"]
    errors.extend(validate_summary(summary))
    for csv_path in output_dir.glob("*.csv"):
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, None)
                if not header:
                    raise ValueError("CSV header is missing")
                for row in reader:
                    if row and len(row) != len(header):
                        raise ValueError(f"CSV row {reader.line_num} has {len(row)} fields; expected {len(header)}")
        except Exception as exc:
            errors.append(f"{csv_path.name} is not readable: {exc}")
    return errors


def metric_requirements() -> Dict[str, Dict[str, Any]]:
    return {
        "fid": {"future_gt": False, "real_reference": True, "metadata": False, "pretrained_model": True},
        "fvd": {"future_gt": False, "real_reference": True, "metadata": False, "pretrained_model": True},
        "boundary": {"future_gt": False, "real_reference": False, "metadata": False, "pretrained_model": True},
        "temporal_warp": {"future_gt": False, "real_reference": False, "metadata": False, "pretrained_model": False},
        "history_future_continuity": {
            "future_gt": False,
            "real_reference": False,
            "metadata": False,
            "pretrained_model": False,
            "history_input": True,
        },
        "flicker": {"future_gt": False, "real_reference": False, "metadata": False, "pretrained_model": False},
        "loftr": {"future_gt": False, "real_reference": False, "metadata": False, "pretrained_model": True},
        "epipolar": {"future_gt": False, "real_reference": False, "metadata": True, "pretrained_model": True},
        "tracking": {"future_gt": False, "real_reference": False, "metadata": False, "pretrained_model": True},
    }


def build_metric_audit(summary: Dict[str, Any]) -> Dict[str, Any]:
    stats = summary.get("dataset_statistics", {})
    metrics = {}
    requirements = metric_requirements()
    for name, req in requirements.items():
        metric = summary.get(name, {})
        metrics[name] = {
            "backend": metric.get("backend"),
            "metric_confidence": metric.get("metric_confidence"),
            "metric_validity": metric.get("metric_validity"),
            "comparison_validity": metric.get("comparison_validity"),
            "requires": req,
            "paper_comparable": "only_val_scale" if name in {"fid", "fvd"} else "no",
            "paper_comparison_reason": (
                "FID/FVD are dataset-level distribution metrics; use validation-scale samples for paper tables."
                if name in {"fid", "fvd"}
                else "History-future continuity is a condition-consistency diagnostic, not a paper leaderboard metric."
                if name == "history_future_continuity"
                else "Diagnostic self-consistency metric, not a paper leaderboard metric."
            ),
        }
    return {
        "data_scale": {
            "num_scenes": stats.get("num_scenes"),
            "num_cameras": len(summary.get("config", {}).get("dataset", {}).get("cameras", [])),
            "history_frame_counts": stats.get("history_frame_counts"),
            "future_frame_counts": stats.get("future_frame_counts"),
            "num_real_reference_images": summary.get("fid", {}).get("fid_num_real_images"),
            "num_generated_images": summary.get("fid", {}).get("fid_num_gen_images"),
            "num_real_clips": summary.get("fvd", {}).get("num_real_clips"),
            "num_generated_clips": summary.get("fvd", {}).get("num_gen_clips"),
        },
        "metrics": metrics,
        "notes": [
            "No overall, weighted, combined, rank, final, normalized, or total score is produced.",
            "FID/FVD must be computed once on global dataset-level real/fake feature distributions.",
        ],
    }


def audit_markdown(audit: Dict[str, Any]) -> str:
    lines = ["# Metric Audit Report", "", "## Data Scale", ""]
    for key, value in audit.get("data_scale", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Metrics", ""])
    for name, metric in audit.get("metrics", {}).items():
        lines.append(f"### {name}")
        lines.append(f"- backend: {metric.get('backend')}")
        lines.append(f"- metric_confidence: {metric.get('metric_confidence')}")
        lines.append(f"- metric_validity: {metric.get('metric_validity')}")
        lines.append(f"- comparison_validity: {metric.get('comparison_validity')}")
        lines.append(f"- paper_comparable: {metric.get('paper_comparable')}")
        lines.append(f"- reason: {metric.get('paper_comparison_reason')}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in audit.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)

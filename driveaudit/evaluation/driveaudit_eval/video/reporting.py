from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from driveaudit_eval.video.runtime_state import RuntimeState
from driveaudit_eval.video.utils import write_csv

def _markdown(summary: Dict[str, Any]) -> str:
    rows = [
        ("FID", summary.get("fid", {}).get("fid"), summary.get("fid", {})),
        ("FVD", summary.get("fvd", {}).get("fvd"), summary.get("fvd", {})),
        (
            "Temporal Consistency",
            summary.get("temporal_consistency", {}).get("warp_error_mean"),
            summary.get("temporal_consistency", {}),
        ),
        (
            "History-Future Continuity",
            summary.get("history_future_continuity", {}).get("history_future_warp_error_mean"),
            summary.get("history_future_continuity", {}),
        ),
        (
            "Multi-view Consistency",
            summary.get("multi_view_consistency", {}).get("loftr_ransac_inlier_ratio_mean"),
            summary.get("multi_view_consistency", {}),
        ),
        (
            "Geometry Consistency",
            summary.get("geometry_consistency", {}).get("epipolar_error_median_px"),
            summary.get("geometry_consistency", {}),
        ),
    ]
    lines = [
        "# Benchmark Evaluation Report",
        "",
        "Protocol: fully deterministic dataset-level evaluation. No overall or weighted score is produced.",
        "",
        "| Metric | Value | Backend | Validity | Confidence |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, value, metric in rows:
        value_text = "null" if value is None else f"{float(value):.6g}"
        lines.append(
            f"| {name} | {value_text} | {metric.get('backend')} | "
            f"{metric.get('metric_validity')} | {metric.get('metric_confidence')} |"
        )
    sampling = summary.get("sampling", {})
    manifest = summary.get("manifest", {})
    lines.extend(
        [
            "",
            "## Locks",
            "",
            f"- dataset_version_hash: {manifest.get('dataset_version_hash')}",
            f"- scene_order_hash: {manifest.get('scene_order_hash')}",
            f"- execution_graph_hash: {summary.get('benchmark', {}).get('execution_graph_hash')}",
            "",
            "## Sampling",
            "",
            f"- seed: {sampling.get('seed')}",
            f"- cameras: {sampling.get('camera_set')}",
            f"- frames_per_scene: {sampling.get('frames_per_scene')}",
            f"- clip_length: {sampling.get('clip_length')}",
            f"- stride: {sampling.get('stride')}",
            f"- real/fake images: {sampling.get('num_real_images')} / {sampling.get('num_fake_images')}",
            f"- real/fake clips: {sampling.get('num_real_clips')} / {sampling.get('num_fake_clips')}",
            "",
            "## Notes",
            "",
            "- FID uses global TF-compatible InceptionV3 pool3 distributions with the official FID checkpoint and preprocessing.",
            "- FVD uses global Kinetics-400 I3D pre-softmax logits (400 dimensions) and reference float bilinear preprocessing.",
            "- Temporal, history-future, multi-view, and geometry metrics are diagnostics on the same locked sampling manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_tables(output_dir: Path, state: RuntimeState, summary: Dict[str, Any]) -> None:
    empty_headers = {
        "scene_metrics": ["scene_id"],
        "camera_metrics": ["scene_id", "camera"],
        "time_window_metrics": ["scene_id", "camera", "camera_pair", "time_window"],
        "frame_metrics": ["scene_id", "camera", "frame_id"],
        "camera_pair_metrics": ["scene_id", "frame_id", "camera_pair"],
        "track_metrics": ["scene_id", "camera", "track_id"],
    }
    metric_rows = []
    for name in (
        "fid",
        "fvd",
        "temporal_consistency",
        "history_future_continuity",
        "multi_view_consistency",
        "geometry_consistency",
    ):
        metric = summary.get(name, {})
        value = (
            metric.get("fid")
            if name == "fid"
            else metric.get("fvd")
            if name == "fvd"
            else metric.get("warp_error_mean")
            if name == "temporal_consistency"
            else metric.get("history_future_warp_error_mean")
            if name == "history_future_continuity"
            else metric.get("loftr_ransac_inlier_ratio_mean")
            if name == "multi_view_consistency"
            else metric.get("epipolar_error_median_px")
        )
        metric_rows.append(
            {
                "metric": name,
                "backend": metric.get("backend"),
                "value": value,
                "metric_validity": metric.get("metric_validity"),
                "metric_confidence": metric.get("metric_confidence"),
                "comparison_validity": metric.get("comparison_validity"),
            }
        )
    write_csv(output_dir / "metric_summary.csv", metric_rows)
    for table, rows in state.rows.items():
        write_csv(output_dir / f"{table}.csv", rows, empty_headers.get(table) if not rows else None)

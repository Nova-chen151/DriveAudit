from __future__ import annotations
from pathlib import Path
from typing import Any
from driveaudit_eval.consistency.mtpc.aggregate import _history_stage, _evaluate
from driveaudit_eval.consistency.mtpc.common import write_csv, write_json

def _ablation_summary(metrics: dict[str, Any], tracked: dict[str, Any]) -> dict[str, Any]:
    return {"support_rate": metrics["presence"]["persistent_agent_visual_support_rate"],
            "normalized_center_error": metrics["projection_geometry"]["normalized_projected_center_error"],
            "flow_epe_px": metrics["image_motion"]["flow_endpoint_error_px"],
            "identity_persistence": metrics["identity"]["identity_persistence"], "fragmentation": metrics["identity"]["fragmentation"],
            "mean_frame_seconds": tracked["mean_frame_seconds"], "gpu_used": tracked["gpu_used"], "peak_gpu_memory_mib": tracked["peak_gpu_memory_mib"]}


def _run_ablations(config: dict[str, Any], pair: dict[str, Any], output: Path, projections: list[dict[str, Any]],
                   motion_rows: list[dict[str, Any]], formal_history: dict[str, Any], formal_tracked: dict[str, Any],
                   formal_metrics: dict[str, Any]) -> dict[str, Any]:
    backend_rows, backend_cache = [], {}
    for mode in config["ablations"]["backend_modes"]:
        if mode == config["tracking"]["formal_mode"]:
            history, tracked, metrics = formal_history, formal_tracked, formal_metrics
        else:
            history, tracked, _ = _history_stage(config, pair, output, 10, mode, 90, output / f"ablations/backend/{mode}/tracks")
            metrics = _evaluate(config, projections, motion_rows, tracked["observations"], history)
        row = {"experiment": "backend", "setting": mode, **_ablation_summary(metrics, tracked)}; backend_rows.append(row); backend_cache[mode] = (history, tracked, metrics)
    history_rows = []
    for length in config["ablations"]["history_lengths"]:
        if length == 10 and "point" in backend_cache: history, tracked, metrics = backend_cache["point"]
        else:
            history, tracked, _ = _history_stage(config, pair, output, int(length), "point", 90, output / f"ablations/history_length/{length}/tracks")
            metrics = _evaluate(config, projections, motion_rows, tracked["observations"], history)
        history_rows.append({"experiment": "history_length", "setting": int(length), **_ablation_summary(metrics, tracked)})
    view_rows = []
    for name, cameras in config["ablations"]["view_sets"].items():
        selected = set(cameras); p = [row for row in projections if row["camera_name"] in selected]
        m = [row for row in motion_rows if row["camera_name"] in selected]; v = [row for row in formal_tracked["observations"] if row["camera_name"] in selected]
        metrics = _evaluate(config, p, m, v, formal_history)
        view_rows.append({"experiment": "view_set", "setting": name, **_ablation_summary(metrics, formal_tracked)})
    conceptual = {"mtpc_scope": "history_anchored pixel projection and optical-flow consistency",
                  "bevformer_scope": "open-world perception utility diagnostic", "scores_directly_comparable": False}
    result = {"status": "completed_low_sample", "backend": backend_rows, "history_length": history_rows, "view_set": view_rows,
              "mtpc_vs_bevformer": conceptual}
    write_json(output / "ablations/summary.json", result); write_csv(output / "ablations/backend.csv", backend_rows)
    write_csv(output / "ablations/history_length.csv", history_rows); write_csv(output / "ablations/view_set.csv", view_rows); return result


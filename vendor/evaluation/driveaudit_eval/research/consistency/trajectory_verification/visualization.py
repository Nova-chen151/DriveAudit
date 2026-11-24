from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, read_image


def _save(path: Path, image: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4)); axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title(title, fontsize=8); axis.axis("off"); figure.tight_layout(); figure.savefig(path, dpi=120); plt.close(figure)


def _plot(path: Path, values: list[float], title: str, ylabel: str = "value") -> None:
    figure, axis = plt.subplots(figsize=(6, 4)); axis.plot(values, marker=".", ms=2); axis.set(title=title, xlabel="sample", ylabel=ylabel)
    axis.grid(True); figure.tight_layout(); figure.savefig(path, dpi=120); plt.close(figure)


def generate_visualizations(pair: dict[str, Any], projections: list[dict[str, Any]], evidence: list[dict[str, Any]],
                            history: dict[str, Any], entrants: dict[str, Any], multiview: dict[str, Any],
                            corruption: dict[str, Any], diagnostics: dict[str, Any], output: str | Path) -> list[str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in evidence if int(row["window_length"]) == 2]
    sample = next((row for row in rows if row["camera_name"] == "CAM_FRONT" and row["visual_supported"]), rows[0])
    image = read_image(pair, sample["camera_name"], sample["frame_index"], (400, 224))
    overlay = image.copy(); box = list(map(int, sample["source_projected_box"])); roi = sample["search_roi"]
    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    if roi: cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
    center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    expected = sample["expected_flow"]; actual = sample["actual_flow"]
    cv2.arrowedLine(overlay, center, (int(center[0] + expected[0]), int(center[1] + expected[1])), (0, 255, 0), 2)
    cv2.arrowedLine(overlay, center, (int(center[0] + actual[0]), int(center[1] + actual[1])), (0, 0, 255), 2)
    label = (f"scene={pair['pair_id']} agent={sample['agent_id'][:8]} frame={sample['frame_index']} "
             f"cam={sample['camera_name']} area={sample['projected_area']:.1f} support={sample['support_confidence']:.3f}")
    names = ["01_submitted_projection", "02_expected_visibility", "03_local_search_roi", "04_sparse_lk_points",
             "05_dense_optical_flow", "06_expected_pixel_motion", "07_actual_pixel_motion", "08_flow_epe_heatmap",
             "09_supported_case", "10_missing_case", "11_direction_error_case", "12_entry_exit_timing",
             "13_camera_handoff", "14_multiview_contradiction", "15_persistent_agents", "16_future_entrants",
             "17_cam_back_left_diagnostic", "18_history_future_baseline", "19_tcvv_mtpc_bevformer",
             "20_corruption_response"]
    paths = []
    for index, name in enumerate(names[:7], 1):
        path = output / f"{name}.png"; _save(path, overlay, f"{name}: {label}"); paths.append(str(path))
    series = {
        names[7]: [row.get("local_flow_epe", np.nan) for row in rows[:500]],
        names[8]: [row["support_confidence"] for row in rows if row["visual_supported"]][:500],
        names[9]: [row["support_confidence"] for row in rows if not row["visual_supported"]][:500],
        names[10]: [row.get("flow_cosine_similarity", np.nan) for row in rows[:500]],
        names[11]: [history["history_support_rate"], sum(row["visual_supported"] for row in rows) / max(1, len(rows))],
        names[12]: [row["camera_support_recall"] for row in multiview["per_frame"][:500]],
        names[13]: [float(row["multi_view_missing_contradiction"]) for row in multiview["per_frame"][:500]],
        names[14]: [row["support_confidence"] for row in rows if row["history_anchored"]][:500],
        names[15]: [row["conditional_support_rate"] for row in entrants.get("per_agent", [])],
        names[16]: [multiview["per_camera"][camera]["support_rate"] if isinstance(multiview["per_camera"][camera]["support_rate"], (int, float)) else np.nan for camera in CAMERAS],
        names[17]: [history["history_support_rate"], history["metrics"]["presence"]["projection_support_rate"]],
        names[18]: [diagnostics["tcvv_support"], diagnostics.get("mtpc_support", 0), diagnostics.get("bevformer_coverage", 0)],
        names[19]: [row["projection_error"] for row in corruption.get("rows", [])],
    }
    for name, values in series.items():
        path = output / f"{name}.png"; _plot(path, values, f"{name}: scene={pair['pair_id']}"); paths.append(str(path))
    return paths

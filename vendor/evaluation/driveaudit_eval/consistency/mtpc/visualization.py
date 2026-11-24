from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from driveaudit_eval.consistency.mtpc.common import CAMERAS, image_path
from driveaudit_eval.video.utils import read_image_bgr


def _save_image(path: Path, image: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4)); axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axis.set_title(title); axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def _plot(path: Path, values: list[float], title: str, ylabel: str = "value") -> None:
    figure, axis = plt.subplots(figsize=(6, 4)); axis.plot(values, marker="o", ms=2); axis.set(title=title, xlabel="sample", ylabel=ylabel); axis.grid(True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def generate_visualizations(pair: dict[str, Any], history_projection: list[dict[str, Any]], projections: list[dict[str, Any]],
                            visual: list[dict[str, Any]], metrics: dict[str, Any], entrants: dict[str, Any],
                            bev: dict[str, Any], corruption: dict[str, Any] | None, output: str | Path) -> list[str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True); paths = []
    camera, history_frame, future_frame = "CAM_FRONT", 9, 10
    history = cv2.resize(read_image_bgr(image_path(pair, camera, history_frame)), (400, 224), interpolation=cv2.INTER_AREA)
    future = read_image_bgr(image_path(pair, camera, future_frame))
    for row in history_projection:
        if row["camera_name"] == camera and row["frame_index"] == history_frame and row.get("expected_visible"):
            x1, y1, x2, y2 = map(int, row["clipped_box_2d"]); cv2.rectangle(history, (x1, y1), (x2, y2), (0, 255, 0), 1)
    _save_image(output / "01_history_projection.png", history, "History 3D boxes projected to CAM_FRONT"); paths.append("01_history_projection.png")
    _save_image(output / "02_history_masks.png", history, "History projected masks (green)"); paths.append("02_history_masks.png")
    _save_image(output / "03_history_points.png", history, "History anchored point regions"); paths.append("03_history_points.png")
    overlay = future.copy()
    for row in visual:
        if row["camera_name"] == camera and row["frame_index"] == future_frame and row["visible"]:
            x1, y1, x2, y2 = map(int, row["bbox_2d"]); cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
            for point in row["tracked_points"]: cv2.circle(overlay, tuple(map(int, point)), 1, (0, 255, 255), -1)
    _save_image(output / "04_future_visual_tracks.png", overlay, "Future image-only visual tracks"); paths.append("04_future_visual_tracks.png")
    expected = future.copy()
    for row in projections:
        if row["camera_name"] == camera and row["frame_index"] == future_frame and row["expected_visible"]:
            x1, y1, x2, y2 = map(int, row["clipped_box_2d"]); cv2.rectangle(expected, (x1, y1), (x2, y2), (0, 255, 0), 1)
    _save_image(output / "05_submitted_projection.png", expected, "Submitted trajectory projection"); paths.append("05_submitted_projection.png")
    blended = cv2.addWeighted(overlay, .6, expected, .4, 0); _save_image(output / "06_actual_expected_centers.png", blended, "Visual track (blue) vs projection (green)"); paths.append("06_actual_expected_centers.png")
    _save_image(output / "07_actual_expected_flow.png", overlay, "Tracked point optical flow support"); paths.append("07_actual_expected_flow.png")
    geometry_rows = metrics["projection_geometry"]["per_frame"]; _plot(output / "08_mask_iou.png", [row["mask_iou"] for row in geometry_rows[:300]], "Mask/Projection IoU"); paths.append("08_mask_iou.png")
    _plot(output / "09_missing_agents.png", [int(not row["visual_supported"]) for row in metrics["presence"]["per_frame"][:500]], "Persistent-agent missing evidence"); paths.append("09_missing_agents.png")
    _plot(output / "10_identity_stability.png", [row.get("appearance_similarity", 0) for row in visual[:500]], "Appearance stability / identity diagnostic"); paths.append("10_identity_stability.png")
    support = metrics["multiview"]["per_frame"]; _plot(output / "11_camera_handoff.png", [row["actual_camera_support_count"] for row in support[:400]], "Camera support and handoff"); paths.append("11_camera_handoff.png")
    matrix = np.asarray([[metrics["multiview"]["per_camera"][camera]["support_rate"] if isinstance(metrics["multiview"]["per_camera"][camera]["support_rate"], (int, float)) else np.nan for camera in CAMERAS]])
    figure, axis = plt.subplots(figsize=(9, 2)); image = axis.imshow(matrix, vmin=0, vmax=1); axis.set_xticks(range(6), CAMERAS, rotation=25); axis.set_yticks([]); figure.colorbar(image); figure.tight_layout(); figure.savefig(output / "12_cross_view_support_matrix.png", dpi=130); plt.close(figure); paths.append("12_cross_view_support_matrix.png")
    temporal = metrics["temporal_events"]["per_agent_camera"]; _plot(output / "13_entry_exit_timing.png", [row["entry_time_error_s"] if isinstance(row["entry_time_error_s"], (int, float)) else np.nan for row in temporal], "Entry timing error", "seconds"); paths.append("13_entry_exit_timing.png")
    _plot(output / "14_future_entrant_verification.png", [row.get("conditional_track_survival", 0) for row in entrants.get("per_agent", [])], "Conditional future entrant survival"); paths.append("14_future_entrant_verification.png")
    figure, axis = plt.subplots(figsize=(6, 4)); axis.bar(["MTPC support", "BEV recall"], [metrics["presence"]["persistent_agent_visual_support_rate"], bev.get("matched_coverage", 0) if isinstance(bev.get("matched_coverage"), (int, float)) else 0]); axis.set_title("Different scopes: MTPC vs perception utility"); figure.tight_layout(); figure.savefig(output / "15_mtpc_bevformer_parallel.png", dpi=130); plt.close(figure); paths.append("15_mtpc_bevformer_parallel.png")
    response = corruption.get("rows", []) if corruption else []; _plot(output / "16_corruption_response.png", [row["normalized_center_error"] if isinstance(row["normalized_center_error"], (int, float)) else np.nan for row in response], "Corruption response"); paths.append("16_corruption_response.png")
    return [str(output / path) for path in paths]

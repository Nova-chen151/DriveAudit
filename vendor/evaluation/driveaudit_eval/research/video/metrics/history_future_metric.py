from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from driveaudit_eval.video.metrics import BaseMetric
from driveaudit_eval.research.video.metrics.temporal_warp_metric import FlowComputer, warp_with_flow
from driveaudit_eval.video.utils import image_to_np, mean_or_none, percentile_or_none, stats


class HistoryFutureContinuityMetric(BaseMetric):
    name = "history_future"

    def __init__(self, cfg: Dict[str, Any], state) -> None:
        super().__init__(cfg, state)
        self.rows: List[Dict[str, Any]] = []
        self.flow_computer: Optional[FlowComputer] = None
        self.backend = self.metric_cfg.get(
            "flow_backend",
            cfg.get("metrics", {}).get("temporal_warp", {}).get("flow_backend", "opencv"),
        )

    def prepare(self):
        if not self.enabled:
            return
        if self.backend in {"none", "raft", "sea_raft"}:
            if self.backend in {"raft", "sea_raft"}:
                self.warn(
                    "optical_flow_backend",
                    f"{self.backend} is not bundled; falling back to OpenCV Farneback.",
                )
                self.backend = "fallback_opencv"
            else:
                self.skip("no_flow_model_for_history_future", message="Flow backend is none.")
                return
        shared_flow = self.state.shared.get("flow_computer")
        if isinstance(shared_flow, FlowComputer) and shared_flow.backend == self.backend:
            self.flow_computer = shared_flow
        else:
            self.flow_computer = FlowComputer(self.cfg, self.state, backend=self.backend)
            self.state.shared["flow_computer"] = self.flow_computer
        self.summary.update(
            {
                "backend": self.backend,
                "flow_backend": self.backend,
                "metric_confidence": "medium" if self.backend in {"opencv", "fallback_opencv"} else "normal",
                "metric_validity": "valid",
                "comparison_validity": "history_future_condition_consistency",
                "l1_error_unit": "mean_abs_rgb_0_1",
                "warp_error_unit": self.metric_cfg.get("warp_error_unit", "mean_abs_rgb_0_1"),
                "boundary": "history_last_frame_to_generated_future_first_frame",
            }
        )

    def update(self, scene_sample):
        if not self.enabled or self.skipped:
            return
        if self.flow_computer is None:
            self.skip("no_flow_model_for_history_future", scene_id=scene_sample.scene_id)
            return
        size = (
            self.cfg["preprocess"]["resize_width"],
            self.cfg["preprocess"]["resize_height"],
        )
        for camera in scene_sample.cameras:
            history = scene_sample.history_images.get(camera, [])
            future = scene_sample.future_images.get(camera, [])
            if not history or not future:
                self.warn(
                    "missing_boundary_frames",
                    f"Need history[-1] and future[0] for {camera}.",
                    scene_sample.scene_id,
                )
                continue
            history_path = history[-1]
            future_path = future[0]
            img_h = image_to_np(history_path, size=size)
            img_f = image_to_np(future_path, size=size)
            l1 = np.abs(img_f.astype(np.float32) - img_h.astype(np.float32)) / 255.0
            flow, img_h_flow, img_f_flow = self.flow_computer.compute(
                scene_sample.scene_id,
                camera,
                -1,
                history_path,
                future_path,
            )
            warped = warp_with_flow(img_h_flow, flow)
            warp_err = np.abs(img_f_flow.astype(np.float32) - warped.astype(np.float32)) / 255.0
            flow_mag = np.linalg.norm(flow, axis=2)
            row = {
                "scene_id": scene_sample.scene_id,
                "camera": camera,
                "frame_t": -1,
                "frame_t1": 0,
                "frame_id": "history_future_boundary",
                "history_frame": str(history_path),
                "future_frame": str(future_path),
                "history_future_l1": float(l1.mean()),
                "history_future_warp_error": float(warp_err.mean()),
                "history_future_flow_mean": float(flow_mag.mean()),
                "history_future_flow_p90": float(np.percentile(flow_mag, 90)),
                "valid_mask_ratio": 1.0,
                "time_window": "history_future_boundary",
            }
            self.rows.append(row)
            self.state.add_row("frame_metrics", row)
        self._add_aggregates(scene_sample.scene_id)

    def _add_aggregates(self, scene_id: str) -> None:
        scene_rows = [row for row in self.rows if row["scene_id"] == scene_id]
        if not scene_rows:
            return
        l1_values = [row["history_future_l1"] for row in scene_rows]
        warp_values = [row["history_future_warp_error"] for row in scene_rows]
        self.state.add_row(
            "scene_metrics",
            {
                "scene_id": scene_id,
                "history_future_l1_mean": mean_or_none(l1_values),
                "history_future_l1_p90": percentile_or_none(l1_values, 90),
                "history_future_warp_error_mean": mean_or_none(warp_values),
                "history_future_warp_error_p90": percentile_or_none(warp_values, 90),
                "history_future_camera_count": len(scene_rows),
            },
        )
        for camera in sorted({row["camera"] for row in scene_rows}):
            camera_rows = [row for row in scene_rows if row["camera"] == camera]
            self.state.add_row(
                "camera_metrics",
                {
                    "scene_id": scene_id,
                    "camera": camera,
                    "history_future_l1": mean_or_none(row["history_future_l1"] for row in camera_rows),
                    "history_future_warp_error": mean_or_none(
                        row["history_future_warp_error"] for row in camera_rows
                    ),
                    "history_future_flow_mean": mean_or_none(
                        row["history_future_flow_mean"] for row in camera_rows
                    ),
                    "history_future_flow_p90": percentile_or_none(
                        (row["history_future_flow_p90"] for row in camera_rows), 90
                    ),
                },
            )

    def compute(self):
        super().compute()
        l1_values = [row["history_future_l1"] for row in self.rows]
        warp_values = [row["history_future_warp_error"] for row in self.rows]
        self.summary.update(stats(l1_values, "history_future_l1_"))
        self.summary.update(stats(warp_values, "history_future_warp_error_"))
        self.summary["history_future_camera_boundaries"] = len(self.rows)
        self.summary["flow_backend"] = self.backend
        self.summary["backend"] = self.backend
        self.summary["metric_confidence"] = "medium" if self.backend in {"opencv", "fallback_opencv"} else "normal"
        if not self.rows and not self.skipped:
            self.skip("no_valid_history_future_boundaries", message="No history/future boundary pairs were available.")
        self.summary["metric_validity"] = "valid" if self.rows and not self.skipped else "skipped"
        self.summary["comparison_validity"] = (
            "history_future_condition_consistency" if self.rows and not self.skipped else "invalid"
        )
        self.summary["l1_error_unit"] = "mean_abs_rgb_0_1"
        self.summary["warp_error_unit"] = self.metric_cfg.get("warp_error_unit", "mean_abs_rgb_0_1")
        self.summary["boundary"] = "history_last_frame_to_generated_future_first_frame"
        return self.summary

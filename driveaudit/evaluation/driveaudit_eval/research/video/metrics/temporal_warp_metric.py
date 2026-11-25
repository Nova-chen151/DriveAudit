from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from driveaudit_eval.video.metrics import BaseMetric
from driveaudit_eval.video.cache import DiskCache
from driveaudit_eval.video.utils import image_to_np, mean_or_none, percentile_or_none, stats, time_window
from driveaudit_eval.video.visualize import save_error_triplet


def warp_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class FlowComputer:
    def __init__(self, cfg: Dict[str, Any], state, backend: str = "opencv") -> None:
        self.cfg = cfg
        self.state = state
        self.backend = backend
        self.cache = DiskCache(
            state.cache_dir / "flow",
            enabled=cfg["output"].get("save_cache", True),
        )
        self.size = (
            cfg["preprocess"]["resize_width"],
            cfg["preprocess"]["resize_height"],
        )

    def compute(self, scene_id: str, camera: str, frame_t: int, path_t: Path, path_t1: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = [self.backend, scene_id, camera, frame_t, path_t, path_t1, self.size]
        cached = self.cache.load_npz("opencv", key)
        img_t = image_to_np(path_t, size=self.size)
        img_t1 = image_to_np(path_t1, size=self.size)
        if cached is not None:
            return cached["flow"], img_t, img_t1
        if self.backend not in {"opencv", "fallback_opencv"}:
            raise RuntimeError(f"unsupported lightweight flow backend: {self.backend}")
        import cv2

        gray_t = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)
        gray_t1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_t,
            gray_t1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)
        self.cache.save_npz("opencv", key, flow=flow)
        return flow, img_t, img_t1


class TemporalWarpMetric(BaseMetric):
    name = "temporal_warp"

    def __init__(self, cfg: Dict[str, Any], state) -> None:
        super().__init__(cfg, state)
        self.rows: List[Dict[str, Any]] = []
        self.flow_computer: Optional[FlowComputer] = None
        self.backend = self.metric_cfg.get("flow_backend", "opencv")

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
                self.skip("no_flow_model_for_temporal_warp", message="Flow backend is none.")
                return
        self.flow_computer = FlowComputer(self.cfg, self.state, backend=self.backend)
        self.state.shared["flow_computer"] = self.flow_computer
        self.summary["flow_backend"] = self.backend
        self.summary["backend"] = self.backend
        self.summary["metric_confidence"] = "medium" if self.backend in {"opencv", "fallback_opencv"} else "normal"
        self.summary["metric_validity"] = "valid"
        self.summary["comparison_validity"] = "temporal_self_consistency"
        self.summary["warp_error_unit"] = self.metric_cfg.get("warp_error_unit", "mean_abs_rgb_0_1")
        self.summary["input_image_range"] = "raw_uint8_rgb_resized_then_l1_0_1"

    def update(self, scene_sample):
        if not self.enabled or self.skipped:
            return
        if self.flow_computer is None:
            self.skip("no_flow_model_for_temporal_warp", scene_id=scene_sample.scene_id)
            return
        windows = self.cfg["time_windows"]
        bad_threshold = float(self.metric_cfg.get("bad_frame_threshold", 0.12))
        max_vis = int(self.metric_cfg.get("max_visualizations", 0))
        vis_count = 0
        for camera in scene_sample.cameras:
            future = scene_sample.future_images.get(camera, [])
            if len(future) < 2:
                self.warn("insufficient_frames", f"Need >=2 future frames for {camera}.", scene_sample.scene_id)
                continue
            for idx in range(len(future) - 1):
                flow, img_t, img_t1 = self.flow_computer.compute(
                    scene_sample.scene_id, camera, idx, future[idx], future[idx + 1]
                )
                warped = warp_with_flow(img_t, flow)
                err = np.abs(img_t1.astype(np.float32) - warped.astype(np.float32)) / 255.0
                flow_mag = np.linalg.norm(flow, axis=2)
                row = {
                    "scene_id": scene_sample.scene_id,
                    "camera": camera,
                    "frame_t": idx,
                    "frame_t1": idx + 1,
                    "frame_id": idx,
                    "warp_error": float(err.mean()),
                    "flow_mean": float(flow_mag.mean()),
                    "flow_p90": float(np.percentile(flow_mag, 90)),
                    "valid_mask_ratio": 1.0,
                    "time_window": time_window(idx, windows),
                }
                self.rows.append(row)
                self.state.add_row("frame_metrics", row)
                if self.cfg["output"].get("save_visualizations", True) and vis_count < max_vis:
                    out = (
                        self.state.output_dir
                        / "visualizations/warp_error_heatmap"
                        / scene_sample.scene_id
                        / f"{camera}_{idx:06d}.jpg"
                    )
                    save_error_triplet(img_t, img_t1, warped, err, out)
                    vis_count += 1
        self._add_aggregates(scene_sample.scene_id, bad_threshold)

    def _add_aggregates(self, scene_id: str, bad_threshold: float) -> None:
        scene_rows = [row for row in self.rows if row["scene_id"] == scene_id]
        scene_values = [row["warp_error"] for row in scene_rows]
        scene_summary = {
            "scene_id": scene_id,
            "warp_error_mean": mean_or_none(scene_values),
            "warp_error_p90": percentile_or_none(scene_values, 90),
            "warp_bad_frame_count": int(sum(v > bad_threshold for v in scene_values)),
            "warp_bad_frame_ratio": None
            if not scene_values
            else float(sum(v > bad_threshold for v in scene_values) / len(scene_values)),
        }
        for window in ("short", "middle", "long"):
            w_rows = [row for row in scene_rows if row["time_window"] == window]
            scene_summary[f"flow_mean_{window}"] = mean_or_none(
                row["flow_mean"] for row in w_rows
            )
            scene_summary[f"flow_p90_{window}"] = percentile_or_none(
                (row["flow_p90"] for row in w_rows), 90
            )
        short_flow = scene_summary.get("flow_mean_short")
        long_flow = scene_summary.get("flow_mean_long")
        scene_summary["motion_collapse_ratio"] = None
        if short_flow is not None and long_flow is not None:
            scene_summary["motion_collapse_ratio"] = float(long_flow / (short_flow + 1e-8))
            threshold = float(self.metric_cfg.get("motion_collapse_warning_threshold", 0.6))
            if scene_summary["motion_collapse_ratio"] < threshold:
                self.warn(
                    "possible_motion_collapse",
                    f"motion_collapse_ratio={scene_summary['motion_collapse_ratio']:.4f} < {threshold}",
                    scene_id=scene_id,
                )
        self.state.add_row("scene_metrics", scene_summary)
        for camera in sorted({row["camera"] for row in scene_rows}):
            rows = [row for row in scene_rows if row["camera"] == camera]
            values = [row["warp_error"] for row in rows]
            camera_summary = {
                "scene_id": scene_id,
                "camera": camera,
                "warp_error_mean": mean_or_none(values),
                "warp_error_p90": percentile_or_none(values, 90),
                "warp_bad_frame_count": int(sum(v > bad_threshold for v in values)),
                "warp_bad_frame_ratio": None
                if not values
                else float(sum(v > bad_threshold for v in values) / len(values)),
            }
            self.state.add_row("camera_metrics", camera_summary)
            for window in ("short", "middle", "long", "full"):
                w_rows = rows if window == "full" else [row for row in rows if row["time_window"] == window]
                w_values = [row["warp_error"] for row in w_rows]
                self.state.add_row(
                    "time_window_metrics",
                    {
                        "scene_id": scene_id,
                        "camera": camera,
                        "time_window": window,
                        "warp_error_mean": mean_or_none(w_values),
                        "flow_mean": mean_or_none(row["flow_mean"] for row in w_rows),
                        "flow_p90": percentile_or_none(
                            (row["flow_p90"] for row in w_rows), 90
                        ),
                    },
                )

    def compute(self):
        super().compute()
        values = [row["warp_error"] for row in self.rows]
        self.summary.update(stats(values, "warp_error_"))
        for window in ("short", "middle", "long"):
            self.summary[f"warp_error_{window}"] = mean_or_none(
                row["warp_error"] for row in self.rows if row["time_window"] == window
            )
            self.summary[f"flow_mean_{window}"] = mean_or_none(
                row["flow_mean"] for row in self.rows if row["time_window"] == window
            )
            self.summary[f"flow_p90_{window}"] = percentile_or_none(
                (row["flow_p90"] for row in self.rows if row["time_window"] == window),
                90,
            )
        short_flow = self.summary.get("flow_mean_short")
        long_flow = self.summary.get("flow_mean_long")
        self.summary["motion_collapse_ratio"] = None
        if short_flow is not None and long_flow is not None:
            self.summary["motion_collapse_ratio"] = float(long_flow / (short_flow + 1e-8))
            threshold_collapse = float(self.metric_cfg.get("motion_collapse_warning_threshold", 0.6))
            if self.summary["motion_collapse_ratio"] < threshold_collapse:
                self.warn(
                    "possible_motion_collapse",
                    f"motion_collapse_ratio={self.summary['motion_collapse_ratio']:.4f} < {threshold_collapse}",
                )
        threshold = float(self.metric_cfg.get("bad_frame_threshold", 0.12))
        self.summary["warp_bad_frame_count"] = int(sum(v > threshold for v in values))
        self.summary["warp_bad_frame_ratio"] = None if not values else float(sum(v > threshold for v in values) / len(values))
        self.summary["flow_backend"] = self.backend
        self.summary["backend"] = self.backend
        self.summary["metric_confidence"] = "medium" if self.backend in {"opencv", "fallback_opencv"} else "normal"
        self.summary["metric_validity"] = "valid" if not self.skipped else "skipped"
        self.summary["comparison_validity"] = "temporal_self_consistency" if not self.skipped else "invalid"
        self.summary["warp_error_unit"] = self.metric_cfg.get("warp_error_unit", "mean_abs_rgb_0_1")
        self.summary["input_image_range"] = "raw_uint8_rgb_resized_then_l1_0_1"
        return self.summary

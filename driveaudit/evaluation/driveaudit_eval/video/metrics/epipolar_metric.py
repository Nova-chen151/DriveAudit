from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from driveaudit_eval.video.metrics import BaseMetric
from driveaudit_eval.video.geometry import (
    epipolar_errors,
    fundamental_from_extrinsics,
    prepare_camera_for_eval,
)
from driveaudit_eval.video.utils import mean_or_none, percentile_or_none, time_window, write_json


class EpipolarMetric(BaseMetric):
    name = "epipolar"

    def __init__(self, cfg: Dict[str, Any], state) -> None:
        super().__init__(cfg, state)
        self.rows: List[Dict[str, Any]] = []
        self.debug_intrinsics: List[Dict[str, Any]] = []

    def _convention(self) -> str:
        return self.state.shared.get(
            "selected_extrinsic_convention",
            self.cfg.get("geometry", {}).get("extrinsic_convention", "sensor_to_ego"),
        )

    def update(self, scene_sample):
        if not self.enabled:
            return
        match_backend = self.state.shared.get("match_backend")
        runtime = self.cfg.get("runtime", {})
        if (
            match_backend in {"orb", "orb_fallback"}
            and runtime.get("strict_heavy", False)
            and not runtime.get("allow_fallback", False)
        ):
            raise RuntimeError(
                "Epipolar metric refuses ORB fallback matches in strict-heavy mode. "
                "Configure Kornia LoFTR and its local checkpoint in tools/configs/research.yaml."
            )
        matches = self.state.shared.get("loftr_matches", {})
        if not matches:
            self.skip(
                "empty_loftr_matches",
                scene_id=scene_sample.scene_id,
                message="Epipolar metric needs LoFTR/feature matches.",
            )
            return
        if not scene_sample.camera_metadata:
            self.skip(
                "no_metadata_for_epipolar",
                scene_id=scene_sample.scene_id,
                message="No camera intrinsics/extrinsics were found.",
            )
            return
        thresholds = self.metric_cfg.get("thresholds_px", [1, 3, 5])
        for (scene_id, frame_id, pair_name), match in matches.items():
            if scene_id != scene_sample.scene_id:
                continue
            actual_frame_id = int(match.get("actual_frame_id", frame_id))
            cam_a = match["camera_a"]
            cam_b = match["camera_b"]
            frame_meta = scene_sample.camera_metadata.get(actual_frame_id)
            if frame_meta is None:
                frame_meta = scene_sample.camera_metadata.get(frame_id)
            if not frame_meta or cam_a not in frame_meta or cam_b not in frame_meta:
                self.warn(
                    "camera_extrinsic_missing",
                    f"Missing calibration for {pair_name} frame {actual_frame_id}.",
                    scene_id=scene_sample.scene_id,
                )
                continue
            target_size = (
                self.cfg["preprocess"]["resize_width"],
                self.cfg["preprocess"]["resize_height"],
            )
            cam_a_eval, dbg_a = prepare_camera_for_eval(
                frame_meta[cam_a],
                frame_meta[cam_a].get("image_size") or scene_sample.image_sizes.get(cam_a),
                target_size,
                scale_k=self.cfg.get("geometry", {}).get("scale_intrinsics_for_resize", True),
            )
            cam_b_eval, dbg_b = prepare_camera_for_eval(
                frame_meta[cam_b],
                frame_meta[cam_b].get("image_size") or scene_sample.image_sizes.get(cam_b),
                target_size,
                scale_k=self.cfg.get("geometry", {}).get("scale_intrinsics_for_resize", True),
            )
            convention = self._convention()
            if convention == "auto":
                convention = "sensor_to_ego"
            f_mat = fundamental_from_extrinsics(cam_a_eval, cam_b_eval, convention)
            if f_mat is None:
                self.warn(
                    "camera_intrinsic_missing",
                    f"Could not compute fundamental matrix for {pair_name}.",
                    scene_id=scene_sample.scene_id,
                )
                continue
            if len(self.debug_intrinsics) < 200:
                self.debug_intrinsics.extend(
                    [
                        {
                            "scene_id": scene_sample.scene_id,
                            "frame_id": actual_frame_id,
                            "camera": cam_a,
                            "convention": convention,
                            **dbg_a,
                        },
                        {
                            "scene_id": scene_sample.scene_id,
                            "frame_id": actual_frame_id,
                            "camera": cam_b,
                            "convention": convention,
                            **dbg_b,
                        },
                    ]
                )
            pts_a = match["points_a"]
            pts_b = match["points_b"]
            if len(pts_a) == 0:
                self.warn("empty_loftr_matches", f"No matches for {pair_name}.", scene_sample.scene_id)
                continue
            errs = epipolar_errors(pts_a, pts_b, f_mat)
            row = {
                "scene_id": scene_sample.scene_id,
                "frame_id": frame_id,
                "actual_frame_id": actual_frame_id,
                "camera_pair": pair_name,
                "epipolar_error_mean_px": mean_or_none(errs),
                "epipolar_error_median_px": percentile_or_none(errs, 50),
                "epipolar_error_p90_px": percentile_or_none(errs, 90),
                "epipolar_error_max_px": None if len(errs) == 0 else float(np.max(errs)),
                "epipolar_valid_match_count": int(len(errs)),
                "epipolar_match_backend": self.state.shared.get("match_backend"),
                "epipolar_metric_confidence": self.state.shared.get(
                    "match_metric_confidence", "low"
                ),
                "epipolar_extrinsic_convention": convention,
                "time_window": time_window(frame_id, self.cfg["time_windows"]),
            }
            for threshold in thresholds:
                row[f"epipolar_inlier_ratio_{int(threshold)}px"] = (
                    None if len(errs) == 0 else float(np.mean(errs <= threshold))
                )
            self.rows.append(row)
            self.state.add_row("camera_pair_metrics", row)
        self._add_aggregates(scene_sample.scene_id)

    def _add_aggregates(self, scene_id: str) -> None:
        rows = [row for row in self.rows if row["scene_id"] == scene_id]
        if not rows:
            return
        self.state.add_row(
            "scene_metrics",
            {
                "scene_id": scene_id,
                "epipolar_error_median_px": percentile_or_none(
                    (row["epipolar_error_median_px"] for row in rows), 50
                ),
                "epipolar_error_p90_px": percentile_or_none(
                    (row["epipolar_error_p90_px"] for row in rows), 90
                ),
                "epipolar_inlier_ratio_1px": mean_or_none(
                    row.get("epipolar_inlier_ratio_1px") for row in rows
                ),
                "epipolar_inlier_ratio_3px": mean_or_none(
                    row.get("epipolar_inlier_ratio_3px") for row in rows
                ),
                "epipolar_inlier_ratio_5px": mean_or_none(
                    row.get("epipolar_inlier_ratio_5px") for row in rows
                ),
            },
        )
        for pair in sorted({row["camera_pair"] for row in rows}):
            p_rows = [row for row in rows if row["camera_pair"] == pair]
            self.state.add_row(
                "time_window_metrics",
                {
                    "scene_id": scene_id,
                    "camera_pair": pair,
                    "time_window": "full",
                    "epipolar_error_median_px": percentile_or_none(
                        (row["epipolar_error_median_px"] for row in p_rows), 50
                    ),
                    "epipolar_inlier_ratio_3px": mean_or_none(
                        row.get("epipolar_inlier_ratio_3px") for row in p_rows
                    ),
                },
            )

    def compute(self):
        super().compute()
        self.summary["epipolar_error_mean_px"] = mean_or_none(
            row["epipolar_error_mean_px"] for row in self.rows
        )
        self.summary["epipolar_error_median_px"] = percentile_or_none(
            (row["epipolar_error_median_px"] for row in self.rows), 50
        )
        self.summary["epipolar_error_p90_px"] = percentile_or_none(
            (row["epipolar_error_p90_px"] for row in self.rows), 90
        )
        for threshold in self.metric_cfg.get("thresholds_px", [1, 3, 5]):
            key = f"epipolar_inlier_ratio_{int(threshold)}px"
            self.summary[key] = mean_or_none(row.get(key) for row in self.rows)
        bad_threshold = float(self.metric_cfg.get("bad_frame_threshold_px", 5.0))
        bad = [
            row
            for row in self.rows
            if row.get("epipolar_error_median_px") is not None
            and row["epipolar_error_median_px"] > bad_threshold
        ]
        self.summary["epipolar_bad_frame_count"] = len(bad)
        self.summary["epipolar_bad_frame_ratio"] = None if not self.rows else float(len(bad) / len(self.rows))
        match_backend = self.state.shared.get("match_backend")
        match_confidence = self.state.shared.get("match_metric_confidence", "low")
        self.summary["backend"] = (
            f"metadata_plus_{match_backend}_matches" if match_backend else "metadata_plus_matches"
        )
        self.summary["epipolar_match_backend"] = match_backend
        self.summary["metric_confidence"] = match_confidence
        self.summary["epipolar_metric_confidence"] = match_confidence
        self.summary["selected_extrinsic_convention"] = self._convention()
        sanity = self.state.shared.get("epipolar_sanity_summary", {})
        status = sanity.get("epipolar_metric_status")
        self.summary["epipolar_metric_status"] = status
        if status == "invalid_due_to_calibration_or_matching":
            self.summary["metric_validity"] = "invalid_calibration_or_matching"
            self.summary["comparison_validity"] = "invalid"
        else:
            self.summary["metric_validity"] = "valid" if match_confidence == "normal" else "valid_low_sample"
            self.summary["comparison_validity"] = "multiview_self_consistency"
        return self.summary

    def save(self, output_dir):
        if self.debug_intrinsics:
            existing = self.state.shared.setdefault("debug_camera_intrinsics", [])
            existing.extend(self.debug_intrinsics)

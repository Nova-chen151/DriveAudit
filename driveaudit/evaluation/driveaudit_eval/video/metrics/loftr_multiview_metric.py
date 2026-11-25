from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from driveaudit_eval.video.metrics import BaseMetric
from driveaudit_eval.video.cache import DiskCache
from driveaudit_eval.video.geometry import ransac_fundamental
from driveaudit_eval.video.utils import image_to_np, mean_or_none, percentile_or_none, time_window
from driveaudit_eval.video.visualize import save_match_visualization


class ORBMatcher:
    name = "orb_fallback"

    def __init__(self, max_features: int = 2000) -> None:
        import cv2

        self.cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img_a: np.ndarray, img_b: np.ndarray):
        gray_a = self.cv2.cvtColor(img_a, self.cv2.COLOR_RGB2GRAY)
        gray_b = self.cv2.cvtColor(img_b, self.cv2.COLOR_RGB2GRAY)
        kpts_a, desc_a = self.orb.detectAndCompute(gray_a, None)
        kpts_b, desc_b = self.orb.detectAndCompute(gray_b, None)
        if desc_a is None or desc_b is None or not kpts_a or not kpts_b:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
        matches = sorted(self.matcher.match(desc_a, desc_b), key=lambda m: m.distance)
        pts_a = np.array([kpts_a[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_b = np.array([kpts_b[m.trainIdx].pt for m in matches], dtype=np.float32)
        conf = np.array([max(0.0, 1.0 - m.distance / 256.0) for m in matches], dtype=np.float32)
        return pts_a, pts_b, conf


class KorniaLoFTRMatcher:
    name = "kornia_loftr"

    def __init__(
        self,
        weights: str = "outdoor",
        device: str = "cuda",
        weights_path: Optional[str] = None,
    ) -> None:
        import cv2
        import torch
        from kornia.feature import LoFTR

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.cv2 = cv2
        self.torch = torch
        self.device = device
        if weights_path:
            path = Path(weights_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"LoFTR weights_path does not exist: {path}")
            self.matcher = LoFTR(pretrained=None)
            checkpoint = torch.load(str(path), map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            cleaned = {}
            for key, value in state.items():
                for prefix in ("matcher.", "loftr.", "module."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                cleaned[key] = value
            matched_keys = set(self.matcher.state_dict()).intersection(cleaned)
            if not matched_keys:
                raise RuntimeError(
                    "LoFTR checkpoint did not match Kornia LoFTR state_dict keys."
                )
            self.matcher.load_state_dict(cleaned, strict=False)
        else:
            self.matcher = LoFTR(pretrained=weights)
        self.matcher.eval().to(device)

    def match(self, img_a: np.ndarray, img_b: np.ndarray):
        gray_a = self.cv2.cvtColor(img_a, self.cv2.COLOR_RGB2GRAY)
        gray_b = self.cv2.cvtColor(img_b, self.cv2.COLOR_RGB2GRAY)
        t0 = self.torch.from_numpy(gray_a.astype(np.float32) / 255.0)[None, None].to(self.device)
        t1 = self.torch.from_numpy(gray_b.astype(np.float32) / 255.0)[None, None].to(self.device)
        with self.torch.no_grad():
            out = self.matcher({"image0": t0, "image1": t1})
        pts_a = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
        pts_b = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
        conf = out["confidence"].detach().cpu().numpy().astype(np.float32)
        return pts_a, pts_b, conf


class LoFTRMultiviewMetric(BaseMetric):
    name = "loftr"

    def __init__(self, cfg: Dict[str, Any], state) -> None:
        super().__init__(cfg, state)
        self.rows: List[Dict[str, Any]] = []
        self.matches: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        self.cache = DiskCache(
            state.cache_dir / "loftr_matches",
            enabled=cfg["output"].get("save_cache", True),
        )
        self.matcher = None
        self.backend = self.metric_cfg.get("backend", "kornia")
        self.metric_confidence = "low"

    def _allow_fallback(self) -> bool:
        return bool(self.state.cfg.get("runtime", {}).get("allow_fallback", False))

    def _strict_heavy(self) -> bool:
        runtime = self.state.cfg.get("runtime", {})
        return bool(runtime.get("strict_heavy", False)) and not self._allow_fallback()

    def prepare(self):
        if not self.enabled:
            return
        if self.state.cfg["runtime"].get("skip_heavy", False):
            message = "LoFTR disabled by --skip-heavy."
            if self._strict_heavy():
                raise RuntimeError(
                    f"{message} Strict-heavy mode requires Kornia LoFTR. "
                    "Check dependencies and runtime.skip_heavy in tools/configs/research.yaml."
                )
            self.backend = "orb_fallback"
            self.warn("no_loftr_model", f"{message} using ORB fallback.")
        elif self.backend == "kornia":
            try:
                self.matcher = KorniaLoFTRMatcher(
                    weights=self.metric_cfg.get("weights", "outdoor"),
                    device=self.state.cfg["runtime"].get("device", "cuda"),
                    weights_path=self.metric_cfg.get("weights_path"),
                )
                self.backend = "kornia_loftr"
                self.metric_confidence = "normal"
            except Exception as exc:
                if self._strict_heavy():
                    raise RuntimeError(
                        "Kornia LoFTR failed in strict-heavy mode. "
                        "Configure the LoFTR dependency and local checkpoint in tools/configs/research.yaml. "
                        f"Original error: {exc}"
                    ) from exc
                self.warn("no_loftr_model", f"Kornia LoFTR unavailable: {exc}; using ORB fallback.")
                self.backend = "orb_fallback"
        elif self.backend == "official":
            message = "Official LoFTR backend is not installed/configured."
            if self._strict_heavy():
                raise RuntimeError(f"{message} Strict-heavy mode does not allow ORB fallback.")
            self.warn("no_loftr_model", f"{message} using ORB fallback.")
            self.backend = "orb_fallback"
        else:
            if self.backend not in {"orb", "orb_fallback"}:
                message = "LoFTR weights are not configured."
                if self._strict_heavy():
                    raise RuntimeError(f"{message} Strict-heavy mode does not allow ORB fallback.")
                self.warn("no_loftr_model", f"{message} using ORB fallback for diagnostics.")
            self.backend = "orb_fallback"
        if self.backend in {"orb", "orb_fallback"}:
            if not (self.metric_cfg.get("allow_orb_fallback", False) and self._allow_fallback()):
                message = (
                    "LoFTR unavailable and ORB fallback disabled. Pass --allow-fallback "
                    "for smoke diagnostics or configure a local LoFTR checkpoint."
                )
                if self._strict_heavy():
                    raise RuntimeError(message)
                self.skip("no_loftr_model", message=message)
                return
            try:
                self.matcher = ORBMatcher()
                self.backend = "orb_fallback"
                self.metric_confidence = "low"
            except Exception as exc:
                self.skip("no_loftr_model", message=f"Could not initialize ORB fallback: {exc}")
        self.summary["loftr_backend"] = self.backend
        self.summary["backend"] = self.backend
        self.summary["metric_confidence"] = self.metric_confidence
        self.summary["loftr_metric_confidence"] = self.metric_confidence
        self.summary["metric_validity"] = "valid" if self.metric_confidence == "normal" else "valid_low_sample"
        self.summary["comparison_validity"] = "multiview_self_consistency"
        self.state.shared["feature_matcher"] = self.matcher
        self.state.shared["match_backend"] = self.backend
        self.state.shared["match_metric_confidence"] = self.metric_confidence

    def _cached_match(self, scene_id, frame_id, pair_name, path_a, path_b):
        size = (
            self.cfg["preprocess"]["resize_width"],
            self.cfg["preprocess"]["resize_height"],
        )
        key = [self.backend, scene_id, frame_id, pair_name, path_a, path_b, size]
        cached = self.cache.load_npz("matches", key)
        img_a = image_to_np(path_a, size=size)
        img_b = image_to_np(path_b, size=size)
        if cached is not None:
            return cached["points_a"], cached["points_b"], cached["confidence"], img_a, img_b
        pts_a, pts_b, conf = self.matcher.match(img_a, img_b)
        self.cache.save_npz("matches", key, points_a=pts_a, points_b=pts_b, confidence=conf)
        return pts_a, pts_b, conf, img_a, img_b

    def update(self, scene_sample):
        if not self.enabled or self.skipped:
            return
        if self.matcher is None:
            self.skip("no_loftr_model", scene_id=scene_sample.scene_id)
            return
        windows = self.cfg["time_windows"]
        pairs = self.metric_cfg.get("camera_pairs", [])
        if self.metric_cfg.get('auto_overlap_pairs'):
            pairs = self.metric_cfg['camera_pairs_by_scene'][scene_sample.scene_id]
        low_threshold = int(self.metric_cfg.get("low_match_threshold", 30))
        max_frames = self.metric_cfg.get("max_frames")
        max_vis = int(self.metric_cfg.get("max_visualizations", 0))
        vis_count = 0
        for frame_id in range(min(len(v) for v in scene_sample.future_images.values())):
            if max_frames is not None and frame_id >= int(max_frames):
                break
            for cam_a, cam_b in pairs:
                imgs_a = scene_sample.future_images.get(cam_a, [])
                imgs_b = scene_sample.future_images.get(cam_b, [])
                if frame_id >= len(imgs_a) or frame_id >= len(imgs_b):
                    self.warn(
                        "missing_camera",
                        f"Missing frame {frame_id} for pair {cam_a}-{cam_b}.",
                        scene_id=scene_sample.scene_id,
                    )
                    continue
                pair_name = f"{cam_a}__{cam_b}"
                try:
                    actual_frame_id = int(imgs_a[frame_id].stem)
                except ValueError:
                    actual_frame_id = frame_id
                pts_a, pts_b, conf, img_a, img_b = self._cached_match(
                    scene_sample.scene_id, frame_id, pair_name, imgs_a[frame_id], imgs_b[frame_id]
                )
                _, inliers = ransac_fundamental(pts_a, pts_b)
                num_matches = int(len(pts_a))
                inlier_count = int(inliers.sum()) if len(inliers) else 0
                mean_conf = float(conf.mean()) if len(conf) else 0.0
                median_conf = float(np.median(conf)) if len(conf) else 0.0
                inlier_ratio = float(inlier_count / num_matches) if num_matches else 0.0
                score_raw = float(inlier_ratio * np.log1p(num_matches) * mean_conf)
                row = {
                    "scene_id": scene_sample.scene_id,
                    "frame_id": frame_id,
                    "actual_frame_id": actual_frame_id,
                    "camera_pair": pair_name,
                    "loftr_num_matches": num_matches,
                    "loftr_mean_confidence": mean_conf,
                    "loftr_median_confidence": median_conf,
                    "loftr_ransac_inlier_count": inlier_count,
                    "loftr_ransac_inlier_ratio": inlier_ratio,
                    "loftr_valid_match_ratio": inlier_ratio,
                    "loftr_score_raw": score_raw,
                    "time_window": time_window(frame_id, windows),
                    "loftr_backend": self.backend,
                }
                self.rows.append(row)
                self.state.add_row("camera_pair_metrics", row)
                self.matches[(scene_sample.scene_id, frame_id, pair_name)] = {
                    "points_a": pts_a,
                    "points_b": pts_b,
                    "confidence": conf,
                    "inliers": inliers,
                    "camera_a": cam_a,
                    "camera_b": cam_b,
                    "actual_frame_id": actual_frame_id,
                }
                if num_matches == 0:
                    self.warn("empty_loftr_matches", f"No matches for {pair_name} frame {frame_id}.", scene_sample.scene_id)
                if self.cfg["output"].get("save_visualizations", True) and vis_count < max_vis:
                    out = (
                        self.state.output_dir
                        / "visualizations/loftr_matches"
                        / scene_sample.scene_id
                        / f"{pair_name}_{frame_id:06d}.jpg"
                    )
                    save_match_visualization(img_a, img_b, pts_a, pts_b, inliers, out)
                    vis_count += 1
        self.state.shared["loftr_matches"] = self.matches
        self._add_aggregates(scene_sample.scene_id, low_threshold)

    def _add_aggregates(self, scene_id: str, low_threshold: int) -> None:
        rows = [row for row in self.rows if row["scene_id"] == scene_id]
        if not rows:
            return
        pair_stats = {}
        for pair in sorted({row["camera_pair"] for row in rows}):
            p_rows = [row for row in rows if row["camera_pair"] == pair]
            counts = [row["loftr_num_matches"] for row in p_rows]
            pair_stats[pair] = {
                "num_matches_mean": mean_or_none(counts),
                "num_matches_p10": percentile_or_none(counts, 10),
                "num_matches_p90": percentile_or_none(counts, 90),
                "ransac_inlier_ratio_mean": mean_or_none(row["loftr_ransac_inlier_ratio"] for row in p_rows),
                "ransac_inlier_ratio_p10": percentile_or_none((row["loftr_ransac_inlier_ratio"] for row in p_rows), 10),
                "ransac_inlier_ratio_p90": percentile_or_none((row["loftr_ransac_inlier_ratio"] for row in p_rows), 90),
            }
            self.state.add_row(
                "time_window_metrics",
                {
                    "scene_id": scene_id,
                    "camera_pair": pair,
                    "time_window": "full",
                    "loftr_num_matches_mean": mean_or_none(counts),
                    "loftr_ransac_inlier_ratio_mean": mean_or_none(row["loftr_ransac_inlier_ratio"] for row in p_rows),
                },
            )
        self.state.shared["loftr_pair_stats"] = pair_stats

    def compute(self):
        super().compute()
        self.summary["loftr_backend"] = self.backend
        self.summary["backend"] = self.backend
        self.summary["metric_confidence"] = self.metric_confidence
        self.summary["loftr_metric_confidence"] = self.metric_confidence
        self.summary["metric_validity"] = "valid" if self.metric_confidence == "normal" else ("skipped" if self.skipped else "valid_low_sample")
        self.summary["comparison_validity"] = "multiview_self_consistency" if not self.skipped else "invalid"
        self.summary["loftr_num_matches_mean"] = mean_or_none(
            row["loftr_num_matches"] for row in self.rows
        )
        self.summary["loftr_num_matches_p10"] = percentile_or_none(
            (row["loftr_num_matches"] for row in self.rows), 10
        )
        self.summary["loftr_mean_confidence_mean"] = mean_or_none(
            row["loftr_mean_confidence"] for row in self.rows
        )
        self.summary["loftr_ransac_inlier_ratio_mean"] = mean_or_none(
            row["loftr_ransac_inlier_ratio"] for row in self.rows
        )
        threshold = int(self.metric_cfg.get("low_match_threshold", 30))
        low = [row for row in self.rows if row["loftr_num_matches"] < threshold]
        self.summary["loftr_low_match_frame_count"] = len(low)
        self.summary["loftr_low_match_frame_ratio"] = None if not self.rows else float(len(low) / len(self.rows))
        self.summary["loftr_pair_stats"] = self.state.shared.get("loftr_pair_stats", {})
        return self.summary

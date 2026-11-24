from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from driveaudit_eval.video.cache import DiskCache
from driveaudit_eval.video.sampling import ClipSample, ImageSample
from driveaudit_eval.video.utils import image_to_np


_EXECUTOR_BINDING_TOKEN = object()

# The reference TF-Hub Kinetics-400 head, as ported by StyleGAN-V. The
# 1024-dimensional activations before its class projection are not FVD logits.
FVD_PROTOCOL = "i3d_kinetics400_logits_torchscript_v2"
FVD_FEATURE_DIM = 400
FVD_OUTPUT_KIND = "kinetics400_logits_pre_softmax"
FVD_REFERENCE_WEIGHTS_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    if len(features_a) < 2 or len(features_b) < 2:
        raise ValueError("Frechet distance requires at least two samples per distribution.")
    from scipy import linalg

    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    if not np.isfinite(features_a).all() or not np.isfinite(features_b).all():
        raise ValueError("Frechet distance features must contain only finite values.")
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)
    covmean = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        eye = np.eye(cov_a.shape[0], dtype=cov_a.dtype)
        covmean = linalg.sqrtm((cov_a + eps * eye) @ (cov_b + eps * eye))
    if not np.isfinite(covmean).all():
        raise RuntimeError("Frechet covariance square root is not finite; check the numerical runtime.")
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    value = diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean)
    if not np.isfinite(value):
        raise RuntimeError("Frechet distance is not finite; check the features and numerical runtime.")
    return float(max(0.0, value))


def _path_signature(paths: Sequence[Path]) -> List[Tuple[str, int, int]]:
    signature = []
    for path in paths:
        stat = path.stat()
        signature.append((str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)))
    return signature


def _clip_signature(clips: Sequence[ClipSample]) -> List[Any]:
    return [
        [
            clip.scene_id,
            clip.camera,
            clip.start_position,
            [list(signature) for signature in _path_signature(clip.paths)],
        ]
        for clip in clips
    ]




class I3DFVDExecutor:
    def __init__(self, cfg: Dict[str, Any], cache_dir: Path, *, binding_token: object = None) -> None:
        if binding_token is not _EXECUTOR_BINDING_TOKEN:
            raise RuntimeError("I3DFVDExecutor can only be created by ExecutionBindingLayer.")
        import torch

        metric_cfg = cfg.get("metrics", {}).get("fvd", {})
        weights_path = metric_cfg.get("weights_path") or cfg.get("weights", {}).get("fvd_i3d_torchscript")
        if not weights_path:
            raise FileNotFoundError("metrics.fvd.weights_path is required for strict I3D-FVD.")
        weights = Path(weights_path).expanduser()
        if not weights.exists():
            raise FileNotFoundError(f"I3D TorchScript weights not found: {weights}")
        runtime = cfg.get("runtime", {})
        requested_device = str(runtime.get("device", "cpu"))
        self.device = requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.batch_size = int(metric_cfg.get("batch_size", 2))
        self.resize = tuple(metric_cfg.get("resize", [224, 224]))
        if self.batch_size < 1:
            raise ValueError("I3D-FVD batch_size must be positive.")
        if self.resize != (224, 224):
            raise ValueError("Reference I3D-FVD requires resize=[224, 224].")
        self.allow_cache_read = bool(metric_cfg.get("allow_cache_read", False))
        self.weights_path = str(weights.resolve())
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.weights_sha256 = digest.hexdigest()
        expected_sha256 = metric_cfg.get("expected_weights_sha256")
        if expected_sha256 and self.weights_sha256 != str(expected_sha256).lower():
            raise ValueError(
                f"I3D-FVD checkpoint SHA256 mismatch: expected {expected_sha256}, "
                f"got {self.weights_sha256}."
            )
        self.torch = torch
        self.model = torch.jit.load(str(weights), map_location=self.device)
        self.model.eval()
        self.cache = DiskCache(cache_dir / "fvd_i3d", enabled=cfg["output"].get("save_cache", True))

    def preprocessing_plan(self, clips: Sequence[ClipSample], domain: str) -> Dict[str, Any]:
        clip_length = len(clips[0].paths) if clips else None
        return {
            "executor": "I3DFVDExecutor",
            "domain": domain,
            "num_items": len(clips),
            "batch_size": self.batch_size,
            "resize": list(self.resize),
            "clip_length": clip_length,
            "protocol": FVD_PROTOCOL,
            "output_kind": FVD_OUTPUT_KIND,
            "feature_dim": FVD_FEATURE_DIM,
            "input_layout": "BTHWC_uint8_rgb_native_resolution",
            "model_layout": "BCTHW_float32",
            "model_input_range": [0.0, 255.0],
            "normalization": "torchscript_rescale_0_255_to_minus1_plus1",
            "resize_backend": "torchscript_torch_interpolate_bilinear",
            "resize_align_corners": False,
            "resize_antialias": False,
            "model_forward_kwargs": {"rescale": True, "resize": True, "return_features": True},
            "ordered_clip_signature": _clip_signature(clips),
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
        }

    def forward(self, clips: Sequence[ClipSample], domain: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        provenance = self.preprocessing_plan(clips, domain)
        key = ["I3DFVDExecutor.forward", json.dumps(provenance, sort_keys=True)]
        cached = self.cache.load_npz("features", key) if self.allow_cache_read else None
        if cached is not None:
            cached_provenance = json.loads(str(cached["provenance_json"].item()))
            if cached_provenance != provenance:
                raise RuntimeError("I3D-FVD feature cache provenance mismatch.")
            cached_features = cached["features"].astype(np.float64)
            self._validate_features(cached_features, len(clips))
            return cached_features, {
                "executor": "I3DFVDExecutor",
                "forward_pass": "i3d_torchscript",
                "forward_executed": False,
                "cache_hit": True,
                "output_shape": list(cached["features"].shape),
                "provenance": provenance,
            }
        features = np.empty((len(clips), FVD_FEATURE_DIM), dtype=np.float64)
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            groups = {}
            for index, clip in enumerate(batch):
                frames = np.stack([image_to_np(path) for path in clip.paths], axis=0)
                groups.setdefault(frames.shape, []).append((start + index, frames))
            # Different cameras may have different native resolutions. Group only
            # equal-shaped clips, then scatter logits back to the manifest order.
            for entries in groups.values():
                arr = np.stack([frames for _, frames in entries], axis=0)
                tensor = self.torch.from_numpy(arr).permute(0, 4, 1, 2, 3).contiguous().to(
                    device=self.device, dtype=self.torch.float32
                )
                with self.torch.no_grad():
                    out = self._forward_i3d_features(tensor)
                values = out.detach().cpu().numpy().astype(np.float64)
                self._validate_features(values, len(entries))
                features[[index for index, _ in entries]] = values
        self.cache.save_npz("features", key, features=features, provenance_json=np.asarray(json.dumps(provenance)))
        return features, {
            "executor": "I3DFVDExecutor",
            "forward_pass": "i3d_torchscript",
            "forward_executed": True,
            "cache_hit": False,
            "output_shape": list(features.shape),
            "provenance": provenance,
        }

    def _forward_i3d_features(self, x):
        return self.model(x, rescale=True, resize=True, return_features=True)

    @staticmethod
    def _validate_features(features: np.ndarray, count: int) -> None:
        if features.shape != (count, FVD_FEATURE_DIM):
            raise RuntimeError(
                f"Invalid I3D-FVD logits shape: {features.shape}; expected ({count}, {FVD_FEATURE_DIM})."
            )
        if not np.isfinite(features).all():
            raise RuntimeError("I3D-FVD logits must contain only finite values.")

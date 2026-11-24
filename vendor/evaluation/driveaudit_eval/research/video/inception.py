"""Offline TF-Inception pool3 features for FID and KID.

Architecture and preprocessing follow mseitzer/pytorch-fid. The already installed
PyIQA port supplies its TensorFlow-compatible Inception blocks; its checkpoint
downloader and image preprocessing are deliberately not invoked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from driveaudit_eval.video.cache import DiskCache
from driveaudit_eval.video.distribution import _EXECUTOR_BINDING_TOKEN, _path_signature
from driveaudit_eval.video.sampling import ImageSample
from driveaudit_eval.video.utils import image_to_np


FID_BACKEND = "pytorch_fid_inception_v3"
FID_PROTOCOL = "pytorch_fid_pool3_bilinear299_align_corners_false_minus1_plus1_v1"
FID_WEIGHTS_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"


class InceptionFIDExecutor:
    def __init__(self, cfg: Dict[str, Any], cache_dir: Path, *, binding_token: object = None) -> None:
        if binding_token is not _EXECUTOR_BINDING_TOKEN:
            raise RuntimeError("InceptionFIDExecutor can only be created by ExecutionBindingLayer.")
        metric_cfg = cfg.get("metrics", {}).get("fid", {})
        backend = metric_cfg.get("backend", FID_BACKEND)
        if backend != FID_BACKEND:
            raise ValueError(f"Unsupported FID backend {backend!r}; canonical FID/KID requires {FID_BACKEND!r}.")
        self.batch_size = int(metric_cfg.get("batch_size", 32))
        if self.batch_size < 1:
            raise ValueError("FID batch_size must be positive.")
        self.resize = tuple(metric_cfg.get("resize", [299, 299]))
        self.feature_dim = int(metric_cfg.get("feature", 2048))
        if self.feature_dim != 2048 or self.resize != (299, 299):
            raise ValueError("Canonical Inception-FID requires feature=2048 and resize=[299, 299].")
        weights_path = metric_cfg.get("weights_path") or cfg.get("weights", {}).get("fid_inception_v3")
        if not weights_path:
            raise FileNotFoundError("metrics.fid.weights_path must name the official local FID checkpoint.")
        weights = Path(weights_path).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"Official FID checkpoint not found: {weights}")
        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        if self.checkpoint_sha256 != FID_WEIGHTS_SHA256:
            raise ValueError(f"FID checkpoint SHA256 mismatch: expected {FID_WEIGHTS_SHA256}, got {self.checkpoint_sha256}.")
        self.weights_path = str(weights.resolve())
        self.allow_cache_read = bool(metric_cfg.get("allow_cache_read", False))

        import torch
        import torchvision
        import pyiqa
        from pyiqa.archs.inception import FIDInceptionA, FIDInceptionC, FIDInceptionE_1, FIDInceptionE_2

        self.torch = torch
        requested_device = str(cfg.get("runtime", {}).get("device", "cpu"))
        self.device = requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.implementation = {
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pyiqa": pyiqa.__version__, "blocks": "pyiqa.archs.inception.FIDInceptionA_C_E1_E2",
        }
        # These blocks preserve TF pooling semantics and the original FID weights.
        model = torchvision.models.inception_v3(
            weights=None, num_classes=1008, aux_logits=False, transform_input=False, init_weights=False,
        )
        model.Mixed_5b = FIDInceptionA(192, pool_features=32)
        model.Mixed_5c = FIDInceptionA(256, pool_features=64)
        model.Mixed_5d = FIDInceptionA(288, pool_features=64)
        model.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        model.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        model.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        model.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        model.Mixed_7b = FIDInceptionE_1(1280)
        model.Mixed_7c = FIDInceptionE_2(2048)
        model.load_state_dict(torch.load(str(weights), map_location="cpu"), strict=True)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.cache = DiskCache(cache_dir / FID_PROTOCOL, enabled=cfg.get("output", {}).get("save_cache", True))

    def preprocessing_plan(self, samples: Sequence[ImageSample], domain: str) -> Dict[str, Any]:
        return {
            "executor": "InceptionFIDExecutor", "domain": domain, "num_items": len(samples),
            "batch_size": self.batch_size, "resize": list(self.resize),
            "resize_implementation": "torch.nn.functional.interpolate_bilinear_align_corners_false",
            "normalization": "uint8_rgb_to_minus1_plus1", "input_layout": "NHWC_uint8_rgb",
            "tensor_layout": "NCHW_float32", "feature": "pool3", "feature_dim": self.feature_dim,
            "backend": FID_BACKEND, "backbone": "tensorflow_inception_2015_12_05",
            "protocol": FID_PROTOCOL, "weights_path": self.weights_path,
            "checkpoint_sha256": self.checkpoint_sha256, "implementation": self.implementation,
            "ordered_path_signature": [list(row) for row in _path_signature([sample.path for sample in samples])],
        }

    def forward(self, samples: Sequence[ImageSample], domain: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        provenance = self.preprocessing_plan(samples, domain)
        # Includes checkpoint content, implementation and preprocessing, so legacy
        # ImageNet features cannot be accepted under the canonical metric's name.
        key = ["InceptionFIDExecutor.forward", provenance]
        cached = self.cache.load_npz("features", key) if self.allow_cache_read else None
        if cached is not None:
            cached_provenance = json.loads(str(cached["provenance_json"].item()))
            features = cached["features"].astype(np.float64)
            if cached_provenance != provenance or features.shape != (len(samples), self.feature_dim) or not np.isfinite(features).all():
                raise RuntimeError("Inception-FID feature cache provenance, shape or finiteness mismatch.")
            return features, self._audit(provenance, features, cache_hit=True)

        features = []
        for start in range(0, len(samples), self.batch_size):
            tensors = []
            for sample in samples[start:start + self.batch_size]:
                # Resize float pixels with the reference's bilinear operator.
                # Resizing before stacking also permits different native sizes.
                rgb = np.array(image_to_np(sample.path), copy=True)
                x = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                x = self.torch.nn.functional.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
                tensors.append(x.squeeze(0))
            tensor = self.torch.stack(tensors).to(self.device) * 2.0 - 1.0
            with self.torch.inference_mode():
                out = self.model(tensor)
            features.append(out.detach().cpu().numpy().astype(np.float64))
        ret = np.concatenate(features, axis=0) if features else np.empty((0, self.feature_dim), dtype=np.float64)
        if ret.shape != (len(samples), self.feature_dim) or not np.isfinite(ret).all():
            raise RuntimeError(f"Invalid Inception-FID features: shape={ret.shape}")
        self.cache.save_npz("features", key, features=ret, provenance_json=np.asarray(json.dumps(provenance)))
        return ret, self._audit(provenance, ret, cache_hit=False)

    @staticmethod
    def _audit(provenance, features, *, cache_hit):
        return {
            "executor": "InceptionFIDExecutor", "forward_pass": FID_BACKEND,
            "forward_executed": not cache_hit, "cache_hit": cache_hit,
            "output_shape": list(features.shape), "provenance": provenance,
        }

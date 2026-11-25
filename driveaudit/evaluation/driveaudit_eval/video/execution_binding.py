from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from driveaudit_eval.video.distribution import (
    I3DFVDExecutor,
    _EXECUTOR_BINDING_TOKEN,
)
from driveaudit_eval.video.sampling import ClipSample, ImageSample


class ExecutionBindingLayer:
    """The only allowed bridge from the engine DAG to backbone executors."""

    def __init__(self, runtime_state, manifest: Dict[str, Any]) -> None:
        self.state = runtime_state
        self.manifest = manifest
        self._fid_executor: Optional[InceptionFIDExecutor] = None
        self._fvd_executor: Optional[I3DFVDExecutor] = None
        self.state.binding_trace.setdefault("binding_layer.initialized", True)

    def _fid(self) -> InceptionFIDExecutor:
        from driveaudit_eval.research.video.inception import InceptionFIDExecutor
        if self._fid_executor is None:
            self._fid_executor = InceptionFIDExecutor(
                self.state.cfg,
                self.state.cache_dir,
                binding_token=_EXECUTOR_BINDING_TOKEN,
            )
        return self._fid_executor

    def _fvd(self) -> I3DFVDExecutor:
        if self._fvd_executor is None:
            self._fvd_executor = I3DFVDExecutor(
                self.state.cfg,
                self.state.cache_dir,
                binding_token=_EXECUTOR_BINDING_TOKEN,
            )
        return self._fvd_executor

    def prepare_fid_images(self, images: Sequence[ImageSample], domain: str) -> Dict[str, Any]:
        plan = self._fid().preprocessing_plan(images, domain)
        self.state.binding_events.append(
            {
                "event": "binding_layer.fid_preprocessing_prepared",
                "domain": domain,
                "num_items": len(images),
                "executor": "InceptionFIDExecutor",
            }
        )
        return plan

    def prepare_fvd_clips(self, clips: Sequence[ClipSample], domain: str) -> Dict[str, Any]:
        plan = self._fvd().preprocessing_plan(clips, domain)
        self.state.binding_events.append(
            {
                "event": "binding_layer.fvd_preprocessing_prepared",
                "domain": domain,
                "num_items": len(clips),
                "executor": "I3DFVDExecutor",
            }
        )
        return plan

    def run_fid_features(self, images: Sequence[ImageSample], domain: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.state.binding_trace["binding_layer.fid_forward_called"] = True
        features, audit = self._fid().forward(images, domain)
        self.state.binding_trace[f"binding_layer.fid_{domain}_forward_called"] = True
        self.state.binding_trace[f"inception_{domain}_forward_executed"] = bool(audit.get("forward_executed"))
        self.state.binding_trace[f"inception_{domain}_cache_verified"] = bool(audit.get("cache_hit"))
        self.state.binding_events.append(
            {
                "event": "binding_layer.fid_forward_called",
                "domain": domain,
                "executor": "InceptionFIDExecutor.forward",
                "forward_executed": bool(audit.get("forward_executed")),
                "cache_hit": bool(audit.get("cache_hit")),
                "output_shape": audit.get("output_shape"),
            }
        )
        return features, audit

    def run_fvd_features(self, videos: Sequence[ClipSample], domain: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.state.binding_trace["binding_layer.fvd_forward_called"] = True
        features, audit = self._fvd().forward(videos, domain)
        self.state.binding_trace[f"binding_layer.fvd_{domain}_forward_called"] = True
        self.state.binding_trace[f"i3d_{domain}_forward_executed"] = bool(audit.get("forward_executed"))
        self.state.binding_trace[f"i3d_{domain}_cache_verified"] = bool(audit.get("cache_hit"))
        self.state.binding_events.append(
            {
                "event": "binding_layer.fvd_forward_called",
                "domain": domain,
                "executor": "I3DFVDExecutor.forward",
                "forward_executed": bool(audit.get("forward_executed")),
                "cache_hit": bool(audit.get("cache_hit")),
                "output_shape": audit.get("output_shape"),
            }
        )
        return features, audit

    def validate_fid_trace(self) -> None:
        self._validate_trace("fid", "inception")

    def validate_fvd_trace(self) -> None:
        self._validate_trace("fvd", "i3d")

    def _validate_trace(self, metric: str, backbone: str) -> None:
        allow_cache = bool(self.state.cfg.get("metrics", {}).get(metric, {}).get("allow_cache_read", False))
        required = [
            f"binding_layer.{metric}_forward_called",
            f"binding_layer.{metric}_real_forward_called",
            f"binding_layer.{metric}_fake_forward_called",
        ]
        missing = [key for key in required if not self.state.binding_trace.get(key)]
        if missing:
            raise RuntimeError(f"{metric.upper()} binding trace missing required events: {missing}")
        for domain in ("real", "fake"):
            forward = bool(self.state.binding_trace.get(f"{backbone}_{domain}_forward_executed"))
            cache = bool(self.state.binding_trace.get(f"{backbone}_{domain}_cache_verified"))
            if not forward and not (allow_cache and cache):
                raise RuntimeError(
                    f"{metric.upper()} requires {backbone} {domain} forward execution "
                    "or an explicitly enabled verified internal cache hit."
                )

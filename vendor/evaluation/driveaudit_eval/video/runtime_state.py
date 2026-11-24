from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from driveaudit_eval.video.utils import EvalState, ensure_dir


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Match the corrected FVD/FID/KID/CMMD reference's native precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    except Exception:
        return


class RuntimeState(EvalState):
    """Single mutable state object shared by the benchmark engine DAG."""

    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        super().__init__(cfg, output_dir)
        self.seed = int(cfg.get("sampling", {}).get("seed", 42))
        self.device = str(cfg.get("runtime", {}).get("device", "cpu"))
        self.dataset_index: Dict[str, Any] = {}
        self.dataset_index_order: List[str] = []
        self.dataset_version_hash: Optional[str] = None
        self.scene_order_hash: Optional[str] = None
        self.manifest: Dict[str, Any] = {}
        self.samples: List[Any] = []
        self.samples_by_id: Dict[str, Any] = {}
        self.plan: Optional[Any] = None
        self.features: Dict[str, Dict[str, Any]] = {}
        self.preprocessing: Dict[str, Dict[str, Any]] = {}
        self.feature_cache_events: List[Dict[str, Any]] = []
        self.binding_trace: Dict[str, Any] = {}
        self.binding_events: List[Dict[str, Any]] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        self.execution_trace: List[Dict[str, Any]] = []
        self.graph: Dict[str, Any] = {}
        self.dataset_statistics: Dict[str, Any] = {}
        self.consistency_samples: List[Any] = []
        self.cache_versions = {
            "fid": "pytorch_fid_pool3_bilinear299_align_corners_false_minus1_plus1_v1",
            "fvd": "i3d_kinetics400_logits_torchscript_v2",
            "sampling": "dataset_manifest_v2",
        }
        if cfg.get('output', {}).get('save_cache', True) and cfg.get('output', {}).get('retention') != 'metrics':
            ensure_dir(self.cache_dir)

    def set_deterministic(self) -> None:
        set_reproducibility(self.seed)

    def register_trace(self, record: Dict[str, Any]) -> None:
        self.execution_trace.append(record)

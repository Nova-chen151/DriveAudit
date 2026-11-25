from __future__ import annotations

import numpy as np

from driveaudit_eval.video.distribution import FVD_FEATURE_DIM, frechet_distance


class FVDMetric:
    name = "fvd"

    @staticmethod
    def compute(real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
        real_embeddings = np.asarray(real_embeddings, dtype=np.float64)
        fake_embeddings = np.asarray(fake_embeddings, dtype=np.float64)
        if real_embeddings.ndim != 2 or fake_embeddings.ndim != 2:
            raise ValueError("FVDMetric.compute expects 2D embedding arrays.")
        if real_embeddings.shape[1] != FVD_FEATURE_DIM or fake_embeddings.shape[1] != FVD_FEATURE_DIM:
            raise ValueError(
                f"FVDMetric requires {FVD_FEATURE_DIM}-dimensional Kinetics-400 I3D logits before softmax, "
                f"got real={real_embeddings.shape}, fake={fake_embeddings.shape}."
            )
        return frechet_distance(real_embeddings, fake_embeddings)

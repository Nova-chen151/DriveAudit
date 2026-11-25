from __future__ import annotations

import numpy as np

from driveaudit_eval.video.distribution import frechet_distance


class FIDMetric:
    name = "fid"

    @staticmethod
    def compute(real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
        real_embeddings = np.asarray(real_embeddings, dtype=np.float64)
        fake_embeddings = np.asarray(fake_embeddings, dtype=np.float64)
        if real_embeddings.ndim != 2 or fake_embeddings.ndim != 2:
            raise ValueError("FIDMetric.compute expects 2D embedding arrays.")
        if real_embeddings.shape[1] != 2048 or fake_embeddings.shape[1] != 2048:
            raise ValueError(
                f"FIDMetric requires InceptionV3 pool3 embeddings with dim 2048, "
                f"got real={real_embeddings.shape}, fake={fake_embeddings.shape}."
            )
        return frechet_distance(real_embeddings, fake_embeddings)

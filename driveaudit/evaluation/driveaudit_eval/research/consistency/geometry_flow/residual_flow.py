from __future__ import annotations

import numpy as np


def reconstruct(source_points: np.ndarray, predicted_points: np.ndarray, image_points: np.ndarray) -> dict:
    expected=np.asarray(predicted_points,float)-np.asarray(source_points,float);residual=np.asarray(image_points,float)-np.asarray(predicted_points,float)
    observed=np.asarray(image_points,float)-np.asarray(source_points,float)
    return {"expected_flow":expected,"residual_flow":residual,"observed_flow":observed,
            "residual_epe":float(np.median(np.linalg.norm(residual,axis=1))) if len(residual) else None}


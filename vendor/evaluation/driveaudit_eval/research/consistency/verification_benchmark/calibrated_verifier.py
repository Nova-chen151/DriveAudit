from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json

FEATURES = (
    "point_support_score", "forward_backward_error", "ransac_inlier_ratio", "dense_flow_support_score",
    "residual_flow_epe", "flow_cosine_similarity", "residual_flow_cosine", "motion_magnitude_relative_error",
    "background_flow_leakage_ratio", "foreground_contrast_score", "scale_change_error", "feature_support_score",
    "valid_point_ratio", "expected_camera_count", "projected_area", "projected_depth", "occlusion_ratio",
)


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row.get(name) or 0.0) for name in FEATURES] for row in rows], float)


def fit(rows: list[dict[str, Any]], calibration_scenes: set[str], output: str | Path, minimum_scenes: int = 2) -> dict[str, Any]:
    train = [row for row in rows if row.get("scene_id") in calibration_scenes and row.get("label") in {0, 1}]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if len(calibration_scenes) < minimum_scenes or len({row["label"] for row in train}) < 2:
        result = {"status": "experimental_not_fitted_insufficient_calibration_scenes", "calibration_scene_count": len(calibration_scenes),
                  "training_sample_count": len(train), "test_or_holdout_read": False, "features": list(FEATURES)}
        write_json(output / "model_audit.json", result); return result
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2026))])
    model.fit(feature_matrix(train), np.asarray([row["label"] for row in train], int))
    payload = {"status": "fitted_experimental", "features": list(FEATURES), "calibration_scene_count": len(calibration_scenes),
               "training_sample_count": len(train), "coefficients": model.named_steps["model"].coef_[0].tolist(),
               "intercept": model.named_steps["model"].intercept_.tolist(), "test_or_holdout_read": False,
               "model": model}
    write_json(output / "model_audit.json", {key: value for key, value in payload.items() if key != "model"})
    return payload


def discrimination(labels: list[int], scores: list[float], threshold: float = .5) -> dict[str, Any]:
    y, p = np.asarray(labels, int), np.asarray(scores, float); prediction = p >= threshold
    if len(np.unique(y)) < 2: return {"status": "not_available_single_class", "sample_count": len(y)}
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel(); fpr_curve, tpr_curve, thresholds = roc_curve(y, p)
    eer_index = int(np.nanargmin(np.abs(fpr_curve - (1 - tpr_curve))))
    bins = np.linspace(0, 1, 11); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    fraction, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    return {"status": "computed", "sample_count": len(y), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)), "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)), "specificity": float(tn / max(1, tn + fp)),
            "false_positive_rate": float(fp / max(1, tn + fp)), "false_negative_rate": float(fn / max(1, fn + tp)),
            "equal_error_rate": float((fpr_curve[eer_index] + 1 - tpr_curve[eer_index]) / 2),
            "brier_score": float(brier_score_loss(y, p)), "expected_calibration_error": float(ece),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "reliability_diagram": {"predicted_probability": predicted.tolist(), "positive_fraction": fraction.tolist()}}


def predict(model_payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[float] | None:
    model = model_payload.get("model")
    return model.predict_proba(feature_matrix(rows))[:, 1].astype(float).tolist() if model is not None else None


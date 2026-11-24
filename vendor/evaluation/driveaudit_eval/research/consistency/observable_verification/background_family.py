from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import logsumexp


def score(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["background_scoring"]
    if not rows:
        return {"background_best_raw_likelihood": 0.0, "background_marginal_likelihood": 0.0,
                "background_complexity_corrected_likelihood": 0.0, "official_background_likelihood": 0.0,
                "best_background_depth": None, "background_model_selection_penalty": 0.0}
    priors = np.asarray([float(cfg["depth_priors"].get(row["depth_source"], 1.0)) for row in rows], float)
    priors /= max(priors.sum(), 1e-12)
    likelihoods = np.clip(np.asarray([float(row.get("likelihood", 0.0)) for row in rows]), 1e-12, 1.0)
    best_index = int(np.argmax(likelihoods)); raw = float(likelihoods[best_index])
    marginal = float(np.exp(logsumexp(np.log(priors) + np.log(likelihoods))))
    penalty = float(cfg["complexity_penalty_per_extra_model"]) * max(0, len(rows)-1)
    corrected = float(raw * math.exp(-penalty))
    method = cfg["image_scoring_method"]
    official = marginal if method == "marginal_likelihood" else corrected
    return {"background_best_raw_likelihood": raw, "background_marginal_likelihood": marginal,
            "background_complexity_corrected_likelihood": corrected, "official_background_likelihood": official,
            "best_background_depth": rows[best_index].get("depth_source"),
            "background_model_selection_penalty": penalty, "background_scoring_method": method,
            "raw_best_of_family_for_ranking": False}

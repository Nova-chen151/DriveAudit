from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.hypothesis_verification.hypothesis_competition import compete

from driveaudit_eval.research.consistency.observable_verification.background_family import score


def evaluate(observed: dict[str, Any], source_calibration: dict[str, Any], target_calibration: dict[str, Any],
             config: dict[str, Any]) -> dict[str, Any]:
    raw = compete(observed, source_calibration, target_calibration, config)
    family = score(raw["background_hypotheses"], config)
    likelihoods = {"target": float(raw["target_likelihood"]), "background": family["official_background_likelihood"],
                   "zero": float(raw["zero_motion_likelihood"]), "other": float(raw["other_motion_likelihood"])}
    winner = max(likelihoods, key=likelihoods.get); competitor = max(likelihoods[name] for name in ("background", "zero", "other"))
    return {**raw, **family, "hypothesis_likelihoods": likelihoods, "winning_hypothesis": winner,
            "best_background_likelihood": likelihoods["background"], "target_hypothesis_margin": likelihoods["target"]-competitor,
            "target_vs_background_margin": likelihoods["target"]-likelihoods["background"],
            "target_vs_zero_margin": likelihoods["target"]-likelihoods["zero"],
            "target_vs_other_margin": likelihoods["target"]-likelihoods["other"],
            "background_explanation_ratio": likelihoods["background"]/max(likelihoods["target"], 1e-6)}

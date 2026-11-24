from __future__ import annotations

from typing import Any
import numpy as np

from driveaudit_eval.research.consistency.hypothesis_verification.background_hypothesis import build as background_build
from driveaudit_eval.research.consistency.hypothesis_verification.other_motion_hypothesis import build as other_build
from driveaudit_eval.research.consistency.hypothesis_verification.robust_likelihood import evaluate
from driveaudit_eval.research.consistency.hypothesis_verification.target_hypothesis import build as target_build
from driveaudit_eval.research.consistency.hypothesis_verification.zero_motion_hypothesis import build as zero_build


def compete(observed:dict[str,Any],source_calibration:dict[str,Any],target_calibration:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    vectors=observed["observed_flow_vectors"]
    target=target_build(observed,config["target_hypothesis"])
    background=background_build(observed,source_calibration,target_calibration,config["background_hypothesis"],tuple(config["image"]["size"]))
    zero=zero_build(observed,config["zero_hypothesis"]);other=other_build(observed,config["other_hypothesis"])
    robust=config["robust_likelihood"]
    target_stats=evaluate(vectors,target["target_expected_flow_per_point"],target["target_model_uncertainty"],robust,
                          prior=.5 if target["target_hypothesis_status"]!="valid" else 1.0)
    background_rows=[]
    for candidate in background["background_hypotheses"]:
        stats=evaluate(vectors,candidate["expected_flow_per_point"],float(config["background_hypothesis"]["model_uncertainty_px"]),robust)
        background_rows.append({**candidate,**stats})
    best=max(background_rows,key=lambda row:row["likelihood"]) if background_rows else {"likelihood":0.0,"depth_source":None}
    zero_stats=evaluate(vectors,zero["zero_expected_flow_per_point"],0.0,robust,prior=zero["zero_visual_prior"])
    other_stats=evaluate(vectors,other["other_expected_flow_per_point"],0.0,robust,penalty=other["other_complexity_penalty"])
    likelihoods={"target":target_stats["likelihood"],"background":best["likelihood"],"zero":zero_stats["likelihood"],"other":other_stats["likelihood"]}
    competitor=max(likelihoods[name] for name in ("background","zero","other"));winner=max(likelihoods,key=likelihoods.get)
    return {**target,"target_stats":target_stats,"background_hypotheses":background_rows,
            "best_background_depth_hypothesis":best.get("depth_source"),"best_background_stats":best,
            **zero,"zero_stats":zero_stats,**other,"other_stats":other_stats,"hypothesis_likelihoods":likelihoods,
            "winning_hypothesis":winner,"target_likelihood":likelihoods["target"],"best_background_likelihood":likelihoods["background"],
            "zero_motion_likelihood":likelihoods["zero"],"other_motion_likelihood":likelihoods["other"],
            "target_hypothesis_margin":likelihoods["target"]-competitor,
            "target_vs_background_margin":likelihoods["target"]-likelihoods["background"],
            "target_vs_zero_margin":likelihoods["target"]-likelihoods["zero"],
            "target_vs_other_margin":likelihoods["target"]-likelihoods["other"],
            "target_inlier_ratio":target_stats["inlier_ratio"],"background_explanation_ratio":likelihoods["background"]/max(likelihoods["target"],1e-6),
            "simple_ring_residual_for_ranking":False}


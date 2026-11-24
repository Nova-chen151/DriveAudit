from __future__ import annotations

from typing import Any
import numpy as np


def huber(values:np.ndarray,delta:float)->np.ndarray:
    absolute=np.abs(values);return np.where(absolute<=delta,.5*absolute**2,delta*(absolute-.5*delta))


def tukey(values:np.ndarray,c:float)->np.ndarray:
    scaled=values/c;return np.where(np.abs(scaled)<1,(c*c/6)*(1-(1-scaled**2)**3),c*c/6)


def trimmed_mean(values:np.ndarray,fraction:float)->float:
    if not len(values):return float("inf")
    count=max(1,int(np.ceil(len(values)*(1-fraction))));return float(np.mean(np.sort(values)[:count]))


def evaluate(observed:list[list[float]],expected:list[list[float]],uncertainty:float,config:dict[str,Any],prior:float=1.0,penalty:float=0.0)->dict[str,Any]:
    observed=np.asarray(observed,float);expected=np.asarray(expected,float);count=min(len(observed),len(expected))
    if not count:return {"robust_residual":None,"normalized_residual":None,"inlier_ratio":0.0,"median_angle_error":None,"magnitude_error":None,"likelihood":0.0,"uncertainty":uncertainty}
    observed,expected=observed[:count],expected[:count];finite=np.isfinite(observed).all(axis=1)&np.isfinite(expected).all(axis=1)
    observed,expected=observed[finite],expected[finite]
    if not len(observed):return {"robust_residual":None,"normalized_residual":None,"inlier_ratio":0.0,"median_angle_error":None,"magnitude_error":None,"likelihood":0.0,"uncertainty":uncertainty}
    residual=np.linalg.norm(observed-expected,axis=1)
    scale=max(float(config["base_scale_px"])+float(uncertainty),1e-6);normalized=residual/scale
    robust=float(np.median([np.median(huber(normalized,float(config["huber_delta"]))),
                            np.median(tukey(normalized,float(config["tukey_c"]))),trimmed_mean(normalized,float(config["trim_fraction"]))]))
    inlier=float(np.mean(normalized<=float(config["inlier_threshold"])))
    dot=np.sum(observed*expected,axis=1);norm=np.linalg.norm(observed,axis=1)*np.linalg.norm(expected,axis=1)
    angles=np.arccos(np.clip(np.divide(dot,norm,out=np.ones_like(dot),where=norm>1e-6),-1,1))
    magnitude=float(np.median(np.abs(np.linalg.norm(observed,axis=1)-np.linalg.norm(expected,axis=1))))
    likelihood=float(np.clip(np.exp(-robust-penalty)*(.25+.75*inlier)*prior,0,1))
    return {"robust_residual":float(np.median(residual)),"normalized_residual":robust,"inlier_ratio":inlier,
            "median_angle_error":float(np.median(angles)),"magnitude_error":magnitude,"likelihood":likelihood,"uncertainty":uncertainty}

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def build(observed:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    old=np.asarray(observed["observed_points"],float);new=np.asarray(observed["observed_target_points"],float)
    matrix=None
    if len(old)>=3:matrix,_=cv2.estimateAffine2D(old,new,method=cv2.LMEDS)
    if matrix is not None and not np.isfinite(matrix).all(): matrix=None
    if matrix is None:
        translation=np.median(new-old,axis=0) if len(old) else np.zeros(2);expected=np.tile(translation,(len(old),1));parameters=None;complexity=2
    else:
        predicted=np.column_stack([old,np.ones(len(old))])@matrix.T;expected=predicted-old;parameters=matrix.tolist();complexity=6
    return {"other_expected_flow_per_point":expected.tolist(),"other_affine_parameters":parameters,
            "other_complexity_penalty":float(config["complexity_penalty_per_parameter"])*complexity,
            "motion_cluster_count":observed.get("motion_cluster_count",0),"unexplained_motion_ratio":None}

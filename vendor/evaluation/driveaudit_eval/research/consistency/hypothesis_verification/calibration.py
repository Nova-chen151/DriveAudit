from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json

PRESENCE_FEATURES=("presence_confidence","core_point_fraction","core_background_contrast","foreground_boundary_score","contamination_probability","blur_score","contrast_score")
MOTION_FEATURES=("target_likelihood","best_background_likelihood","zero_motion_likelihood","other_motion_likelihood","target_hypothesis_margin","target_inlier_ratio","target_observed_cosine","expected_motion_magnitude","observed_motion_magnitude","projected_area")


def calibrate(rows:list[dict[str,Any]],calibration_scenes:set[str],config:dict[str,Any],output:str|Path)->dict[str,Any]:
    count=len(calibration_scenes);minimum=int(config["calibration"]["min_scenes_experimental"]);output=Path(output)
    if count<minimum:
        result={"status":"experimental_not_fitted_insufficient_scenes","independent_calibration_scenes":count,
                "presence_model_status":"not_fitted","motion_model_status":"not_fitted","holdout_read":False,
                "minimum_scenes_experimental":minimum,"minimum_scenes_provisional":int(config["calibration"]["min_scenes_provisional"])}
        write_json(output/"calibration_status.json",result);return result
    training=[row for row in rows if row.get("physical_scene_id") in calibration_scenes]
    presence=_fit(training,PRESENCE_FEATURES,"presence_label");motion=_fit(training,MOTION_FEATURES,"motion_label")
    result={"status":"experimental_fitted","independent_calibration_scenes":count,"presence_model_status":presence["status"],
            "motion_model_status":motion["status"],"presence_coefficients":presence.get("coefficients"),"motion_coefficients":motion.get("coefficients"),"holdout_read":False}
    write_json(output/"calibration_status.json",result);return result


def _fit(rows,features,label):
    selected=[row for row in rows if row.get(label) in {0,1}]
    if len({row[label] for row in selected})<2:return {"status":"not_fitted_single_class"}
    x=np.asarray([[float(row.get(key) or 0) for key in features] for row in selected]);y=np.asarray([row[label] for row in selected])
    model=Pipeline([("standardizer",StandardScaler()),("classifier",LogisticRegression(class_weight="balanced",max_iter=1000,random_state=2026))]);model.fit(x,y)
    return {"status":"fitted","coefficients":model.named_steps["classifier"].coef_[0].tolist()}


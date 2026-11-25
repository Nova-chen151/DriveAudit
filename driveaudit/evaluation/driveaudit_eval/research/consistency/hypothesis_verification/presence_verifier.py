from __future__ import annotations

from typing import Any
import numpy as np


def verify(observed:dict[str,Any],contamination:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    if observed["image_quality_status"].startswith("unscorable"):
        return _result("presence_unscorable",0.0,"image_quality",observed,contamination)
    core=float(observed.get("core_support_ratio",0));outside=float(observed.get("roi_support_ratio",0));boundary=float(observed.get("boundary_support_ratio",0))
    contrast=float(np.clip(.5+.5*(core-outside),0,1));appearance=float(observed.get("feature_support_score",0));motion_boundary=float(np.clip(boundary-outside+.5,0,1))
    cues=[contrast>=float(config["minimum_core_background_contrast"]),boundary>=float(config["minimum_boundary_score"]),
          observed.get("core_point_count",0)>=int(config["minimum_core_points"]),appearance>=float(config["minimum_appearance_support"])]
    confidence=float(np.mean([contrast,boundary,min(1.0,observed.get("core_point_count",0)/max(1,int(config["minimum_core_points"]))),appearance]))
    if contamination["contamination_probability"]>=float(config["maximum_contamination_probability"]):
        return _result("presence_ambiguous",confidence,"contamination",observed,contamination,contrast,motion_boundary)
    if sum(cues)>=int(config["minimum_independent_cues"]):state,reason="presence_supported","multiple_object_level_cues"
    elif confidence<float(config["contradiction_confidence"]):state,reason="presence_contradicted","insufficient_object_level_evidence"
    else:state,reason="presence_ambiguous","insufficient_independent_cues"
    return _result(state,confidence,reason,observed,contamination,contrast,motion_boundary)


def _result(state,confidence,reason,observed,contamination,contrast=0.0,motion_boundary=0.0):
    return {"presence_state":state,"presence_confidence":confidence,"presence_reason":reason,
            "foreground_boundary_score":float(observed.get("boundary_support_ratio",0)),"core_background_contrast":contrast,
            "appearance_template_similarity":float(observed.get("feature_support_score",0)),"motion_boundary_score":motion_boundary,
            "multiview_presence_support":None,"presence_scorable":state!="presence_unscorable",
            "presence_supported":state=="presence_supported","presence_contradicted":state=="presence_contradicted",
            "presence_ambiguous":state=="presence_ambiguous","presence_unscorable":state=="presence_unscorable"}


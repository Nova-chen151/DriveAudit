from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, finite_mean, rate, write_json


def select(rows:list[dict[str,Any]],config:dict[str,Any],output:str|Path)->dict[str,Any]:
    grouped=defaultdict(list)
    for row in rows:grouped[(row["camera_name"],row["parameter_profile"])].append(row)
    selected={};diagnostics={}
    for camera in CAMERAS:
        candidates=[]
        for profile in config["tracking"]["parameter_profiles"]:
            values=grouped.get((camera,profile),[]);usable=[row for row in values if row.get("guided_support_state")!="unscorable"]
            metric={"profile":profile,"history_sample_count":len(values),"history_valid_point_ratio":finite_mean(row.get("guided_valid_point_ratio") for row in values),
                    "history_flow_cosine":finite_mean(row.get("guided_flow_cosine") for row in usable),"history_direction_agreement":finite_mean(row.get("guided_motion_direction_agreement") for row in usable),
                    "history_epe":finite_mean(row.get("guided_residual_epe") for row in usable),"history_support":rate(row.get("guided_support_state")=="supported" for row in usable)}
            candidates.append(metric)
        def key(value):
            numeric=lambda name,default:float(value[name]) if isinstance(value.get(name),(int,float)) else default
            return (numeric("history_valid_point_ratio",-1),numeric("history_flow_cosine",-1),numeric("history_direction_agreement",-1),-numeric("history_epe",1e9))
        winner=max(candidates,key=key);parameters={**config["tracking"]["global_parameters"],**config["tracking"]["parameter_profiles"][winner["profile"]]}
        selected[camera]={"profile":winner["profile"],"parameters":parameters,"history_metrics":winner};diagnostics[camera]=candidates
    result={"source_frames":[0,9],"future_evaluation_frames_read":False,"selection_rule":"maximize valid-point ratio, then cosine, direction, then minimize EPE",
            "search_space":config["tracking"]["parameter_profiles"],"selected_parameters":selected,"all_candidates":diagnostics}
    output=Path(output);path=output/"history_calibration/per_camera_parameters.yaml";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(result,sort_keys=False),encoding="utf-8");write_json(output/"history_calibration/per_camera_parameters.json",result);return result


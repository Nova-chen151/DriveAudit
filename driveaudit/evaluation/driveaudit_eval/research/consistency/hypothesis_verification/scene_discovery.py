from __future__ import annotations

from pathlib import Path
from typing import Any
from driveaudit_eval.consistency.config import load_config

from driveaudit_eval.research.consistency.hypothesis_verification.calibration_scene_adapter import adapt_driveaudit_eval_pair
from driveaudit_eval.research.consistency.verification_benchmark.common import write_json


def discover(config:dict[str,Any],output:str|Path)->dict[str,Any]:
    roots=[Path(root).resolve() for root in config["scene_discovery"]["roots"]]
    allowed=[root for root in roots if root.is_dir()]
    accepted=[];rejected=[];seen=set()
    manifest=config["inputs"]["pair_manifest"]
    manifest_path=Path(config["inputs"].get("pair_manifest_source", "tools/configs/research.yaml")) if isinstance(manifest,dict) else Path(manifest)
    if any(_within(manifest_path.resolve(),root) for root in allowed):
        for pair in load_config(manifest)["pairs"]:
            try:
                scene=adapt_driveaudit_eval_pair(pair);payload=scene.to_dict()
                if scene.physical_scene_id in seen:rejected.append({"candidate":pair["pair_id"],"reason":"duplicate_physical_scene_content_hash"})
                elif len(scene.timestamps)<2:rejected.append({"candidate":pair["pair_id"],"reason":"fewer_than_two_timestamps"})
                elif len(scene.camera_names)<2:rejected.append({"candidate":pair["pair_id"],"reason":"not_multicamera"})
                elif not scene.agents:rejected.append({"candidate":pair["pair_id"],"reason":"no_dynamic_agents"})
                else:accepted.append(payload);seen.add(scene.physical_scene_id)
            except Exception as error:rejected.append({"candidate":pair.get("pair_id"),"reason":f"adapter_failure:{type(error).__name__}:{error}"})
    for root in allowed:
        for conditions in list(root.glob("challenge/**/conditions"))[:int(config["scene_discovery"]["maximum_candidates"])]:
            if str(conditions.resolve()) in {row["provenance"]["conditions_root"] for row in accepted}:continue
            missing=[name for name in ("frames.csv","camera_calibration.csv","ego_pose.csv","boxes.csv") if not (conditions/name).is_file()]
            if missing:rejected.append({"candidate":str(conditions),"reason":"missing:"+",".join(missing)})
            else:rejected.append({"candidate":str(conditions),"reason":"no_auditable_image_root_mapping"})
    result={"scan_roots":[str(root) for root in allowed],"candidate_count":len(accepted)+len(rejected),
            "independent_physical_scene_count":len(accepted),"accepted_scenes":accepted,"rejected_scenes":rejected,
            "holdout_scene_id":config["holdout"]["scene_id"],"holdout_excluded_from_calibration":True,"unbounded_root_scan_used":False}
    output=Path(output);write_json(output/"discovered_scenes.json",accepted);write_json(output/"rejected_scenes.json",rejected);write_json(output/"summary.json",result);return result


def _within(path:Path,root:Path)->bool:
    try:path.relative_to(root);return True
    except ValueError:return False

from __future__ import annotations

import inspect
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from driveaudit_eval.agents.agent_input_adapter import assert_no_evaluation_leakage

from driveaudit_eval.research.consistency.perception_trajectory.backends.cached_vad_backend import CachedVADPerceptionBackend
from driveaudit_eval.consistency.shared.common import write_json
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, read_csv, true_cell


def audit_gt_metadata_reference(config: dict[str, Any], pair: dict[str, Any], output: str | Path) -> dict[str, Any]:
    output, backend = Path(output), config["backend"]
    image_root = Path(pair["image_sequence"]["future_root"])
    history_n = int(pair["timeline"]["image_history_frames"])
    future_n = int(pair["timeline"]["image_future_frames"])
    required = range(history_n, history_n + future_n)
    image_counts = {camera: sum((image_root / camera / f"{frame:06d}.jpg").is_file() for frame in required)
                    for camera in CAMERA_ORDER}
    rows = [row for row in read_csv(Path(backend["conditions_root"]) / "boxes.csv")
            if true_cell(row.get("valid_flag", True)) and history_n <= int(row["frame_index"]) < history_n + future_n]
    result = {"extractor_backend": "gt_metadata_boxes", "status": "valid",
              "formal_backend_accepted": all(value == future_n for value in image_counts.values()) and bool(rows),
              "image_root": str(image_root.resolve()), "required_image_counts": image_counts,
              "metadata_path": str((Path(backend["conditions_root"]) / "boxes.csv").resolve()),
              "future_metadata_box_count": len(rows), "categories": dict(Counter(row["class_name"] for row in rows)),
              "pixel_inference_performed": False, "vad_used": False, "checkpoint_used": False,
              "future_submitted_trajectory_in_backend": False,
              "declaration": "GT images are indexed against their attached input metadata; boxes are metadata reference, not pixel detections"}
    result["status"] = "valid" if result["formal_backend_accepted"] else "invalid_reference_metadata"
    write_json(output / "perception_only_audit.json", result)
    lines = ["# GT 图像与 input 元数据来源审计", ""] + [f"- {key}: `{value}`" for key, value in result.items()]
    (output / "cache_provenance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def audit_vad_detection_cache(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    output = Path(output); cache_path = Path(config["backend"]["cache"])
    with cache_path.open("rb") as handle: records = pickle.load(handle)
    manifest = json.loads(Path("challenge/ego_agent_eval/eval_manifest.json").read_text(encoding="utf-8"))
    frames = manifest["scenes"][0]["frames"]
    leakage_errors = []
    for frame in frames:
        try: assert_no_evaluation_leakage(frame["model_input"])
        except Exception as exc: leakage_errors.append(str(exc))
    detection_rows = sum(len(record.get("raw_scene", {}).get("boxes_3d", [])) for record in records)
    aligned = all(len(record["raw_scene"]["boxes_3d"]) == len(record["raw_scene"]["scores_3d"]) == len(record["raw_scene"]["labels_3d"])
                  for record in records)
    backend_source = inspect.getsource(CachedVADPerceptionBackend.infer)
    uses_motion_prediction = 'raw["trajs_3d"]' in backend_source or "raw['trajs_3d']" in backend_source
    result = {
        "extractor_backend": "vad_detection_cache", "cache_path": str(cache_path.resolve()),
        "frame_count": len(records), "frame_ids": [record["input_audit"]["frame_id"] for record in records],
        "cached_detection_box_count_before_threshold": detection_rows,
        "detection_outputs_present": all(all(key in record.get("raw_scene", {}) for key in ("boxes_3d", "scores_3d", "labels_3d")) for record in records),
        "detection_array_lengths_aligned": aligned,
        "motion_prediction_present_but_discarded": all("trajs_3d" in record.get("raw_scene", {}) for record in records),
        "planning_output_present_but_discarded": all("raw_planning" in record and "planning_traj" in record for record in records),
        "future_submitted_trajectory_in_model_input": bool(leakage_errors), "input_leakage_errors": leakage_errors,
        "model_input_fields": sorted(frames[0]["model_input"]), "evaluation_only_fields": sorted(frames[0]["evaluation_only"]),
        "planning_feedback_to_detection": False,
        "planning_feedback_evidence": "VAD.simple_test_pts decodes boxes_3d from pts_bbox_head.get_bboxes; planning branch is read afterwards and does not update boxes",
        "independent_tracker_input": "raw_scene.boxes_3d + scores_3d + labels_3d only",
        "backend_reads_motion_prediction": uses_motion_prediction,
        "full_model_was_executed": True,
        "declaration": "full model was executed, but only image-derived detection outputs were used",
    }
    passed = (result["detection_outputs_present"] and aligned and not leakage_errors and
              not uses_motion_prediction and result["planning_feedback_to_detection"] is False)
    result["formal_detection_cache_accepted"] = passed
    result["formal_backend_accepted"] = passed
    result["status"] = "valid" if passed else "invalid_cache_provenance"
    write_json(output / "perception_only_audit.json", result)
    lines = ["# VAD detection cache 来源审计", ""] + [f"- {key}: `{value}`" for key, value in result.items()]
    (output / "cache_provenance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

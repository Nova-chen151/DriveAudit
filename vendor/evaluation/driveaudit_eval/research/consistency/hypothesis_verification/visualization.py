from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc.common import image_path


def render_holdout_examples(pair: dict[str, Any], rows: list[dict[str, Any]], output: str | Path, limit: int = 24) -> dict[str, Any]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row.get("target_hypothesis_margin", 0.0))
    chosen = (ordered[:limit // 2] + ordered[-limit // 2:]) if ordered else []
    for index, row in enumerate(chosen):
        image = cv2.imread(str(image_path(pair, row["camera_name"], int(row["frame_index"]))))
        if image is None: continue
        if image.shape[1::-1] != (400, 224): image = cv2.resize(image, (400, 224))
        polygon = np.asarray(row["target_polygon"], np.int32); cv2.polylines(image, [polygon], True, (0, 255, 255), 1)
        for point, observed, target in zip(row.get("observed_points", []), row.get("observed_flow_vectors", []), row.get("target_expected_flow_per_point", [])):
            p = tuple(np.round(point).astype(int)); q = tuple(np.round(np.asarray(point) + observed).astype(int)); t = tuple(np.round(np.asarray(point) + target).astype(int))
            cv2.arrowedLine(image, p, q, (0, 255, 0), 1, tipLength=.3); cv2.arrowedLine(image, p, t, (0, 0, 255), 1, tipLength=.3)
        label = f"{row['motion_state']} win={row['winning_hypothesis']} margin={row['target_hypothesis_margin']:.3f}"
        cv2.putText(image, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f"hypothesis_{index:03d}_{row['camera_name']}.jpg"), image)
    return {"status": "written", "count": len(chosen)}


def export_human_audit(pair: dict[str, Any], rows: list[dict[str, Any]], output: str | Path, limit: int = 50,
                       *, render_media: bool = True) -> dict[str, Any]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda row: row.get("target_hypothesis_margin", 0.0))
    if len(selected) > limit:
        indices = np.linspace(0, len(selected) - 1, limit).astype(int); selected = [selected[i] for i in indices]
    fields = ("sample_id", "agent_id", "camera_name", "frame_index", "target_frame_index", "target_likelihood",
              "best_background_likelihood", "zero_motion_likelihood", "other_motion_likelihood", "target_hypothesis_margin",
              "presence_state", "motion_state", "hard_rule_triggers", "contamination_probability",
              "human_target_present_1_5", "human_image_judgeable_1_5", "human_motion_matches_1_5",
              "human_target_vs_background_1_5", "human_static_repeat_1_5", "human_neighbor_contamination_1_5",
              "human_presence_reasonable_1_5", "human_motion_reasonable_1_5")
    with (output / "annotation_template_h.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for index, row in enumerate(selected):
            record = {key: row.get(key, "") for key in fields}; record["sample_id"] = f"tcvv_h_{index:04d}"
            writer.writerow(record)
    (output / "status.json").write_text('{"human_validation_status":"pending_annotation","correlation_computed":false}', encoding="utf-8")
    if render_media:
        render_holdout_examples(pair, selected, output / "overlays", limit)
    return {"human_validation_status": "pending_annotation", "exported_samples": len(selected), "correlation_computed": False}

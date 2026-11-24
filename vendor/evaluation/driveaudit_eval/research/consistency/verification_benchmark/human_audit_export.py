from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc.common import image_path

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json


def export(pair: dict[str, Any], rows: list[dict[str, Any]], output: str | Path, count: int = 24,
           *, render_media: bool = True) -> dict[str, Any]:
    output = Path(output); clips = output / "clips"
    (clips if render_media else output).mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row.get("tcvv_rule_support", 0)))
    representatives = []
    for family in sorted({row.get("negative_family") for row in ordered if row.get("negative_family")}):
        representatives.append(next(row for row in ordered if row.get("negative_family") == family))
    fill_count = max(0, min(count, len(ordered)) - len(representatives))
    quantiles = [ordered[index] for index in np.linspace(0, max(0, len(ordered) - 1), fill_count, dtype=int)] if ordered and fill_count else []
    selected = (representatives + quantiles)[:count]
    annotation_rows = []
    for number, row in enumerate(selected):
        sample_id = f"audit_{number:04d}"
        images = [cv2.imread(str(image_path(pair, row["camera_name"], int(row[key])))) for key in ("frame_index", "target_frame_index")]
        images = [cv2.resize(image, (400, 224)) for image in images if image is not None]
        if len(images) != 2: continue
        raw_path, overlay_path = clips / f"{sample_id}_raw.mp4", clips / f"{sample_id}_overlay.mp4"
        if render_media:
            _video(raw_path, images); overlay = [_overlay(image.copy(), row) for image in images]; _video(overlay_path, overlay)
        annotation_rows.append({"sample_id": sample_id, "annotator_id": "", "target_exists_1_5": "", "position_consistency_1_5": "",
                                "motion_direction_1_5": "", "motion_magnitude_1_5": "", "temporal_consistency_1_5": "",
                                "overall_consistency": "", "difficult_to_judge": "", "failure_reason": "",
                                "automatic_score": row.get("tcvv_rule_support"), "camera": row["camera_name"],
                                "agent_id": row["agent_id"], "timestamp": row.get("timestamp"),
                                "negative_family": row.get("negative_family"), "severity": row.get("severity"),
                                "raw_clip": str(raw_path) if render_media else "", "overlay_clip": str(overlay_path) if render_media else ""})
        if not render_media:
            annotation_rows[-1].update({key+'_source_path': str(image_path(pair, row['camera_name'], int(row[key])))
                                       for key in ('frame_index', 'target_frame_index')})
    fields = list(annotation_rows[0]) if annotation_rows else ["sample_id"]
    with (output / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(annotation_rows)
    (output / "instructions.md").write_text(
        "# TCVV-Bench human audit\n\n" + ("Review the raw clip first, then the overlay. " if render_media else
        "Read the original images at frame_index_source_path and target_frame_index_source_path; no media copies were exported. ") +
        "Score presence, position, motion direction, motion magnitude, timing, and overall consistency from 1 (inconsistent) to 5 (consistent). Mark difficult cases instead of guessing. Do not use the automatic score when judging.\n",
        encoding="utf-8")
    result = {"status": "pending_annotation", "exported_sample_count": len(annotation_rows), "annotator_count": 0,
              "automatic_human_correlation": None, "selection": "negative-family representatives plus score-stratified windows"}
    write_json(output / "export_manifest.json", result); return result


def _video(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (frames[0].shape[1], frames[0].shape[0]))
    for _ in range(4):
        for frame in frames: writer.write(frame)
    writer.release()


def _overlay(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, row["source_projected_box"]); cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
    center = ((x1 + x2) // 2, (y1 + y2) // 2); expected = np.asarray(row.get("expected_relative_flow", [0, 0]), int)
    observed = np.asarray(row.get("observed_relative_flow", [0, 0]), int)
    cv2.arrowedLine(image, center, tuple((np.asarray(center) + expected).tolist()), (0, 255, 0), 2)
    cv2.arrowedLine(image, center, tuple((np.asarray(center) + observed).tolist()), (0, 0, 255), 2)
    cv2.putText(image, f"{row['camera_name']} score={row.get('tcvv_rule_support', 0):.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return image

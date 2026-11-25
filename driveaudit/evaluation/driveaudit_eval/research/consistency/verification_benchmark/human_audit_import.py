from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from driveaudit_eval.research.consistency.verification_benchmark.common import write_json


def import_annotations(path: str | Path, output: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        result = {"status": "pending_annotation", "annotator_count": 0, "correlations": None}
        write_json(Path(output) / "human_validation.json", result); return result
    with path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row.get("overall_consistency")]
    annotators = sorted({row.get("annotator_id") for row in completed if row.get("annotator_id")})
    if not completed:
        result = {"status": "pending_annotation", "annotator_count": len(annotators), "correlations": None}
    else:
        auto = np.asarray([float(row["automatic_score"]) for row in completed]); human = np.asarray([float(row["overall_consistency"]) for row in completed])
        result = {"status": "completed", "annotation_count": len(completed), "annotator_count": len(annotators),
                  "pearson": float(stats.pearsonr(auto, human).statistic), "spearman": float(stats.spearmanr(auto, human).statistic),
                  "kendall": float(stats.kendalltau(auto, human).statistic),
                  "krippendorff_alpha": "not_available_dependency", "fleiss_kappa": "not_available_requires_balanced_raters"}
    write_json(Path(output) / "human_validation.json", result); return result


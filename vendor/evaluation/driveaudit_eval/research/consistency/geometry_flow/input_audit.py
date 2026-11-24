from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.observed_flow.gt_data_audit import audit as v1_audit
from driveaudit_eval.research.consistency.observed_flow.schema import sha256, write_json


def audit(config: dict[str, Any], v1_config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    result = v1_audit(v1_config, Path(output) / "input_audit")
    result.update(version=config["version"], uses_submission_data=False, uses_bevformer=False,
                  uses_tracker=False, uses_planning_pdms_closed_loop=False,
                  v1_result_sha256=sha256(Path(config["inputs"]["gt_vtc_v1_results"]) / "final_metrics.json"))
    write_json(Path(output) / "input_audit/input_audit.json", result); return result


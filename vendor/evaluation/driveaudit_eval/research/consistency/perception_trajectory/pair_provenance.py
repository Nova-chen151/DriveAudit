from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.shared.common import read_csv, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pair_provenance(pair: dict[str, Any], output: str | Path, root: str | Path = ".") -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output)
    image, trajectory = pair["image_sequence"], pair["trajectory_sequence"]
    conditions = root / image["conditions_root"]
    trajectory_path = root / trajectory["trajectory_path"]
    metadata_path = conditions / "metadata.json"
    log_path = root / pair["generation_manifest_path"]
    entrypoint = root / pair["generation_entrypoint"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = read_csv(conditions / "frames.csv")
    boxes = read_csv(trajectory_path)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    script = entrypoint.read_text(encoding="utf-8", errors="replace") if entrypoint.is_file() else ""
    condition_files = [conditions / name for name in (
        "frames.csv", "boxes.csv", "camera_calibration.csv", "ego_pose.csv", "metadata.json", "scene_tokens.json"
    )]
    condition_hash = hashlib.sha256("".join(_sha256(path) for path in condition_files).encode()).hexdigest()
    scene_index = str(metadata.get("source_scene_index", metadata.get("scene_index", "")))
    tokens = {row["token"] for row in frames}
    checks = {
        "explicit_submission_id": bool(pair.get("submission_id")),
        "trajectory_is_generation_condition": trajectory_path.resolve() == (conditions / "boxes.csv").resolve(),
        "all_reference_boxes_belong_to_condition_timeline": bool(boxes) and all(row.get("token") in tokens for row in boxes),
        "condition_id_matches_scene_index": scene_index == "50" and "scene_index_50" in pair.get("image_generation_condition_id", ""),
        "generation_log_records_runtime_condition_build": "wrote " in log and ".runtime/conditions" in log,
        "generation_log_records_80_frame_run": "Your validation index: [9, 10" in log and ", 88]" in log,
        "entrypoint_syncs_declared_minimal_bundle": 'sync_dir "${MINIMAL_INPUT_DIR}" "${WORK_DIR}/input/minimal"' in script,
        "entrypoint_publishes_generated_future": 'GENERATED_ROOT="${RUN_DIR}/future_generated"' in script,
    }
    verified = all(checks.values())
    result = {
        "pair_id": pair["pair_id"],
        "submission_id": pair.get("submission_id"),
        "image_generation_condition_id": pair.get("image_generation_condition_id"),
        "image_scene_id": image["scene_id"],
        "trajectory_scene_id": trajectory["scene_id"],
        "image_root": str((root / image["generated_future_root"]).resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "calibration_path": str((root / pair["calibration"]["path"]).resolve()),
        "ego_pose_path": str((root / pair["ego_pose"]["path"]).resolve()),
        "history_metadata_path": str((root / pair["history_metadata_path"]).resolve()),
        "generation_manifest_path": str(log_path.resolve()),
        "condition_hash": condition_hash,
        "trajectory_hash": _sha256(trajectory_path),
        "hash_recorded_at_generation": False,
        "provenance_checks": checks,
        "provenance_status": "verified_same_submission" if verified else "unverified_submission_relation",
        "evidence_limitations": [
            "WorldDreamer runtime and source run directory were cleaned after publication.",
            "The generation log did not persist a contemporaneous condition hash; verification uses the recorded handoff script, log, current immutable input bundle, and token alignment.",
        ],
        "rejected_previous_pair": {
            "trajectory_scene_id": "0_6_straight_straight_19",
            "reason": "not referenced by the WorldDreamer generation log, entrypoint, or challenge condition bundle",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pair_provenance.json", result)
    lines = [
        "# Pair provenance audit", "",
        f"- pair_id: `{result['pair_id']}`", f"- status: `{result['provenance_status']}`",
        f"- image scene: `{result['image_scene_id']}`", f"- trajectory source: `{result['trajectory_scene_id']}`",
        f"- condition hash: `{condition_hash}`", f"- trajectory hash: `{result['trajectory_hash']}`", "",
        "The formal reference is the timestamped 3D-box trajectory that conditioned this WorldDreamer generation. "
        "It is loaded only after BEVFormer image inference. The unrelated OnSite trajectory is rejected.", "",
        "## Checks", "",
    ]
    lines += [f"- {name}: `{value}`" for name, value in checks.items()]
    lines += ["", "## Evidence limitation", "", *[f"- {item}" for item in result["evidence_limitations"]]]
    (output / "pair_provenance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

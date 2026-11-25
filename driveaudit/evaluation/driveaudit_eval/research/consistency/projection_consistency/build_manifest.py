from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.projection_consistency.schema import read_json, sha256, write_json


def _scene_records(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    return {str(row["scene_id"]): row for row in data.get("scenes", [])}


def build(config: dict[str, Any], output: str | Path | None = None, scene_ids: list[str] | None = None) -> dict[str, Any]:
    fixed_path = Path(config["dataset"]["manifest"])
    fixed = read_json(fixed_path)
    fixed_ids = list(fixed["scene_ids"])
    selected = scene_ids or fixed_ids
    if config["dataset"].get("fixed_scene_count") and len(fixed_ids) != int(config["dataset"]["fixed_scene_count"]):
        raise ValueError(f"fixed manifest requires {config['dataset']['fixed_scene_count']} scenes, got {len(fixed_ids)}")
    if not set(selected).issubset(fixed_ids):
        raise ValueError(f"requested scenes are outside the frozen manifest: {sorted(set(selected) - set(fixed_ids))}")
    candidate_root = Path(config["inputs"]["candidate_image_root"])
    trajectory_root = Path(config["inputs"]["trajectory_root"])
    candidate_manifest = candidate_root.parent / "submission_manifest.json"
    trajectory_manifest = trajectory_root.parent / "reference_manifest.json"
    if not candidate_manifest.is_file():
        raise FileNotFoundError(f"explicit candidate manifest is required: {candidate_manifest}")
    if not trajectory_manifest.is_file():
        raise FileNotFoundError(f"explicit trajectory/reference manifest is required: {trajectory_manifest}")
    candidate = _scene_records(candidate_manifest)
    trajectory = _scene_records(trajectory_manifest)
    fixed_records = {row["scene_name"]: row for row in fixed.get("scenes", [])}
    pairs = []
    for scene_id in selected:
        if scene_id not in candidate or scene_id not in trajectory or scene_id not in fixed_records:
            raise ValueError(f"scene lacks explicit manifest pairing: {scene_id}")
        source_tokens = {
            str(candidate[scene_id].get("source_scene_token")),
            str(trajectory[scene_id].get("source_scene_token")),
            str(fixed_records[scene_id].get("scene_token")),
        }
        if len(source_tokens) != 1:
            raise ValueError(f"source scene token mismatch for {scene_id}: {sorted(source_tokens)}")
        token = source_tokens.pop()
        pairs.append({
            "pair_id": f"nuscenes:{token}:{scene_id}",
            "scene_id": scene_id,
            "source_scene_token": token,
            "candidate_scene_id": scene_id,
            "trajectory_scene_id": scene_id,
            "candidate_scene_root": str(candidate_root / scene_id),
            "trajectory_scene_root": str(trajectory_root / scene_id),
            "pairing_source": "explicit_submission_and_reference_manifests",
        })
    result = {
        "protocol": config["version"],
        "scene_count": len(pairs),
        "fixed_manifest": str(fixed_path),
        "fixed_manifest_sha256": sha256(fixed_path),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "trajectory_manifest": str(trajectory_manifest),
        "trajectory_manifest_sha256": sha256(trajectory_manifest),
        "pairs": pairs,
    }
    if output is not None:
        write_json(Path(output) / "input_audit/manifest.json", result)
    return result

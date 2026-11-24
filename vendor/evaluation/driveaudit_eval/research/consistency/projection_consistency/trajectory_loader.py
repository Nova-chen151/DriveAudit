from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks

from driveaudit_eval.research.consistency.projection_consistency.schema import read_csv, read_json


def scene_root(config: dict[str, Any], scene_id: str) -> Path:
    return Path(config["inputs"]["trajectory_root"]) / scene_id


def conditions_root(config: dict[str, Any], scene_id: str) -> Path:
    root = scene_root(config, scene_id)
    candidate = root / "conditions"
    return candidate if candidate.is_dir() else root


def load(config: dict[str, Any], scene_id: str) -> dict[str, Any]:
    root = conditions_root(config, scene_id)
    tracks = condition_reference_tracks(root, 0, 90, 1e9)
    return {
        "scene_id": scene_id,
        "root": str(root),
        "tracks": tracks,
        "frames": read_csv(root / "frames.csv"),
        "boxes": read_csv(root / "boxes.csv"),
        "metadata": read_json(root / "metadata.json"),
    }


from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from driveaudit_eval.agents.core import WORLD_DREAMER_CAMERA_ORDER
CAMERAS = list(WORLD_DREAMER_CAMERA_ORDER)
DYNAMIC_CLASSES = {"car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle"}
CATEGORY_MAP = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus": "bus",
    "vehicle.trailer": "trailer", "vehicle.construction": "construction_vehicle",
    "human.pedestrian": "pedestrian", "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle", "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}


def category(name: str) -> str:
    return next((mapped for prefix, mapped in CATEGORY_MAP.items() if name.startswith(prefix)), "other")


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), separators=(",", ":")) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})
    os.replace(tmp, path)


def sha256(path: str | Path) -> str:
    path = Path(path); digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(str(path.resolve()).encode())
    elif path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    elif path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file() or p.is_symlink()):
            digest.update(str(item.relative_to(path)).encode()); digest.update(sha256(item).encode())
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()


def output_roots(config: dict[str, Any]) -> dict[str, Path]:
    name = config["dataset"]["name"]
    return {
        "input": Path("challenge/input") / name,
        "output": Path("challenge/output") / name / "gt_submission",
        "output_native": Path("challenge/output") / name / "gt_submission_native",
        "reference": Path("challenge/gt") / name,
        "evaluation": Path("challenge/eval_output") / name,
        "docs": Path("challenge/eval_output") / name / "reports",
    }

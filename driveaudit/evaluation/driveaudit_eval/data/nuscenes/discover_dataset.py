from __future__ import annotations

import importlib.metadata
import os
import shutil
from pathlib import Path

from .schema import write_json

ENV_KEYS = ("NUSCENES_ROOT", "NUSCENES_DATAROOT", "DATA_ROOT")
KNOWN_ROOTS = (
    "/media/yuhang/35d64c58-50a6-4088-bb4e-6a2cea99c35b/Nuscenes-data/nuscenes",
    "/home/yuhang/datasets/nuscenes", "/data/nuscenes",
)


def candidates(config: dict) -> list[Path]:
    configured = config["dataset"].get("dataroot", "auto")
    values = ([] if configured == "auto" else [configured]) + [os.environ.get(k) for k in ENV_KEYS] + list(KNOWN_ROOTS)
    return list(dict.fromkeys(Path(v).expanduser() for v in values if v))


def valid_root(path: Path) -> tuple[bool, list[str]]:
    required = ["samples", "sweeps", "maps", "v1.0-trainval"]
    missing = [name for name in required if not (path / name).exists()]
    if (path / "v1.0-test").exists() and not (path / "v1.0-trainval").exists():
        missing.append("trainval_required_test_forbidden")
    return not missing, missing


def discover(config: dict, output: Path) -> dict:
    checked = []
    selected = None
    for path in candidates(config):
        ok, missing = valid_root(path)
        checked.append({"path": str(path), "exists": path.exists(), "valid": ok, "missing": missing})
        if ok and selected is None:
            selected = path
    usage = shutil.disk_usage(Path.cwd())
    result = {
        "status": "completed" if selected else "blocked",
        "dataroot": str(selected) if selected else "not_available",
        "version": "v1.0-trainval" if selected else "not_available",
        "devkit_version": importlib.metadata.version("nuscenes-devkit"),
        "checked_locations": checked,
        "workspace_free_bytes": usage.free,
        "source_readable": bool(selected and os.access(selected, os.R_OK)),
        "source_writable": bool(selected and os.access(selected, os.W_OK)),
        "materialization_capabilities": {"symlink": True, "copy": True, "hardlink_same_filesystem_only": True},
    }
    if selected:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version="v1.0-trainval", dataroot=str(selected), verbose=False)
        result.update(scene_count=len(nusc.scene), sample_count=len(nusc.sample), sample_data_count=len(nusc.sample_data), map_count=len(nusc.map))
    write_json(output, result)
    return result


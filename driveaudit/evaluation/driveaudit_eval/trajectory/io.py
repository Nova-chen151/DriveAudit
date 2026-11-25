from __future__ import annotations

import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np

from driveaudit_eval.trajectory.schema import Trajectory


class RestrictedUnpickler(pickle.Unpickler):
    """Load plain containers and NumPy arrays without arbitrary globals."""

    ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("forbidden pickle global: %s.%s" % (module, name))


def safe_pickle_load(path):
    with Path(path).open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def derive_scalar_motion(positions, valid_mask, timestamps):
    n = len(timestamps)
    speed = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)
    adjacent = valid_mask[1:] & valid_mask[:-1]
    idx = np.flatnonzero(adjacent) + 1
    if len(idx):
        dt = timestamps[idx] - timestamps[idx - 1]
        speed[idx] = np.linalg.norm(positions[idx] - positions[idx - 1], axis=1) / dt
    triplets = valid_mask[2:] & valid_mask[1:-1] & valid_mask[:-2]
    idx = np.flatnonzero(triplets) + 2
    idx = idx[np.isfinite(speed[idx]) & np.isfinite(speed[idx - 1])]
    if len(idx):
        acceleration[idx] = (speed[idx] - speed[idx - 1]) / (timestamps[idx] - timestamps[idx - 1])
    return speed, acceleration


def _legacy_tracks(data, source, declarations):
    required = {"scene_name", "ids", "types", "shapes", "valid_mask", "positions", "headings"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("legacy pickle missing fields: %s" % ", ".join(missing))
    if "sim_freq" not in data:
        raise ValueError("legacy pickle does not declare sim_freq/dt")
    coordinate = declarations.get("coordinate_system")
    heading_unit = declarations.get("heading_unit")
    if not coordinate or not heading_unit:
        raise ValueError("legacy coordinate_system and heading_unit must be explicitly declared in config")
    ids = [str(value) for value in data["ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("agent_id values are not unique")
    positions = np.asarray(data["positions"], dtype=float)
    headings = np.asarray(data["headings"], dtype=float)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    types = np.asarray(data["types"])
    shapes = np.asarray(data["shapes"], dtype=float)
    if positions.shape[:2] != valid.shape or headings.shape != valid.shape:
        raise ValueError("legacy trajectory array shapes are inconsistent")
    dt = 1.0 / float(data["sim_freq"])
    timestamps = np.arange(valid.shape[1], dtype=float) * dt
    type_names = {0: "vehicle", 1: "pedestrian", 2: "bicycle"}
    ego_ids = set(str(v) for v in declarations.get("ego_ids", []))
    tracks = []
    for i, agent_id in enumerate(ids):
        speed, acceleration = derive_scalar_motion(positions[i], valid[i], timestamps)
        agent_type = "ego" if agent_id in ego_ids else type_names.get(int(types[i]), "unknown")
        metadata = {
            "dt": dt,
            "heading_unit": heading_unit,
            "history_frames": declarations.get("history_frames"),
            "predict": bool(np.asarray(data.get("predict_mask", np.ones(len(ids), dtype=bool)))[i]),
            "input_fields": sorted(data.keys()),
            "derived_fields": ["timestamps from sim_freq", "velocities from adjacent valid positions", "accelerations from adjacent valid speeds"],
            "declarations": {
                "coordinate_system": "config",
                "heading_unit": "config",
                "history_frames": "config" if declarations.get("history_frames") is not None else "missing",
            },
        }
        tracks.append(Trajectory(
            scene_id=str(data["scene_name"]), agent_id=agent_id, agent_type=agent_type, source=source,
            coordinate_system=coordinate, timestamps=timestamps.copy(), positions=positions[i].copy(),
            headings=headings[i].copy(), velocities=speed, accelerations=acceleration,
            valid_mask=valid[i].copy(), shape={"length": shapes[i, 0], "width": shapes[i, 1]}, metadata=metadata,
        ))
    return tracks


def load_tracks(path, source="generated", declarations=None):
    path = Path(path)
    declarations = declarations or {}
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return _legacy_tracks(safe_pickle_load(path), source, declarations)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("trajectories", [data])
        return [Trajectory.from_dict(dict(item, source=item.get("source", source))) for item in items]
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return _legacy_tracks(dict(data), source, declarations)
    if suffix == ".csv":
        return _load_canonical_csv(path, source, declarations)
    raise ValueError("unsupported trajectory file: %s" % path)


def _load_canonical_csv(path, source, declarations):
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8-sig")))
    required = {"scene_id", "agent_id", "agent_type", "timestamp", "x", "y", "heading", "valid", "length", "width"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV is not canonical trajectory format; missing %s" % sorted(required - set(rows[0] if rows else [])))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scene_id"], row["agent_id"]), []).append(row)
    tracks = []
    for (scene_id, agent_id), values in grouped.items():
        values.sort(key=lambda row: float(row["timestamp"]))
        timestamps = np.asarray([float(row["timestamp"]) for row in values])
        positions = np.asarray([[float(row["x"]), float(row["y"])] for row in values])
        valid = np.asarray([str(row["valid"]).lower() in {"1", "true", "yes"} for row in values])
        supplied_v = all(row.get("velocity", "") != "" for row in values)
        supplied_a = all(row.get("acceleration", "") != "" for row in values)
        derived_v, derived_a = derive_scalar_motion(positions, valid, timestamps)
        metadata = {
            "dt": float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else None,
            "heading_unit": values[0].get("heading_unit") or declarations.get("heading_unit"),
            "history_frames": int(values[0].get("history_frames") or declarations.get("history_frames")) if (values[0].get("history_frames") or declarations.get("history_frames")) is not None else None,
            "input_fields": list(values[0].keys()),
            "derived_fields": ([] if supplied_v else ["velocities from adjacent valid positions"]) + ([] if supplied_a else ["accelerations from adjacent valid speeds"]),
        }
        tracks.append(Trajectory(
            scene_id=scene_id, agent_id=agent_id, agent_type=values[0]["agent_type"], source=source,
            coordinate_system=values[0].get("coordinate_system") or declarations.get("coordinate_system", ""),
            timestamps=timestamps, positions=positions,
            headings=np.asarray([float(row["heading"]) for row in values]),
            velocities=np.asarray([float(row["velocity"]) for row in values]) if supplied_v else derived_v,
            accelerations=np.asarray([float(row["acceleration"]) for row in values]) if supplied_a else derived_a,
            valid_mask=valid,
            shape={"length": float(values[0]["length"]), "width": float(values[0]["width"])}, metadata=metadata,
        ))
    return tracks


def inspect_file(path):
    path = Path(path)
    result = {"path": str(path), "suffix": path.suffix.lower(), "size_bytes": path.stat().st_size}
    try:
        if path.suffix.lower() in {".pkl", ".pickle"}:
            value = safe_pickle_load(path)
            result["container_type"] = type(value).__name__
            result["fields"] = sorted(value.keys()) if isinstance(value, dict) else []
            if isinstance(value, dict):
                result["array_shapes"] = {key: list(val.shape) for key, val in value.items() if isinstance(val, np.ndarray)}
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            result["container_type"] = type(value).__name__
            result["fields"] = sorted(value.keys()) if isinstance(value, dict) else []
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                result["fields"] = next(csv.reader(handle), [])
        result["readable"] = True
    except Exception as exc:
        result["readable"] = False
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    return result

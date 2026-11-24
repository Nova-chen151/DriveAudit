from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


ALLOWED_AGENT_TYPES = {"vehicle", "pedestrian", "bicycle", "ego"}
ALLOWED_SOURCES = {"generated", "ground_truth"}
ALLOWED_COORDINATES = {"world", "ego", "frenet", "history_last_ego"}
FIXED_METRIC_COORDINATES = {"world", "history_last_ego"}
ALLOWED_HEADING_UNITS = {"rad", "degree"}


def share_fixed_metric_frame(a, b) -> bool:
    return a.coordinate_system == b.coordinate_system and a.coordinate_system in FIXED_METRIC_COORDINATES


class SchemaError(ValueError):
    pass


@dataclass
class Trajectory:
    scene_id: str
    agent_id: str
    agent_type: str
    source: str
    coordinate_system: str
    timestamps: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    valid_mask: np.ndarray
    shape: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self):
        return Trajectory.from_dict(self.to_dict())

    @property
    def dt(self):
        diffs = np.diff(self.timestamps)
        return float(np.median(diffs)) if len(diffs) else None

    def validate(self, expected_dt: Optional[float] = None) -> List[str]:
        errors = []
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            errors.append("scene_id_empty")
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            errors.append("agent_id_empty")
        if self.agent_type not in ALLOWED_AGENT_TYPES:
            errors.append("agent_type_invalid")
        if self.source not in ALLOWED_SOURCES:
            errors.append("source_invalid")
        if self.coordinate_system not in ALLOWED_COORDINATES:
            errors.append("coordinate_system_missing_or_invalid")
        heading_unit = self.metadata.get("heading_unit")
        if heading_unit not in ALLOWED_HEADING_UNITS:
            errors.append("heading_unit_missing_or_invalid")
        arrays = {
            "timestamps": self.timestamps,
            "positions": self.positions,
            "headings": self.headings,
            "velocities": self.velocities,
            "accelerations": self.accelerations,
            "valid_mask": self.valid_mask,
        }
        lengths = {name: len(value) for name, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            errors.append("array_length_mismatch:%s" % lengths)
        if self.positions.ndim != 2 or (self.positions.ndim == 2 and self.positions.shape[1] != 2):
            errors.append("positions_shape_invalid")
        if len(self.timestamps) < 2:
            errors.append("timestamps_insufficient")
        else:
            diffs = np.diff(self.timestamps)
            if np.any(diffs <= 0):
                errors.append("timestamps_not_strictly_increasing")
            if expected_dt is None and self.metadata.get("dt") is None:
                errors.append("dt_missing")
            dt = expected_dt if expected_dt is not None else self.metadata.get("dt")
            if dt is not None and (not np.isfinite(dt) or float(dt) <= 0):
                errors.append("dt_invalid")
        if not np.all(np.isfinite(self.positions)):
            errors.append("positions_non_finite")
        if not np.all(np.isfinite(self.timestamps)):
            errors.append("timestamps_non_finite")
        if self.valid_mask.dtype != bool:
            errors.append("valid_mask_not_boolean")
        if float(self.shape.get("length", 0)) <= 0 or float(self.shape.get("width", 0)) <= 0:
            errors.append("shape_non_positive")
        return errors

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "source": self.source,
            "coordinate_system": self.coordinate_system,
            "timestamps": self.timestamps.tolist(),
            "positions": self.positions.tolist(),
            "headings": self.headings.tolist(),
            "velocities": self.velocities.tolist(),
            "accelerations": self.accelerations.tolist(),
            "valid_mask": self.valid_mask.tolist(),
            "shape": dict(self.shape),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data):
        missing = [key for key in (
            "scene_id", "agent_id", "agent_type", "source", "coordinate_system", "timestamps",
            "positions", "headings", "velocities", "accelerations", "valid_mask", "shape"
        ) if key not in data]
        if missing:
            raise SchemaError("missing fields: %s" % ", ".join(missing))
        if any(not isinstance(value, (bool, np.bool_)) for value in data["valid_mask"]):
            raise SchemaError("valid_mask values must be boolean")
        return cls(
            scene_id=str(data["scene_id"]),
            agent_id=str(data["agent_id"]),
            agent_type=str(data["agent_type"]),
            source=str(data["source"]),
            coordinate_system=str(data["coordinate_system"]),
            timestamps=np.asarray(data["timestamps"], dtype=float),
            positions=np.asarray(data["positions"], dtype=float),
            headings=np.asarray(data["headings"], dtype=float),
            velocities=np.asarray(data["velocities"], dtype=float),
            accelerations=np.asarray(data["accelerations"], dtype=float),
            valid_mask=np.asarray(data["valid_mask"], dtype=bool),
            shape={"length": float(data["shape"]["length"]), "width": float(data["shape"]["width"])},
            metadata=dict(data.get("metadata", {})),
        )


def validate_unique_agent_ids(tracks):
    seen = set()
    duplicates = []
    for track in tracks:
        key = (track.scene_id, track.source, track.agent_id)
        if key in seen:
            duplicates.append(track.agent_id)
        seen.add(key)
    return sorted(set(duplicates))

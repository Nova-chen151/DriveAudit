from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def _read(path):
    return json.loads(Path(path).read_text())


def _rigid(value, name):
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name}: finite 4x4 transform required')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5):
        raise ValueError(f'{name}: invalid rigid calibration; no automatic repair')
    return matrix



from typing import Any
from driveaudit_eval.utils.io import numpy_json_default as _json_default

def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

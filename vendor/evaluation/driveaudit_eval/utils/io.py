"""Shared JSON output: strict JSON, atomic replacement, no fabricated finite scores."""
import csv
import hashlib
import json
import tempfile
import math
import os
from pathlib import Path


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def numpy_json_default(value):
    """Existing research JSON fallback; distinct from strict recursive jsonable."""
    import numpy as np
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_ordered_csv(path, rows):
    """Research tables retain first-seen columns and an empty-table status header."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path, rows, headers=()):
    keys = list(dict.fromkeys([*headers, *(key for row in rows for key in row)]))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else jsonable(value) for key, value in row.items()})




def file_sha256(path):
    """SHA256 of file bytes (including a file symlink's target), streamed."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

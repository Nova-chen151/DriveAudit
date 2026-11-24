from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


from driveaudit_eval.utils.io import file_sha256


def cache_key(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]


def load_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_cache(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as output:
            pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

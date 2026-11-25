"""Internal backend configuration; the formal configuration is tools/configs/research.yaml."""
from copy import deepcopy
from pathlib import Path


def load_config(value):
    if isinstance(value, dict):
        return deepcopy(value)
    import yaml
    return yaml.safe_load(Path(value).read_text(encoding="utf-8"))


def save_snapshot(source, target, retain_intermediates=True):
    """Keep legacy snapshots, or reference existing evidence without copying bytes."""
    import os
    import shutil
    source, target = Path(source).resolve(strict=True), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if retain_intermediates:
        shutil.copy2(source, target)
    elif target.is_symlink() and target.resolve() == source:
        return
    elif target.exists() or target.is_symlink():
        raise FileExistsError('Refusing to replace existing evidence: ' + str(target))
    else:
        target.symlink_to(os.path.relpath(source, target.parent))

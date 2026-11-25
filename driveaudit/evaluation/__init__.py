"""Offline paired policy evaluation for submitted scenes."""
from pathlib import Path
import sys

EVALUATION_ROOT = Path(__file__).resolve().parent


def enable_engine_imports():
    """Expose the scoring engine under its native module name."""
    root = str(EVALUATION_ROOT)
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)

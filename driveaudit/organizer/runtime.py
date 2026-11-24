"""Resolve the native evaluator in this source checkout, including child interpreters."""
import os
import sys
from pathlib import Path

from driveaudit.resources import resource_root


def native_evaluator():
    vendor = str(resource_root() / "vendor/evaluation")
    sys.path.insert(0, vendor)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([vendor, *(p for p in paths if p and p != vendor)])

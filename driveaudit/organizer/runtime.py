"""Resolve the scoring engine for the organizer and child interpreters."""
import os

from driveaudit.evaluation import EVALUATION_ROOT, enable_engine_imports


def native_evaluator():
    enable_engine_imports()
    root = str(EVALUATION_ROOT)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([root, *(p for p in paths if p and p != root)])

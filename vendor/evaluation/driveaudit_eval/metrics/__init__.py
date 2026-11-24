"""Core aggregators with diagnostic and legacy metadata in the central registry."""
from driveaudit_eval.metrics.core import build_core, calculate_tvc, compute_rpdms
from driveaudit_eval.metrics.registry import METRICS, CORE_IDS, DEPRECATED_ALIASES, get_registry

def __getattr__(name):
    if name == 'compute_trs':
        from driveaudit_eval.trajectory.trs import compute_trs
        return compute_trs
    raise AttributeError(name)

"""Shared aggregation with explicit empty-value conventions at callers."""
import math
from statistics import mean as exact_mean
import numpy as np


def finite(values):
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def mean(values, missing=None):
    values = finite(values)
    return float(np.mean(values)) if values else missing


def mean_or_na(values):
    return mean(values, 'not_available')


def weighted_mean(values, weights):
    pairs = [(float(v), float(w)) for v, w in zip(values, weights)
             if isinstance(v, (int, float)) and isinstance(w, (int, float))
             and math.isfinite(float(v)) and math.isfinite(float(w)) and float(w) > 0]
    if not pairs:
        return 'not_available'
    v, w = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
    return float(np.average(v, weights=w))


def average_ranks(values, higher=False):
    valid = sorted((v for v in values if v is not None), reverse=higher)
    return [None if v is None else (valid.index(v) + 1 + len(valid) - valid[::-1].index(v)) / 2
            for v in values]


def metric_number(value):
    if value is None or value in ('', 'not_available', 'unavailable', 'N/A'):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('Invalid numeric metric: %r' % (value,))
    if not math.isfinite(value):
        raise ValueError('Non-finite metric')
    return value

def metric_mean(values):
    values = [metric_number(v) for v in values]
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None

def percentile(values, q):
    return float(np.percentile(values, q)) if values else None

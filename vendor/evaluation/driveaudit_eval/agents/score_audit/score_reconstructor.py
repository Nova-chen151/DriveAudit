from __future__ import annotations

import math


def pdms_open_loop(nc, dac, ttc, comfort, ep):
    if dac is None:
        return None
    return float(nc) * float(dac) * (5.0 * float(ttc) + 2.0 * float(comfort) + 5.0 * float(ep)) / 12.0


def driving_score_open_loop(pdms_mean, rc):
    return None if pdms_mean is None or rc is None else float(pdms_mean) * float(rc)


def reconstruct_frame(row, prefix=""):
    get = lambda key: row.get(prefix + key)
    return pdms_open_loop(get("NC"), get("DAC"), get("TTC_score"), get("comfort_score"), get("EP"))


def formula_matches(original, reconstructed, tolerance=1e-9):
    if original is None or reconstructed is None:
        return original is None and reconstructed is None
    return math.isclose(float(original), float(reconstructed), rel_tol=tolerance, abs_tol=tolerance)

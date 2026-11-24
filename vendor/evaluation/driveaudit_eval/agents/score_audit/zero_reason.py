from __future__ import annotations


ZERO_REASONS = (
    "ZERO_INVALID_PREDICTION", "ZERO_TRAJECTORY_EMPTY", "ZERO_TRAJECTORY_NAN_INF", "ZERO_TRAJECTORY_TOO_SHORT",
    "ZERO_COLLISION_GATE", "ZERO_NC_GATE", "ZERO_DAC_OFFROAD", "ZERO_DAC_MAP_QUERY_FAILURE",
    "ZERO_DAC_COORDINATE_ERROR", "ZERO_ROUTE_COMPLETION", "ZERO_NO_PROGRESS", "ZERO_TTC_GATE",
    "ZERO_COMFORT_GATE", "ZERO_EP_GATE", "ZERO_SCORE_FORMULA_PRODUCT", "ZERO_SCENE_AGGREGATION",
    "ZERO_DENOMINATOR_ERROR", "ZERO_NOT_AVAILABLE_CAST_TO_ZERO", "ZERO_UNKNOWN",
)


def zero_components(row, dac_key="DAC"):
    keys = ("NC", dac_key, "TTC_score", "comfort_score", "EP")
    return [key for key in keys if row.get(key) is not None and float(row[key]) == 0.0]


def classify_frame(row, dac_key="DAC", pdms_key="PDMS_open_loop"):
    if not row.get("prediction_valid", row.get("valid_trajectory", False)):
        return "prediction_valid", "ZERO_INVALID_PREDICTION", ["prediction_valid"]
    if row.get(pdms_key) is None:
        return "not_available", "ZERO_NOT_AVAILABLE_CAST_TO_ZERO", []
    components = zero_components(row, dac_key)
    if float(row[pdms_key]) != 0.0:
        return None, None, components
    if float(row.get("NC", 1.0)) == 0.0:
        return "NC", "ZERO_COLLISION_GATE", components
    if float(row.get(dac_key, 1.0)) == 0.0:
        reason = "ZERO_DAC_MAP_QUERY_FAILURE" if row.get("map_query_status") != "available" else "ZERO_DAC_OFFROAD"
        return dac_key, reason, components
    if float(row.get("TTC_score", 1.0)) == 0.0 and float(row.get("comfort_score", 1.0)) == 0.0 and float(row.get("EP", 1.0)) == 0.0:
        return "weighted_quality_term", "ZERO_SCORE_FORMULA_PRODUCT", components
    return "formula", "ZERO_UNKNOWN", components


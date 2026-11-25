from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.categories import semantic_category
from driveaudit_eval.consistency.shared.common import box_polygon, rotated_iou, wrap_angle


DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
MIN_RECALL, MIN_PRECISION = 0.1, 0.1
VISIBILITY_BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001))


def _flatten(tracks: list[dict[str, Any]], reference: bool) -> list[dict[str, Any]]:
    rows = []
    for track in tracks:
        for row in track["observations"]:
            if reference and not row.get("expected_visible", False):
                continue
            if not reference and row.get("source") != "image_detection":
                continue
            rows.append({**row, "category": semantic_category(track["category"]),
                         "identity": track.get("agent_id", track.get("image_track_id")),
                         "score": 1.0 if reference else float(row.get("score", track.get("mean_confidence", 0.0)))})
    return rows


def _scale_error(a: dict[str, Any], b: dict[str, Any]) -> float:
    # nuScenes scale_iou: 3D IoU after aligning translation and rotation.
    sa = np.asarray([a["width"], a["length"], a.get("height", 1.0)], dtype=float)
    sb = np.asarray([b["width"], b["length"], b.get("height", 1.0)], dtype=float)
    intersection = float(np.prod(np.minimum(sa, sb)))
    return 1.0 - intersection / float(np.prod(sa) + np.prod(sb) - intersection)


def _velocity(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row.get("kalman_velocity", row.get("velocity", [0.0, 0.0])), dtype=float)[:2]


def _errors(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    a, b = reference["box_bev"], prediction["box_bev"]
    return {
        "trans_err": float(np.linalg.norm([a["x"] - b["x"], a["y"] - b["y"]])),
        "scale_err": _scale_error(a, b),
        "orient_err": abs(wrap_angle(float(a["yaw"]) - float(b["yaw"]))),
        "vel_err": float(np.linalg.norm(_velocity(reference) - _velocity(prediction))),
    }


def _accumulate(reference: list[dict[str, Any]], prediction: list[dict[str, Any]], category: str,
                match_threshold: float, iou_threshold: float | None = None) -> dict[str, Any]:
    refs = [row for row in reference if row["category"] == category]
    preds = sorted((row for row in prediction if row["category"] == category), key=lambda row: row["score"], reverse=True)
    if not refs:
        return {"ap": NOT_AVAILABLE, "matches": [], "npos": 0, "predictions": len(preds)}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in refs:
        by_frame[int(row["frame_index"])].append(row)
    taken: set[tuple[int, int]] = set(); tp = []; fp = []; scores = []; matches = []
    for pred in preds:
        frame = int(pred["frame_index"]); best, best_value = None, math.inf
        for index, ref in enumerate(by_frame.get(frame, [])):
            if (frame, index) in taken:
                continue
            if iou_threshold is None:
                value = _errors(ref, pred)["trans_err"]
                valid = value < match_threshold
            else:
                value = _iou3d(ref["box_bev"], pred["box_bev"])
                valid = value >= iou_threshold
                value = -value
            if valid and value < best_value:
                best, best_value = index, value
        scores.append(pred["score"])
        if best is None:
            tp.append(0.0); fp.append(1.0)
        else:
            taken.add((frame, best)); tp.append(1.0); fp.append(0.0)
            ref = by_frame[frame][best]
            matches.append({"reference": ref, "prediction": pred, "score": pred["score"], **_errors(ref, pred)})
    confidence = np.zeros(101); tp_metrics: dict[str, Any] = {}
    if not tp:
        precision = np.zeros(101); recall = np.linspace(0, 1, 101)
    else:
        cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
        raw_recall = cumulative_tp / len(refs); raw_precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        recall = np.linspace(0, 1, 101)
        precision = np.interp(recall, raw_recall, raw_precision, right=0)
        confidence = np.interp(recall, raw_recall, np.asarray(scores), right=0)
        if matches:
            match_scores = np.asarray([row["score"] for row in matches])
            for key in ("trans_err", "scale_err", "orient_err", "vel_err"):
                errors = np.asarray([row[key] for row in matches]); cumulative_mean = np.cumsum(errors) / np.arange(1, len(errors) + 1)
                interpolated = np.interp(confidence[::-1], match_scores[::-1], cumulative_mean[::-1])[::-1]
                first = round(100 * MIN_RECALL) + 1
                nonzero = np.flatnonzero(confidence > 0); last = int(nonzero[-1]) if len(nonzero) else -1
                tp_metrics[key] = float(np.mean(interpolated[first:last + 1])) if last >= first else 1.0
    clipped = np.maximum(precision[round(100 * MIN_RECALL) + 1:] - MIN_PRECISION, 0)
    ap = min(1.0, max(0.0, float(np.mean(clipped) / (1.0 - MIN_PRECISION))))
    return {"ap": ap, "matches": matches, "npos": len(refs), "predictions": len(preds),
            "precision": precision.tolist(), "recall": recall.tolist(), "confidence": confidence.tolist(),
            "tp_metrics": tp_metrics}


def _official_tp_error(accumulated: dict[str, Any], key: str) -> float | str:
    return accumulated.get("tp_metrics", {}).get(key, NOT_AVAILABLE)


def _iou3d(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = Polygon(box_polygon(a)), Polygon(box_polygon(b))
    area = pa.intersection(pb).area
    az, bz = float(a.get("z", 0.0)), float(b.get("z", 0.0))
    ah, bh = float(a.get("height", 1.0)), float(b.get("height", 1.0))
    overlap_z = max(0.0, min(az + ah / 2, bz + bh / 2) - max(az - ah / 2, bz - bh / 2))
    intersection = area * overlap_z
    va, vb = pa.area * ah, pb.area * bh
    return float(intersection / (va + vb - intersection)) if va + vb > intersection else 0.0


def _iou_ap(reference: list[dict[str, Any]], prediction: list[dict[str, Any]], threshold: float) -> tuple[Any, Any, dict[str, Any]]:
    aps, aphs, per_class = [], [], {}
    for category in sorted({row["category"] for row in reference}):
        result = _accumulate(reference, prediction, category, math.inf, threshold)
        if not isinstance(result["ap"], (int, float)):
            continue
        heading = [0.5 * (1.0 + math.cos(row["orient_err"])) for row in result["matches"]]
        aph = result["ap"] * float(np.mean(heading)) if heading else 0.0
        aps.append(result["ap"]); aphs.append(aph); per_class[category] = {"ap": result["ap"], "aph": aph}
    return (float(np.mean(aps)) if aps else NOT_AVAILABLE,
            float(np.mean(aphs)) if aphs else NOT_AVAILABLE, per_class)


def _occupancy_iou(reference: list[dict[str, Any]], prediction: list[dict[str, Any]], category: str | None) -> float | str:
    frames = sorted({int(row["frame_index"]) for row in reference})
    values = []
    for frame in frames:
        ra = [Polygon(box_polygon(row["box_bev"])) for row in reference if int(row["frame_index"]) == frame and (category is None or row["category"] == category)]
        pa = [Polygon(box_polygon(row["box_bev"])) for row in prediction if int(row["frame_index"]) == frame and (category is None or row["category"] == category)]
        if not ra and not pa:
            continue
        from shapely.ops import unary_union
        r, p = unary_union(ra), unary_union(pa)
        union = r.union(p).area
        values.append(r.intersection(p).area / union if union else 1.0)
    return float(np.mean(values)) if values else NOT_AVAILABLE


def compute_bevformer_submission_metrics(reference_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    reference, prediction = _flatten(reference_tracks, True), _flatten(image_tracks, False)
    categories = sorted({row["category"] for row in reference})
    class_results: dict[str, Any] = {}; all_ap = []
    tp_errors: dict[str, list[float]] = defaultdict(list)
    for category in categories:
        thresholds = {}
        for threshold in DISTANCE_THRESHOLDS:
            result = _accumulate(reference, prediction, category, threshold)
            thresholds[str(threshold)] = result["ap"]
            if isinstance(result["ap"], (int, float)): all_ap.append(result["ap"])
            if threshold == 2.0:
                for key in ("trans_err", "scale_err", "orient_err", "vel_err"):
                    value = _official_tp_error(result, key)
                    if isinstance(value, (int, float)): tp_errors[key].append(value)
        class_results[category] = thresholds
    ap05, aph05, per05 = _iou_ap(reference, prediction, 0.5)
    ap07, aph07, per07 = _iou_ap(reference, prediction, 0.7)
    result = {
        "reference_semantics": "submitted_trajectory_as_instruction_reference",
        "implementation_level": "DIRECT_WITH_PROTOCOL_DIFFERENCE",
        "submission_ref_center_distance_AP": class_results,
        "submission_ref_center_distance_mAP": float(np.mean(all_ap)) if all_ap else NOT_AVAILABLE,
        "submission_ref_mATE": float(np.mean(tp_errors["trans_err"])) if tp_errors["trans_err"] else NOT_AVAILABLE,
        "submission_ref_mASE": float(np.mean(tp_errors["scale_err"])) if tp_errors["scale_err"] else NOT_AVAILABLE,
        "submission_ref_mAOE": float(np.mean(tp_errors["orient_err"])) if tp_errors["orient_err"] else NOT_AVAILABLE,
        "submission_ref_mAVE": float(np.mean(tp_errors["vel_err"])) if tp_errors["vel_err"] else NOT_AVAILABLE,
        "submission_ref_mAAE": NOT_AVAILABLE,
        "submission_ref_mAAE_unavailable_reason": "not_available_missing_attributes",
        "submission_ref_NDS_adapted": NOT_AVAILABLE,
        "submission_ref_NDS_adapted_unavailable_reason": "mAAE_missing_no_reweighting_allowed",
        "submission_ref_3d_AP_iou_0_5": ap05, "submission_ref_3d_APH_iou_0_5": aph05,
        "submission_ref_3d_AP_iou_0_7": ap07, "submission_ref_3d_APH_iou_0_7": aph07,
        "submission_ref_3d_AP_by_class": {"0.5": per05, "0.7": per07},
        "submission_ref_vehicle_occupancy_iou": _occupancy_iou(reference, prediction, "vehicle"),
        "submission_ref_pedestrian_occupancy_iou": _occupancy_iou(reference, prediction, "pedestrian"),
        "submission_ref_dynamic_occupancy_iou": _occupancy_iou(reference, prediction, None),
        "occupancy_implementation_level": "CONCEPTUAL_ADAPTATION",
        "sample_count": len(reference),
    }
    visibility = {}
    for low, high in VISIBILITY_BINS:
        refs = [row for row in reference if low <= 1.0 - float(row.get("occlusion_ratio", 0.0)) < high]
        frames = {int(row["frame_index"]) for row in refs}; preds = [row for row in prediction if int(row["frame_index"]) in frames]
        match = [_accumulate(refs, preds, category, 2.0) for category in sorted({row["category"] for row in refs})]
        npos, npred = sum(row["npos"] for row in match), sum(row["predictions"] for row in match)
        matched = sum(len(row["matches"]) for row in match)
        associations: dict[tuple[str, str], int] = defaultdict(int)
        for row in match:
            for item in row["matches"]:
                associations[(str(item["reference"]["identity"]), str(item["prediction"]["identity"]))] += 1
        ref_ids, pred_ids = sorted({key[0] for key in associations}), sorted({key[1] for key in associations})
        idtp = 0
        if ref_ids and pred_ids:
            matrix = np.asarray([[-associations[(a, b)] for b in pred_ids] for a in ref_ids])
            idtp = int(sum(-matrix[i, j] for i, j in zip(*linear_sum_assignment(matrix))))
        visibility[f"{int(low*100)}-{int(min(high,1)*100)}%"] = {
            "reference_count": npos, "recall": matched / npos if npos else NOT_AVAILABLE,
            "precision": matched / npred if npred else NOT_AVAILABLE,
            "center_distance_ap": float(np.mean([row["ap"] for row in match if isinstance(row["ap"], (int, float))])) if match else NOT_AVAILABLE,
            "mATE": float(np.mean([item["trans_err"] for row in match for item in row["matches"]])) if matched else NOT_AVAILABLE,
            "mASE": float(np.mean([item["scale_err"] for row in match for item in row["matches"]])) if matched else NOT_AVAILABLE,
            "mAOE": float(np.mean([item["orient_err"] for row in match for item in row["matches"]])) if matched else NOT_AVAILABLE,
            "mAVE": float(np.mean([item["vel_err"] for row in match for item in row["matches"]])) if matched else NOT_AVAILABLE,
            "IDF1": 2 * idtp / (npos + npred) if npos + npred else NOT_AVAILABLE,
            "agent_missing_ratio": 1.0 - matched / npos if npos else NOT_AVAILABLE,
            "matched_coverage": matched / npos if npos else NOT_AVAILABLE,
        }
    result["visibility_estimation"] = "projection_based_approximation"
    result["visibility_stratified"] = visibility
    return result


def temporal_gain(temporal: dict[str, Any], independent: dict[str, Any], temporal_identity: Any = NOT_AVAILABLE,
                  independent_identity: Any = NOT_AVAILABLE, temporal_fragmentation: Any = NOT_AVAILABLE,
                  independent_fragmentation: Any = NOT_AVAILABLE) -> dict[str, Any]:
    def gain(a: Any, b: Any) -> Any:
        return float(a) - float(b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else NOT_AVAILABLE
    low_t = temporal.get("visibility_stratified", {}).get("0-40%", {}).get("recall", NOT_AVAILABLE)
    low_i = independent.get("visibility_stratified", {}).get("0-40%", {}).get("recall", NOT_AVAILABLE)
    return {"ablation_name": "temporal_ablation_reset_prev_bev",
            "temporal_recall_gain": gain(temporal.get("submission_ref_detection_recall"), independent.get("submission_ref_detection_recall")),
            "temporal_low_visibility_recall_gain": gain(low_t, low_i),
            "temporal_mATE_reduction": gain(independent.get("submission_ref_mATE"), temporal.get("submission_ref_mATE")),
            "temporal_mAOE_reduction": gain(independent.get("submission_ref_mAOE"), temporal.get("submission_ref_mAOE")),
            "temporal_mAVE_reduction": gain(independent.get("submission_ref_mAVE"), temporal.get("submission_ref_mAVE")),
            "temporal_IDF1_gain": gain(temporal_identity, independent_identity),
            "temporal_fragmentation_reduction": gain(independent_fragmentation, temporal_fragmentation)}

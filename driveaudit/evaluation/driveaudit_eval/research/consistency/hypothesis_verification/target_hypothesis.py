from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError


def build(observed: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(observed["observed_points"], float)
    source = observed["source_projection"]; target = observed["target_projection"]
    a, b = np.asarray(source.get("projected_corners", []), float), np.asarray(target.get("projected_corners", []), float)
    count = min(len(a), len(b)); a, b = a[:count], b[:count]
    expected, model, uncertainty, face = None, "none", 1.0, None
    if len(points) and count >= 4:
        hull = cv2.convexHull(a.astype(np.float32), returnPoints=False).reshape(-1)
        matrix, residual = _least_squares_affine(a[hull], b[hull])
        if matrix is not None and residual <= float(config["maximum_visible_face_residual_px"]):
            expected = _apply(matrix, points) - points; model = "visible_face_affine"; uncertainty = residual; face = hull.tolist()
        else:
            expected = _piecewise(a, b, points)
            if expected is not None: model = "piecewise_box_affine"; uncertainty = residual if matrix is not None else 2.0
    if expected is None and len(points):
        expected = _center_scale_rotation(points, source, target); model = "center_scale_rotation"; uncertainty = float(config["fallback_uncertainty"])
    status = "valid" if expected is not None and uncertainty <= float(config["maximum_model_uncertainty"]) else "uncertain_projection_geometry"
    return {"target_expected_flow_per_point": expected.tolist() if expected is not None else [], "target_model_type": model,
            "target_model_uncertainty": float(uncertainty), "target_projection_validity": bool(expected is not None),
            "target_visible_face": face, "target_expected_scale_change": _scale(source,target),
            "target_expected_rotation_change": _rotation(source,target), "target_hypothesis_status": status}


def _least_squares_affine(a: np.ndarray,b: np.ndarray):
    if len(a)<3:return None,float("inf")
    design=np.column_stack([a,np.ones(len(a))])
    if np.linalg.matrix_rank(design)<3:return None,float("inf")
    matrix=np.linalg.lstsq(design,b,rcond=None)[0].T
    if not np.isfinite(matrix).all(): return None,float("inf")
    predicted=_apply(matrix,a)
    return matrix,float(np.median(np.linalg.norm(predicted-b,axis=1)))


def _apply(matrix:np.ndarray,points:np.ndarray)->np.ndarray:
    return np.column_stack([points,np.ones(len(points))])@matrix.T


def _piecewise(a:np.ndarray,b:np.ndarray,points:np.ndarray):
    try: triangulation=Delaunay(a)
    except QhullError:return None
    simplex=triangulation.find_simplex(points); result=np.empty_like(points)
    fallback,_=_least_squares_affine(a,b)
    for index,(point,tri) in enumerate(zip(points,simplex)):
        if tri<0:
            if fallback is None:return None
            result[index]=_apply(fallback,point.reshape(1,2))[0];continue
        transform=triangulation.transform[tri]; bary=transform[:2]@(point-transform[2]); weights=np.append(bary,1-bary.sum())
        result[index]=weights@b[triangulation.simplices[tri]]
    return result-points


def _center_scale_rotation(points:np.ndarray,source:dict[str,Any],target:dict[str,Any])->np.ndarray:
    ca=np.asarray(source["projected_center"],float); cb=np.asarray(target["projected_center"],float)
    scale=_scale(source,target); angle=_rotation(source,target); rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    predicted=(points-ca)@rotation.T*scale+cb;return predicted-points


def _scale(a,b):return float(np.sqrt(max(b.get("projected_area",1e-6),1e-6)/max(a.get("projected_area",1e-6),1e-6)))
def _rotation(a,b):
    aa=cv2.minAreaRect(np.asarray(a["projected_polygon"],np.float32))[2];bb=cv2.minAreaRect(np.asarray(b["projected_polygon"],np.float32))[2]
    return float(np.deg2rad((bb-aa+90)%180-90))

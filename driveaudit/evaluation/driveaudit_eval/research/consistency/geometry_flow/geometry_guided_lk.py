from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.geometry_flow.geometric_warp import transform_points, warp_patch
from driveaudit_eval.research.consistency.geometry_flow.residual_flow import reconstruct


def track(image_a: np.ndarray, image_b: np.ndarray, points: np.ndarray, target_mask: np.ndarray, model: dict[str, Any], parameters: dict[str, Any], guided: bool) -> dict[str, Any]:
    points=np.asarray(points,np.float32).reshape(-1,2)
    if not len(points):return _empty("no_source_points",guided)
    gray_a=cv2.cvtColor(image_a,cv2.COLOR_BGR2GRAY);gray_b=cv2.cvtColor(image_b,cv2.COLOR_BGR2GRAY);p0=points.reshape(-1,1,2)
    predicted=transform_points(points,model["forward_warp"]).astype(np.float32);initial=predicted.reshape(-1,1,2).copy() if guided else None
    options={"winSize":(int(parameters["win_size"]),)*2,"maxLevel":int(parameters["max_level"]),
             "criteria":(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,int(parameters.get("iterations",30)),float(parameters.get("epsilon",.01)))}
    flags=cv2.OPTFLOW_USE_INITIAL_FLOW if guided else 0
    forward,status,_=cv2.calcOpticalFlowPyrLK(gray_a,gray_b,p0,initial,flags=flags,**options)
    if forward is None:return _empty("forward_lk_failed",guided,len(points))
    forward_points=forward.reshape(-1,2);back_initial=points.reshape(-1,1,2).copy() if guided else None
    backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray_b,gray_a,forward,back_initial,flags=flags,**options)
    if backward is None:return _empty("backward_lk_failed",guided,len(points))
    fb=np.linalg.norm(points-backward.reshape(-1,2),axis=1);xy=np.round(forward_points).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<target_mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<target_mask.shape[0]);inside[inside]=target_mask[xy[inside,1],xy[inside,0]]>0
    keep=(status.reshape(-1)>0)&(back_status.reshape(-1)>0)&inside&(fb<=float(parameters["fb_threshold_px"]))
    source,observed,pred=points[keep].astype(float),forward_points[keep].astype(float),predicted[keep].astype(float)
    residual=observed-pred;inlier_ratio=0.;affine=None
    if len(residual)>=3:
        affine,inliers=cv2.estimateAffinePartial2D(pred,observed,method=cv2.RANSAC,ransacReprojThreshold=float(parameters.get("ransac_threshold_px",3)))
        if inliers is not None:
            flags_inlier=inliers.reshape(-1).astype(bool);inlier_ratio=float(flags_inlier.mean());source,observed,pred,residual=source[flags_inlier],observed[flags_inlier],pred[flags_inlier],residual[flags_inlier]
    flows=reconstruct(source,pred,observed)
    return {"status":"tracked" if len(source)>=int(parameters["minimum_valid_points"]) else "insufficient_points","guided":guided,"selected_count":len(points),"valid_count":len(source),
            "valid_point_ratio":len(source)/max(1,len(points)),"source_points":source.tolist(),"predicted_points":pred.tolist(),"image_observed_points":observed.tolist(),
            "expected_flow_per_point":flows["expected_flow"].tolist(),"residual_flow_per_point":flows["residual_flow"].tolist(),"observed_flow_per_point":flows["observed_flow"].tolist(),
            "residual_epe":flows["residual_epe"],"forward_backward_error":float(np.median(fb[keep])) if keep.any() else None,"ransac_inlier_ratio":inlier_ratio,
            "residual_affine":affine.tolist() if affine is not None else None,"geometry_prediction_used_only_as_initial_flow":guided,"gt_prediction_used_as_observation":False}


def patch_errors(image_a:np.ndarray,image_b:np.ndarray,source_mask:np.ndarray,target_mask:np.ndarray,model:dict[str,Any])->dict[str,Any]:
    warped=warp_patch(image_a,model,(image_b.shape[1],image_b.shape[0]));warped_mask=warp_patch((source_mask*255).astype(np.uint8),model,(image_b.shape[1],image_b.shape[0]))>127;valid=warped_mask&(target_mask>0)
    if not valid.any():return {"photometric_warp_error":None,"gradient_warp_error":None,"valid_warped_area":0,"warped_source":warped}
    gray_w=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY);gray_b=cv2.cvtColor(image_b,cv2.COLOR_BGR2GRAY);photo=float(np.mean(np.abs(gray_w[valid].astype(float)-gray_b[valid].astype(float))))
    grad_w=cv2.magnitude(cv2.Sobel(gray_w,cv2.CV_32F,1,0),cv2.Sobel(gray_w,cv2.CV_32F,0,1));grad_b=cv2.magnitude(cv2.Sobel(gray_b,cv2.CV_32F,1,0),cv2.Sobel(gray_b,cv2.CV_32F,0,1))
    return {"photometric_warp_error":photo,"gradient_warp_error":float(np.mean(np.abs(grad_w[valid]-grad_b[valid]))),"valid_warped_area":int(valid.sum()),"warped_source":warped}


def _empty(reason,guided,selected=0):return {"status":"unscorable","reason":reason,"guided":guided,"selected_count":selected,"valid_count":0,"valid_point_ratio":0.,"source_points":[],"predicted_points":[],"image_observed_points":[],"expected_flow_per_point":[],"residual_flow_per_point":[],"observed_flow_per_point":[],"residual_epe":None,"forward_backward_error":None,"ransac_inlier_ratio":0.,"gt_prediction_used_as_observation":False}


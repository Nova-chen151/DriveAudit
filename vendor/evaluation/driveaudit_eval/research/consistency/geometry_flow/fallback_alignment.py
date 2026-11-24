from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def align(warped_source:np.ndarray,target:np.ndarray,target_mask:np.ndarray,config:dict[str,Any])->dict[str,Any]:
    gray_a=cv2.cvtColor(warped_source,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;gray_b=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;ys,xs=np.nonzero(target_mask)
    if not len(xs):return _none("empty_target_core")
    x1,x2,y1,y2=max(0,xs.min()-4),min(gray_a.shape[1],xs.max()+5),max(0,ys.min()-4),min(gray_a.shape[0],ys.max()+5);a,b=gray_a[y1:y2,x1:x2],gray_b[y1:y2,x1:x2]
    if min(a.shape)<8:return _none("patch_too_small")
    warp=np.eye(2,3,dtype=np.float32)
    try:
        score,warp=cv2.findTransformECC(b,a,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,1e-4),None,1)
        residual=np.asarray([warp[0,2],warp[1,2]],float)
        if np.linalg.norm(residual)<=float(config.get("residual_search_radius_px",18)):return {"backend_used":"geometry_guided_ecc","fallback_reason":"lk_insufficient","residual":residual.tolist(),"confidence":float(score)}
    except cv2.error:pass
    shift,response=cv2.phaseCorrelate(a,b)
    if np.linalg.norm(shift)<=float(config.get("residual_search_radius_px",18)) and response>0:return {"backend_used":"geometry_guided_phase_correlation","fallback_reason":"ecc_failed","residual":list(map(float,shift)),"confidence":float(response)}
    sift=cv2.SIFT_create(nfeatures=64);ka,da=sift.detectAndCompute((a*255).astype(np.uint8),None);kb,db=sift.detectAndCompute((b*255).astype(np.uint8),None)
    if da is not None and db is not None:
        matches=cv2.BFMatcher().knnMatch(da,db,k=2);good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
        if len(good)>=3:
            residual=np.median(np.asarray([kb[m.trainIdx].pt for m in good])-np.asarray([ka[m.queryIdx].pt for m in good]),axis=0)
            if np.linalg.norm(residual)<=float(config.get("residual_search_radius_px",18)):return {"backend_used":"geometry_guided_sift","fallback_reason":"ecc_phase_failed","residual":residual.astype(float).tolist(),"confidence":min(1.,len(good)/10)}
    return _none("all_fallbacks_failed")


def _none(reason):return {"backend_used":"unscorable","fallback_reason":reason,"residual":None,"confidence":0.}

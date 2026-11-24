from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def target_count(area: int, config: dict[str, Any]) -> int:
    values=list(map(int,config["target_points_by_area"]));return values[0] if area<1000 else (values[1] if area<5000 else values[2])


def sample(image: np.ndarray, mask: np.ndarray, parameters: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);goal=min(int(config["maximum_points"]),target_count(int(mask.sum()),config));candidates=[]
    shi=cv2.goodFeaturesToTrack(gray,mask=mask*255,maxCorners=goal,qualityLevel=float(parameters["quality_level"]),minDistance=float(parameters["min_distance_px"])) if config.get("shi_tomasi",True) else None
    if shi is not None:candidates.extend((point,"shi_tomasi") for point in shi.reshape(-1,2))
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1));ys,xs=np.nonzero(mask)
    if len(xs):
        stride=max(2,int(np.sqrt(len(xs)/max(goal,1))))
        if config.get("uniform_grid",True):
            for y in range(int(ys.min()),int(ys.max())+1,stride):
                for x in range(int(xs.min()),int(xs.max())+1,stride):
                    if mask[y,x] and gradient[y,x]>=float(config["minimum_gradient"]):candidates.append((np.asarray([x,y],float),"grid"))
        if config.get("gradient_points",True):
            order=np.argsort(gradient[ys,xs])[::-1][:goal]
            candidates.extend((np.asarray([xs[index],ys[index]],float),"gradient") for index in order if gradient[ys[index],xs[index]]>=float(config["minimum_gradient"]))
    selected=[];sources=[];distance=max(1.,float(parameters["min_distance_px"])/2)
    for point,source in candidates:
        x,y=np.round(point).astype(int)
        if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]):continue
        if any(np.linalg.norm(point-existing)<distance for existing in selected):continue
        selected.append(np.asarray(point,float));sources.append(source)
        if len(selected)>=goal:break
    points=np.asarray(selected,float).reshape(-1,2);counts={name:sources.count(name) for name in ("shi_tomasi","grid","gradient")};counts["total_selected"]=len(points)
    return points,sources,counts

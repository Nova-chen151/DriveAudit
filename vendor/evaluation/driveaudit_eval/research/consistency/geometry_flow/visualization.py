from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import image_path


def generate(v1_config: dict[str, Any], rows: list[dict[str, Any]], projections: list[dict[str, Any]], output: str | Path) -> list[str]:
    destination = Path(output) / "visualizations"; destination.mkdir(parents=True, exist_ok=True)
    projection = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in projections}
    ordered = sorted(rows, key=lambda row: (_metric(row.get("guided_residual_epe")), row["camera_name"]))
    picks = (ordered[:3] + ordered[-3:] + [row for row in rows if row["camera_name"] in {"CAM_BACK_LEFT", "CAM_BACK_RIGHT"}][:6])[:12]
    files=[]
    for index,row in enumerate(picks,1):
        image=cv2.imread(str(image_path(v1_config,row["camera_name"],int(row["frame_index"]))))
        if image is None:continue
        item=projection.get((row["agent_id"],row["camera_name"],int(row["frame_index"])))
        if item and item.get("projected_polygon"):
            polygon=np.asarray(item["projected_polygon"],np.int32);cv2.polylines(image,[polygon],True,(0,255,255),2)
            center=np.asarray(item["projected_center"],float);expected=np.asarray(row.get("guided_expected_flow_median",[0,0]),float);observed=np.asarray(row.get("guided_observed_flow_median",[0,0]),float)
            cv2.arrowedLine(image,tuple(np.round(center).astype(int)),tuple(np.round(center+expected).astype(int)),(0,255,0),2)
            cv2.arrowedLine(image,tuple(np.round(center).astype(int)),tuple(np.round(center+observed).astype(int)),(0,0,255),2)
        labels=[f"{row['camera_name']} frame={row['frame_index']} agent={row['agent_id'][:8]}",f"model={row['geometry_model']} area={row['projected_area']:.1f}",f"EPE={row.get('guided_residual_epe')} cosine={row.get('guided_flow_cosine')}",f"state={row.get('guided_support_state')} backend={row.get('backend_used')}"]
        for line,text in enumerate(labels):cv2.putText(image,text,(12,28+25*line),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
        path=destination/f"{index:02d}_{row['camera_name'].lower()}_{row['frame_index']:03d}.jpg";cv2.imwrite(str(path),image);files.append(str(path))
    return files


def _metric(value):return float(value) if isinstance(value,(int,float)) else 1e9

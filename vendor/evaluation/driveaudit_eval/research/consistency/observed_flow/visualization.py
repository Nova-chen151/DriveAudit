from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, image_path


NAMES = ["01_gt_3d_box_projection", "02_visible_face_polygon", "03_foreground_core", "04_gt_instance_mask_not_available",
         "05_occlusion_before_after", "06_expected_flow", "07_observed_flow", "08_flow_residual", "09_high_consistency",
         "10_low_consistency", "11_unscorable", "12_small_object", "13_high_occlusion", "14_cam_back_left",
         "15_cam_back_right", "16_time_alignment", "17_resize_mapping", "18_expected_observed_arrows",
         "19_multiview_same_agent", "20_per_camera_metrics"]


def generate(config: dict[str, Any], metric_rows: list[dict[str, Any]], projections: list[dict[str, Any]], output: str | Path) -> list[str]:
    folder = Path(output) / "visualizations"; folder.mkdir(parents=True, exist_ok=True)
    pmap = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in projections}
    usable = [row for row in metric_rows if row.get("window_scorable")]; low = sorted(usable, key=lambda row: float(row.get("gt_flow_epe", 1e9)), reverse=True)
    high = sorted(usable, key=lambda row: float(row.get("gt_flow_epe", 1e9))); unscorable = [row for row in metric_rows if not row.get("window_scorable")]
    selected = []
    for index, name in enumerate(NAMES):
        if "not_available" in name:
            canvas = np.full((360, 640, 3), 245, np.uint8); cv2.putText(canvas, "GT instance mask: not_available", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
        else:
            pool = unscorable if "unscorable" in name else (low if any(word in name for word in ("low", "small", "occlusion")) else high)
            camera = "CAM_BACK_LEFT" if "back_left" in name else ("CAM_BACK_RIGHT" if "back_right" in name else None)
            row = next((value for value in pool if camera is None or value["camera_name"] == camera), pool[index % len(pool)] if pool else None)
            if row is None: canvas = np.full((360, 640, 3), 245, np.uint8)
            else:
                canvas = cv2.imread(str(image_path(config, row["camera_name"], int(row["frame_index"]))))
                projection = pmap.get((row["agent_id"], row["camera_name"], int(row["frame_index"])), {})
                polygon = np.round(np.asarray(projection.get("projected_polygon", []), float)).astype(np.int32)
                if len(polygon) >= 3: cv2.polylines(canvas, [polygon], True, (0, 255, 255), 3)
                center = np.asarray(projection.get("projected_center", [60, 60]), int); expected = np.asarray(row.get("expected_flow_median", [0, 0]), float); observed = np.asarray(row.get("observed_flow_median", [0, 0]), float)
                cv2.arrowedLine(canvas, tuple(center), tuple(np.round(center + expected * 5).astype(int)), (0, 255, 0), 3)
                cv2.arrowedLine(canvas, tuple(center), tuple(np.round(center + observed * 5).astype(int)), (0, 0, 255), 3)
                labels = [f"scene={config['scene']['scene_id']} agent={row['agent_id'][:8]} camera={row['camera_name']}",
                    f"frame={row['frame_index']} ts={row['timestamp']:.3f} area={row['projected_area']:.1f} source={row['instance_region_source']}",
                    f"expected={expected.round(2)} observed={observed.round(2)} EPE={row.get('gt_flow_epe')} cos={row.get('flow_cosine')} state={row.get('gt_motion_support_state')}"]
                for line, y in zip(labels, (30, 60, 90)): cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 3); cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
        path = folder / f"{name}.jpg"; cv2.imwrite(str(path), canvas); selected.append(str(path))
    return selected


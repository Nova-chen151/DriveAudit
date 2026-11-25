from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import cv2

from driveaudit_eval.consistency.mtpc.common import image_path


def plots(positive: list[dict[str, Any]], negatives: list[dict[str, Any]], output: str | Path) -> list[str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True); paths = []
    fig, axis = plt.subplots(figsize=(7, 4)); axis.hist([row["tcvv_rule_support"] for row in positive], 20, alpha=.6, label="history positive")
    axis.hist([row["tcvv_rule_support"] for row in negatives], 20, alpha=.6, label="synthetic negative"); axis.legend(); axis.set_xlabel("TCVV-R support")
    path = output / "rule_score_distribution.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); paths.append(str(path))
    families = sorted({row.get("negative_family") for row in negatives if row.get("negative_family")})
    means = [np.mean([row["tcvv_rule_support"] for row in negatives if row.get("negative_family") == family]) for family in families]
    fig, axis = plt.subplots(figsize=(12, 5)); axis.bar(range(len(families)), means); axis.set_xticks(range(len(families)), families, rotation=80); axis.set_ylabel("mean TCVV-R support")
    path = output / "negative_family_response.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); paths.append(str(path))
    cameras = sorted({row["camera_name"] for row in positive}); leakage = [np.mean([row["background_flow_leakage_ratio"] for row in positive if row["camera_name"] == camera]) for camera in cameras]
    fig, axis = plt.subplots(figsize=(8, 4)); axis.bar(cameras, leakage); axis.tick_params(axis="x", rotation=45); axis.set_ylabel("background leakage")
    path = output / "background_leakage_by_camera.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); paths.append(str(path))
    return paths


def failure_cases(pair: dict[str, Any], future: list[dict[str, Any]], negatives: list[dict[str, Any]], output: str | Path) -> list[str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("01_high_support_high_consistency", future, lambda r: r.get("tcvv_rule_support", 0) - .1 * r.get("residual_flow_epe", 0)),
        ("02_high_support_low_motion", future, lambda r: r.get("tcvv_rule_support", 0) + .1 * r.get("residual_flow_epe", 0)),
        ("03_high_support_background_leakage", future, lambda r: r.get("tcvv_rule_support", 0) + r.get("background_flow_leakage_ratio", 0)),
        ("04_low_rule_support", future, lambda r: -r.get("tcvv_rule_support", 0)),
        ("05_wrong_scene", [], lambda r: 0),
        ("06_repeated_static_frame", [r for r in negatives if r.get("negative_family") == "repeated_static_frame"], lambda r: r.get("tcvv_rule_support", 0)),
        ("07_wrong_motion_direction", [r for r in negatives if r.get("negative_family") == "motion_direction_reverse"], lambda r: r.get("residual_flow_epe", 0)),
        ("08_future_entrant_false_support", [r for r in future if not r.get("history_anchored")], lambda r: r.get("tcvv_rule_support", 0)),
        ("09_multiview_risk", future, lambda r: r.get("expected_camera_count", 0) * (1-r.get("tcvv_rule_support", 0))),
        ("10_camera_calibration_risk", [r for r in future if r.get("camera_name") == "CAM_BACK_LEFT"], lambda r: r.get("residual_flow_epe", 0)),
        ("11_small_target", future, lambda r: -r.get("projected_area", 0)),
        ("12_high_occlusion", future, lambda r: r.get("occlusion_ratio", 0)),
        ("13_low_texture", future, lambda r: -r.get("foreground_contrast_score", 0)),
        ("14_low_contrast_or_night", future, lambda r: -r.get("inside_gradient", 0)),
        ("15_long_range", future, lambda r: r.get("projected_depth", 0)),
    ]
    paths = []
    for name, rows, key in definitions:
        path = output / f"{name}.png"; figure, axis = plt.subplots(figsize=(8, 4.5))
        if not rows:
            axis.text(.5, .5, "not available: independent wrong-scene pair absent", ha="center", va="center"); axis.axis("off")
        else:
            row = max(rows, key=key); image = cv2.imread(str(image_path(pair, row["camera_name"], int(row["frame_index"]))))
            image = cv2.resize(image, (400, 224)); x1, y1, x2, y2 = map(int, row["source_projected_box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1); center = np.asarray([(x1+x2)//2, (y1+y2)//2])
            for vector, color in ((row.get("expected_relative_flow", [0, 0]), (0, 255, 0)),
                                  (row.get("observed_relative_flow", [0, 0]), (0, 0, 255)),
                                  (row.get("background_flow", [0, 0]) or [0, 0], (255, 0, 0))):
                cv2.arrowedLine(image, tuple(center), tuple(np.round(center + np.asarray(vector)).astype(int)), color, 2)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axis.set_title(
                f"{name}\n{row['camera_name']} agent={row['agent_id'][:8]} score={row.get('tcvv_rule_support',0):.3f} "
                f"resEPE={row.get('residual_flow_epe',0):.3f} leak={row.get('background_flow_leakage_ratio',0):.3f}")
            axis.axis("off")
        figure.tight_layout(); figure.savefig(path); plt.close(figure); paths.append(str(path))
    return paths

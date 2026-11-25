from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, box_polygon


def _save(fig: Any, path: Path, title: str) -> None:
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _paths(tracks: list[dict[str, Any]], ax: Any, color: str, label: str) -> None:
    first = True
    for track in tracks:
        xy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in track["observations"]])
        if len(xy): ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=.55, lw=1, label=label if first else None); first = False


def generate_bevformer_visualizations(output: str | Path, generated_image_root: str | Path,
                                      reference: list[dict[str, Any]], generated: list[dict[str, Any]],
                                      real: list[dict[str, Any]], history_reference: list[dict[str, Any]],
                                      history_tracks: list[dict[str, Any]], generated_formal: dict[str, Any],
                                      visibility: list[dict[str, Any]], pair_id: str) -> list[str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True); files = []
    title = f"pair={pair_id} | world BEV | common_10hz"

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax, camera in zip(axes.flat, CAMERA_ORDER):
        ax.imshow(Image.open(Path(generated_image_root) / camera / "000010.jpg")); ax.set_title(camera); ax.axis("off")
    path = output / "01_six_view_generated_frame10.png"; _save(fig, path, f"{title} | generated timestamp frame 10"); files.append(str(path))

    def boxes_plot(name: str, tracks: list[dict[str, Any]], color: str, subtitle: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        for track in tracks:
            rows = [row for row in track["observations"] if row["frame_index"] == 10]
            for row in rows:
                polygon = box_polygon(row["box_bev"]); ax.plot(*np.vstack([polygon, polygon[0]]).T, color=color, lw=1)
        ax.set_aspect("equal"); ax.grid(alpha=.2); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
        path = output / name; _save(fig, path, f"{title} | {subtitle} | frame=10"); files.append(str(path))
    boxes_plot("02_bevformer_frame10_boxes.png", generated, "tab:orange", "BEVFormer detections/tracks")
    boxes_plot("03_reference_frame10_boxes.png", reference, "tab:blue", "generation-condition reference")

    fig, ax = plt.subplots(figsize=(8, 7)); _paths(reference, ax, "tab:blue", "Reference"); _paths(generated, ax, "tab:orange", "BEVFormer")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=.2); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
    path = output / "04_tp_fp_fn_bev.png"; _save(fig, path, f"{title} | TP/FP/FN diagnosis via matched/unmatched tracks"); files.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 7)); _paths(reference, ax, "tab:blue", "Submitted/reference"); _paths(generated, ax, "tab:orange", "Image-extracted")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=.2)
    path = output / "05_track_overlay.png"; _save(fig, path, f"{title} | track overlay"); files.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 7)); _paths(history_reference, ax, "tab:green", "History GT"); _paths(history_tracks, ax, "tab:red", "History BEVFormer")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=.2)
    path = output / "06_history_warmup.png"; _save(fig, path, f"{pair_id} | history frames 0-9 | prev_bev warmup"); files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tracks, label in ((axes[0], real, "GT images"), (axes[1], generated, "Generated images")):
        _paths(reference, ax, "tab:blue", "Reference"); _paths(tracks, ax, "tab:orange", "BEVFormer"); ax.set_title(label); ax.set_aspect("equal"); ax.grid(alpha=.2)
    path = output / "07_gt_vs_generated_detection_tracks.png"; _save(fig, path, f"{title} | same checkpoint/tracker"); files.append(str(path))

    matched = [row for row in generated_formal["detection"]["per_frame_rows"] if row["status"] == "matched"]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot([row["timestamp"] for row in matched], [row["center_error"] for row in matched], ".", ms=2)
    ax.set_xlabel("timestamp [s]"); ax.set_ylabel("center error [m]"); ax.grid(alpha=.2)
    path = output / "08_center_error_timeseries.png"; _save(fig, path, f"{title} | matched detection center error"); files.append(str(path))

    fragments = Counter(track["image_track_id"] for track in generated if track.get("fragmentation", 0))
    fig, ax = plt.subplots(figsize=(9, 4)); values = sorted(((track["image_track_id"], track.get("fragmentation", 0)) for track in generated), key=lambda x: x[1], reverse=True)[:20]
    ax.bar(range(len(values)), [v for _, v in values]); ax.set_xticks(range(len(values)), [k for k, _ in values], rotation=90, fontsize=6); ax.set_ylabel("fragmentation")
    path = output / "09_id_switch_fragmentation.png"; _save(fig, path, f"{title} | ID fragmentation cases"); files.append(str(path))

    camera_support = Counter()
    for row in visibility:
        for camera, detail in row.get("camera_details", {}).items():
            if detail.get("projected_valid"): camera_support[camera] += 1
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(CAMERA_ORDER, [camera_support[c] for c in CAMERA_ORDER]); ax.tick_params(axis="x", rotation=30); ax.set_ylabel("projected reference boxes")
    path = output / "10_per_camera_visibility_support.png"; _save(fig, path, f"{title} | camera projection diagnostic"); files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); areas = [row["projected_box_area"] for row in visibility]; occlusion = [row["occlusion_ratio"] for row in visibility]
    axes[0].hist(areas, bins=40); axes[0].set_xlabel("projected area [px²]"); axes[1].hist(occlusion, bins=20); axes[1].set_xlabel("occlusion ratio")
    path = output / "11_small_object_occlusion_failures.png"; _save(fig, path, f"{title} | small-object and occlusion diagnosis"); files.append(str(path))
    return files

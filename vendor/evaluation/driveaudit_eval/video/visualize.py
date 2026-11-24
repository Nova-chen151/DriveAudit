from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

from driveaudit_eval.video.utils import ensure_dir


def save_error_triplet(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    warped: np.ndarray,
    error: np.ndarray,
    out: Path,
) -> None:
    error = np.asarray(error, dtype=np.float32)
    if error.ndim == 3:
        error = error.mean(axis=2)
    if error.size and np.max(error) > 0:
        heat = np.clip(error / np.percentile(error, 99), 0, 1)
    else:
        heat = np.zeros_like(error)
    heat_rgb = np.stack(
        [heat * 255, np.zeros_like(heat), (1 - heat) * 255], axis=2
    ).astype(np.uint8)
    panels = [
        Image.fromarray(np.asarray(image_t, dtype=np.uint8)),
        Image.fromarray(np.asarray(image_t1, dtype=np.uint8)),
        Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8)),
        Image.fromarray(heat_rgb),
    ]
    width, height = panels[0].size
    label_h = 22
    canvas = Image.new("RGB", (width * len(panels), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["G_t", "G_t1", "warped_G_t", "error"]
    for idx, panel in enumerate(panels):
        if panel.size != (width, height):
            panel = panel.resize((width, height), Image.BILINEAR)
        x = idx * width
        canvas.paste(panel, (x, label_h))
        draw.text((x + 5, 4), labels[idx], fill=(0, 0, 0))
    ensure_dir(out.parent)
    canvas.save(out)


def save_match_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    inliers: Optional[np.ndarray],
    out: Path,
    max_points: int = 80,
) -> None:
    img_a = Image.fromarray(np.asarray(image_a, dtype=np.uint8)).convert("RGB")
    img_b = Image.fromarray(np.asarray(image_b, dtype=np.uint8)).convert("RGB")
    if img_a.height != img_b.height:
        img_b = img_b.resize((img_b.width, img_a.height), Image.BILINEAR)
    canvas = Image.new("RGB", (img_a.width + img_b.width, img_a.height), "white")
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width, 0))
    draw = ImageDraw.Draw(canvas)
    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)
    if inliers is None:
        inliers = np.ones((len(points_a),), dtype=bool)
    step = max(1, len(points_a) // max_points) if len(points_a) else 1
    for idx in range(0, len(points_a), step):
        p0 = points_a[idx]
        p1 = points_b[idx] + np.array([img_a.width, 0])
        color = (0, 220, 0) if bool(inliers[idx]) else (230, 40, 40)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=1)
        r = 2
        draw.ellipse((p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r), fill=color)
        draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), fill=color)
    ensure_dir(out.parent)
    canvas.save(out)


def save_curve_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

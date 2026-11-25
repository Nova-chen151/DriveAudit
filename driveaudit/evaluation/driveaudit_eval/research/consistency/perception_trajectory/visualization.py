from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from driveaudit_eval.consistency.shared.common import CAMERA_ORDER


def render_six_camera(image_root: str | Path, frame_index: int, output: str | Path) -> None:
    image_root, output = Path(image_root), Path(output)
    images = [Image.open(image_root / camera / f"{frame_index:06d}.jpg").convert("RGB") for camera in CAMERA_ORDER]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3, height * 2 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (camera, image) in enumerate(zip(CAMERA_ORDER, images)):
        x, y = index % 3 * width, index // 3 * height + 28
        canvas.paste(image, (x, y)); draw.text((x + 5, y - 22), f"{camera} frame={frame_index}", fill="black")
        image.close()
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)


def render_bev_tracks(tracks: list[dict[str, Any]], output: str | Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    for track in tracks:
        rows = [row for row in track["observations"] if row.get("source") == "image_detection"]
        if not rows: continue
        axis.plot([row["box_bev"]["x"] for row in rows], [row["box_bev"]["y"] for row in rows],
                  linewidth=1, label=track["image_track_id"] if len(tracks) < 15 else None)
    axis.set(xlabel="world x (m)", ylabel="world y (m)", title=title); axis.axis("equal"); axis.grid(True, alpha=.3)
    if len(tracks) < 15: axis.legend(fontsize=7)
    figure.tight_layout(); Path(output).parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=140); plt.close(figure)


def render_history(samples: list[dict[str, Any]], output: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples: grouped.setdefault(sample["agent_id"], []).append(sample)
    for values in grouped.values():
        axis.plot([row["box_bev"]["x"] for row in values], [row["box_bev"]["y"] for row in values], alpha=.5)
    axis.set(xlabel="world x (m)", ylabel="world y (m)", title="History known boxes: 10-frame ID warmup"); axis.axis("equal"); axis.grid(True, alpha=.3)
    figure.tight_layout(); Path(output).parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=140); plt.close(figure)


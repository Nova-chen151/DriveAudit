"""Validate all published six-camera frames using the shared GenBench contract."""

import json
import sys
from pathlib import Path


def main(root, model, marker, expected):
    from genbench_gt2.dataset import load_generation_set
    from genbench_gt2.output import validate_completed_scene
    from PIL import Image

    scenes = load_generation_set(root, "participant_joint")
    if expected != [scene.scene_id for scene in scenes]:
        raise ValueError("Unexpected generation set")
    rows = []
    for scene in scenes:
        folder = root / "predictions" / model / scene.dataset / "scenes" / scene.scene_id
        result = validate_completed_scene(folder, model, scene)
        if not result.valid:
            raise ValueError(f"{folder}: {result}")
        images = sorted((folder / "future/rgb").glob("*/*.jpg"))
        if len(images) != 480:
            raise ValueError(f"Expected 480 RGB files at {folder}")
        for filename in images:
            with Image.open(filename) as im:
                im.load()
                if im.size != (400, 224) or im.mode != "RGB":
                    raise ValueError(f"Invalid image: {filename}")
        rows.append(dict(scene=scene.scene_id, images=len(images)))
    marker.write_text(json.dumps(dict(status="PASS", model=model, scenes=rows), indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:])

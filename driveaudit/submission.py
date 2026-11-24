"""Export a participant-format submission from CAT-K-conditioned renderer outputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .generation import MODELS, VENDOR
from .generation.__main__ import scene_ids
from .generation.staging import relative_scene

CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def package(inputs, conditions, renderer, cases, metadata, destination):
    if sys.flags.optimize:
        raise RuntimeError("Run the public validation script with Python assertions enabled")
    scene_ids(cases)
    if not inputs.joinpath("dataset.json").is_file():
        raise ValueError("Point --inputs at the extracted participant directory")
    dest = destination.absolute()
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    validator = VENDOR / "catk/participant_tools/validate_submission.py"
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        bundle = Path(tmp) / "submission"
        bundle.mkdir()
        shutil.copyfile(metadata, bundle / "submission.json")
        for case in cases:
            output = bundle / "scenes" / case / "future"
            output.mkdir(parents=True)
            stage = conditions / relative_scene(case) / "future"
            model = conditions / "predictions" / renderer / "nuscenes/scenes" / case
            status = json.loads((model / "status.json").read_bytes())
            if (status["model"], status["scene_id"], status["state"], status["observed_image_count"]) != (renderer, case, "complete", 480):
                raise ValueError(f"Incomplete or mismatched generation: {renderer}/{case}")
            shutil.copyfile(stage / "future_trajectories.csv", output / "future_trajectories.csv")
            for camera in CAMERAS:
                target = output / "rgb" / camera
                target.mkdir(parents=True)
                for frame in range(10, 90):
                    name = f"{frame:06d}.jpg"
                    shutil.copyfile(model / "future/rgb" / camera / name, target / name)
            env = dict(os.environ)
            env.pop("PYTHONOPTIMIZE", None)
            subprocess.run([sys.executable, str(validator), "--inputs", str(inputs), "--submission",
                            str(bundle), "--scene", case], check=True, env=env)
        os.rename(bundle, dest)
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "conditions", "metadata", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--renderer", choices=MODELS, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    args = p.parse_args()
    print(package(args.inputs, args.conditions, args.renderer, args.cases, args.metadata, args.output))


if __name__ == "__main__":
    main()

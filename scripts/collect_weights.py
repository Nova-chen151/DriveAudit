#!/usr/bin/env python3
"""Collect the released model assets needed by the generation and evaluation profiles in one weights directory."""
import argparse
import errno
import filecmp
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaudit.agents.registry import POLICIES
from driveaudit.generation.installation import WEIGHTS
from driveaudit.generation.native import CATK_ASSETS

BASE_CLIP = "base-pretrained/pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K"
IGNORED = {".git", ".cache", "__pycache__", "examples", "assets", "logs"}


def selected(args):
    rendering = {Path(v) for fields in WEIGHTS.values() for v in fields.values()}
    rendering.update([Path(BASE_CLIP), Path(WEIGHTS["basedreamer"]["render_checkpoint"]).parent/"hydra",
                      Path(WEIGHTS["horizondrive"]["render_checkpoint"]).with_name("horizondrive-vae.pkl")])
    for path in sorted(rendering):
        yield args.renderer_assets/path, Path("renderers")/path
    for path in CATK_ASSETS:
        yield args.catk_assets/path, Path("catk")/path
    for path in ["ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt", "pretrained/CogVideoX-2b/vae", "pretrained/t5-v1_1-xxl"]:
        yield args.v2_assets/path, Path("magicdrivev2")/path
    for model, _, files in POLICIES.values():
        for name in files:
            path = Path(model)/name
            yield args.agent_assets/path, Path("agents")/path
    for name in ("UniAD/data_temp/nus_vis_pose_info.pkl",):
        yield args.agent_assets/name, Path("agents")/name
    for path in ("models/video/loftr/loftr_outdoor.ckpt", "home/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
                 "home/torch/hub/pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth", "home/torch/hub/clip/RN50.pt"):
        yield args.metric_cache/path, Path("evaluation")/path


def materialize(source, dest, root, inventory):
    if source.is_dir():
        for child in sorted(source.iterdir()):
            if child.name not in IGNORED and not child.name.startswith(".") and child.suffix not in (".log", ".pyc"):
                materialize(child, dest / child.name, root, inventory)
        return
    original = source.resolve(strict=True)
    if not original.is_file():
        raise ValueError(f"Expected regular model asset at {original}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            os.link(original, dest)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP, errno.EMLINK):
                raise
            shutil.copyfile(original, dest)
    if not dest.samefile(original) and not filecmp.cmp(original, dest, shallow=False):
        raise ValueError(f"Model asset differs from original: {dest}")
    inventory[str(dest.relative_to(root))] = dest.stat().st_size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("renderer-assets", "agent-assets", "v2-assets", "catk-assets", "metric-cache", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    args = p.parse_args()
    dest = args.output.absolute()
    # A fresh clone has a tracked empty marker in weights/.
    marker = dest / ".gitkeep"
    scaffold = (dest.is_dir() and not dest.is_symlink()
                and set(dest.iterdir()) == {marker} and marker.is_file()
                and not marker.is_symlink() and marker.stat().st_size == 0)
    if (dest.exists() or dest.is_symlink()) and not scaffold:
        raise FileExistsError(f"Use a fresh directory: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".weights-") as tmp:
        stage = Path(tmp)/"weights"
        inventory = {}
        for source, relative in selected(args):
            materialize(source, stage/relative, stage, inventory)
        (stage/"inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True)+"\n")
        if scaffold:
            (stage / ".gitkeep").touch()
            marker.unlink()
        try:
            os.rename(stage, dest)
        finally:
            if scaffold and not marker.exists():
                marker.touch(exist_ok=True)
    print(f"{dest}: {len(inventory)} verified files, {sum(inventory.values())} logical bytes")


if __name__ == "__main__":
    main()

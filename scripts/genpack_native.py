#!/usr/bin/env python3
"""Build the offline generation source distribution from installed native projects."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

# Explicit inference components. Dataset payloads, example scenes, caches, checkpoints and build outputs stay external.
WORLD = "gen-benchmark/DriveAudit-video/WorldDreamer/"
ROOTS = [
    "catk/third_party/catk-official/src",
    "MagicDrive-V2/rd/magicdrivedit", "MagicDrive-V2/rd/advg", "MagicDrive-V2/rd/configs",
    "MagicDrive-V2/rd/scripts/adv_inference_magicdrive.py",
    WORLD + "BaseDreamer/projects", WORLD + "BaseDreamer/configs", WORLD + "BaseDreamer/dataset",
    WORLD + "DreamForge/dreamforge", WORLD + "DreamForge/configs",
    WORLD + ".build/dit/dreamforgedit",
    "gen-benchmark/MagicDrive-video/configs", "gen-benchmark/MagicDrive-video/magicdrive",
    "gen-benchmark/MagicDrive-video/third_party/diffusers/src",
    "gen-benchmark/MagicDrive-bevfusion/mmdet3d", "gen-benchmark/MagicDrive-bevfusion/setup.py",
    "gen-benchmark/OpenDWM-main/src/dwm", "gen-benchmark/OpenDWM-main/examples/ctsd_unimlvg_6views_video_generation.json",
    "gen-benchmark/HorizonDrive-main/horizondrive", "gen-benchmark/HorizonDrive-main/videox_fun",
    "gen-benchmark/panacea-main/sgm", "gen-benchmark/panacea-main/configs/inference_nuscenes.yaml",
]
LICENSE_ROOTS = [
    "catk/third_party/catk-official", "MagicDrive-V2", "MagicDrive-V2/rd",
    WORLD+"BaseDreamer", WORLD+"DreamForge", WORLD+".build/dit/dreamforgedit",
    "gen-benchmark/MagicDrive-main", "gen-benchmark/MagicDrive-video",
    "gen-benchmark/MagicDrive-video/third_party/diffusers", "gen-benchmark/MagicDrive-bevfusion",
    "gen-benchmark/OpenDWM-main", "gen-benchmark/HorizonDrive-main", "gen-benchmark/panacea-main",
]
EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".h", ".cuh", ".cc", ".c", ".cu", ".cpp", ".hpp", ".smart"}
SKIP = {"__pycache__", ".git", "build", "tests", "test"}


# These editable packages live outside site-packages in the tested environments.
DEPENDENCIES = [
    "DriveAudit-video/WorldDreamer/.build/base/bevfusion/mmdet3d",
    "DriveAudit-video/WorldDreamer/.build/base/diffusers/src",
    "DriveAudit-video/WorldDreamer/.build/base/xformers/xformers",
    "DriveAudit-video/WorldDreamer/.build/base/nuplan-devkit/nuplan",
    "DriveAudit-video/WorldDreamer/.build/forge/diffusers/src",
    "DriveAudit-video/WorldDreamer/.build/forge/mmdetection3d/mmdet3d",
    "MagicDrive-xformers/xformers",
    "panacea-main/third_party/mmdetection3d/mmdet3d",
]


def source_files(source, dependencies):
    result = {}
    requests = [(source, name, "") for name in ROOTS]
    requests.extend((dependencies, name, "gen-benchmark/") for name in DEPENDENCIES)
    notices = {(source / name, source, "") for name in LICENSE_ROOTS}
    for base_root, name, prefix in requests:
        base = base_root / name
        if not base.exists():
            raise FileNotFoundError(base)
        if base_root == dependencies:
            notices.add((base.parent, base_root, prefix))
        for path in ([base] if base.is_file() else base.rglob("*")):
            if (path.is_file() and not path.is_symlink() and path.suffix in EXTENSIONS
                    and not SKIP.intersection(path.relative_to(base.parent).parts)):
                result[prefix + path.relative_to(base_root).as_posix()] = path
                notices.add((path.parent, base_root, prefix))
    for directory, base_root, prefix in notices:
        for notice in directory.iterdir():
            if notice.is_file() and notice.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                result[prefix + notice.relative_to(base_root).as_posix()] = notice
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", required=True, type=Path)
    parser.add_argument("--dependencies-root", required=True, type=Path, help="Installed gen-benchmark package and compiled-extension roots")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_code.resolve()
    content = {}
    with args.output.open("xb") as dest, gzip.GzipFile(fileobj=dest, filename="", mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as archive:
            for name, path in sorted(source_files(root, args.dependencies_root.resolve()).items()):
                data = path.read_bytes()
                if name.endswith("ctsd_unimlvg_6views_video_generation.json"):
                    # The inference adapter supplies both weight paths and constructs its dataset directly.
                    doc = json.loads(data)
                    doc = {key: doc[key] for key in ("device", "generator_seed", "pipeline")}
                    for key in ("model_checkpoint_path", "pretrained_model_name_or_path"):
                        doc["pipeline"].pop(key)
                    data = (json.dumps(doc, indent=2) + "\n").encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
                content[name] = hashlib.sha256(data).hexdigest()
    manifest = {"archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "file_count": len(content), "files": content}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

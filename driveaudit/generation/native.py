"""Install the bundled native inference code, precompiled extensions and external model assets offline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from driveaudit.resources import resource_root
from . import GENERATION
BUNDLES = GENERATION / "bundles"


def adapter_sources():
    """Map native launch paths to their single maintained source."""
    layout = json.loads((BUNDLES / "adapters.json").read_text())
    for name, source in layout.items():
        path = Path(source)
        if path.parts[0] == "driveaudit":
            yield Path(name), GENERATION.parent / path.relative_to("driveaudit")
        else:
            yield Path(name), resource_root() / path
# Upstream import and launch paths are reconstructed only in the native runtime.
WORLD = "gen-benchmark/DriveAudit-video/WorldDreamer/"
SOURCE_LAYOUT = {
    "catk/third_party/catk-official": "trajectory/CAT-K",
    WORLD + "BaseDreamer": "image/BaseDreamer",
    WORLD + ".build/base": "image/BaseDreamer/third_party",
    WORLD + "DreamForge": "image/DreamForge",
    WORLD + ".build/forge": "image/DreamForge/third_party",
    WORLD + ".build/dit/dreamforgedit": "image/DreamForge/third_party/dreamforgedit",
    "gen-benchmark/HorizonDrive-main": "image/HorizonDrive",
    "gen-benchmark/MagicDrive-video": "image/MagicDrive",
    "gen-benchmark/MagicDrive-main/LICENSE": "image/MagicDrive/LICENSE.upstream-main",
    "gen-benchmark/MagicDrive-bevfusion": "image/MagicDrive/third_party/bevfusion",
    "gen-benchmark/MagicDrive-xformers": "image/MagicDrive/third_party/xformers",
    "MagicDrive-V2/rd": "image/MagicDrive-V2",
    "MagicDrive-V2/LICENSE": "image/MagicDrive-V2/LICENSE.upstream-project",
    "gen-benchmark/OpenDWM-main": "image/OpenDWM",
    "gen-benchmark/panacea-main": "image/Panacea",
}


def generation_source(runtime_name):
    """Locate a pinned upstream file in the model-centric source checkout."""
    path = Path(runtime_name)
    for prefix in (path, *path.parents):
        folder = SOURCE_LAYOUT.get(prefix.as_posix())
        if folder is not None:
            return resource_root() / "models/generation" / folder / path.relative_to(prefix)
    raise ValueError(f"Unmapped native source: {runtime_name}")


ABI_ROOTS = (
    "DriveAudit-video/WorldDreamer/.build/base/bevfusion/mmdet3d",
    "DriveAudit-video/WorldDreamer/.build/base/xformers/xformers",
    "DriveAudit-video/WorldDreamer/.build/forge/mmdetection3d/mmdet3d",
    "DriveAudit-video/WorldDreamer/.build/dit/dreamforgedit/mmdet_plugin",
    "MagicDrive-bevfusion/mmdet3d",
    "MagicDrive-xformers/xformers",
    "panacea-main/third_party/mmdetection3d/mmdet3d",
)
CATK_ASSETS = (
    "checkpoints/clsft_E9.ckpt", "src/smart/tokens/cluster_frame_5_2048_remove_duplicate.pkl",
    "src/smart/tokens/agent_vocab_555_s2.pkl", "src/smart/tokens/map_traj_token5.pkl",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_assets(dest, args):
    links = {
        Path("catk/third_party/catk-official") / rel: (args.catk_assets / rel).resolve(strict=True)
        for rel in CATK_ASSETS
    }
    links.update({
        Path("MagicDrive-V2/rd/ckpts"): args.v2_checkpoints.resolve(strict=True),
        Path("MagicDrive-V2/rd/pretrained"): args.v2_pretrained.resolve(strict=True),
        Path("gen-benchmark/DriveAudit-video/WorldDreamer/BaseDreamer/pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K"):
            (args.weights_root / "base-pretrained/pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K").resolve(strict=True)
    })
    for relative, source in links.items():
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return {key.as_posix(): str(value) for key, value in links.items()}


def install(args):
    dest = args.destination.resolve()
    if dest.exists():
        raise FileExistsError(f"Existing installation {dest}. Use --verify or a new destination.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BUNDLES / "native-sources.tar.json").read_text())
    with tempfile.TemporaryDirectory(prefix=".native-install-", dir=dest.parent) as temp:
        staging = Path(temp) / "native"
        staging.mkdir()
        hashes = {}
        for name in manifest["files"]:
            src = generation_source(name)
            dst = staging / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            hashes[name] = sha(dst)
        for relative, src in adapter_sources():
            dst = staging / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            hashes[relative.as_posix()] = sha(dst)
        binaries = {}
        links = {}
        if not args.source_only:
            for name in ABI_ROOTS:
                compiled = args.extensions_root / name
                if not compiled.is_dir():
                    raise FileNotFoundError(compiled)
                for src in sorted(compiled.rglob("*.so")):
                    relative = Path("gen-benchmark") / src.relative_to(args.extensions_root)
                    dst = staging / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    binaries[relative.as_posix()] = sha(dst)
            if not binaries:
                raise ValueError("Missing native ABI-specific extensions in the supplied installation")
            links = link_assets(staging, args)
        receipt = dict(installation="source_only" if args.source_only else "gpu_runtime",
                       source_bundle_sha256=manifest["archive_sha256"], source_files=hashes,
                       compiled_extensions=binaries, external_assets=links)
        (staging / "NATIVE_INSTALL.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.rename(staging, dest)
    return receipt


def verify(destination):
    receipt = json.loads((destination / "NATIVE_INSTALL.json").read_text())
    for group in ("source_files", "compiled_extensions"):
        for name, expected in receipt[group].items():
            path = destination / name
            if path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError(f"Changed native installation: {path}")
    for name, original in receipt["external_assets"].items():
        if (destination / name).resolve(strict=True) != Path(original):
            raise ValueError(f"Changed native resource link: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--source-only", action="store_true", help="Materialize the complete source tree for inspection")
    options = ("weights-root", "catk-assets", "v2-checkpoints", "v2-pretrained", "extensions-root")
    for option in options:
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--weight-store", type=Path, help="Unified weights directory produced by collect_weights.py")
    args = parser.parse_args()
    if args.weight_store:
        store = args.weight_store
        args.weights_root = store / "renderers"
        args.catk_assets = store / "catk"
        args.v2_checkpoints = store / "magicdrivev2/ckpts"
        args.v2_pretrained = store / "magicdrivev2/pretrained"
    if args.verify:
        result = verify(args.destination)
    else:
        missing = [name for name in options if getattr(args, name.replace("-", "_")) is None]
        if missing and not args.source_only:
            parser.error("Installation requires " + ", ".join("--" + name for name in missing))
        result = install(args)
    print(json.dumps(dict(destination=str(args.destination.resolve()), sources=len(result["source_files"]),
                          compiled_extensions=len(result["compiled_extensions"]), assets=len(result["external_assets"]),
                          bundle_sha256=result["source_bundle_sha256"]), indent=2))


if __name__ == "__main__":
    main()

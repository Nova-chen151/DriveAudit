"""DriveAudit participant traffic and rendering command line."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from . import MODELS, VENDOR
from .renderer import render
from .installation import model_settings
from .staging import read, stage


def scene_ids(values):
    if (not isinstance(values, list) or not values or not all(isinstance(v, str) and
            re.fullmatch(r"case_[0-9]{6}", v) for v in values) or len(values) != len(set(values))):
        raise ValueError("scenes must be a nonempty list of distinct participant case IDs")
    return values


def load_config(path):
    config = read(path)
    scene_ids(config["scenes"])
    paths = ["archive", "work_dir", "engine_code_root", "models_json", "catk_python",
             "v2_python", "catk_run", "catk_official"]
    for key in paths:
        if key in config:
            value = Path(config[key]).expanduser()
            config[key] = str((path.parent / value).resolve())
    for model, value in config.get("cuda_homes", {}).items():
        config["cuda_homes"][model] = str((path.parent / Path(value).expanduser()).resolve())
    if "models" in config:
        for settings in config["models"].values():
            for key, value in settings.items():
                if key == "python":
                    value["renderer"] = str((path.parent / Path(value["renderer"]).expanduser()).resolve())
                elif isinstance(value, str):
                    settings[key] = str((path.parent / Path(value).expanduser()).resolve())
    for key in ("archive", "work_dir"):
        if not config.get(key):
            raise ValueError(f"Missing configuration field {key}")
    return config


def traffic(cfg):
    work = Path(cfg["work_dir"])
    official = cfg.get("catk_official") or str(Path(cfg["engine_code_root"]) / "catk/third_party/catk-official")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("GENBENCH_")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="4",
               OMP_NUM_THREADS="4", DRIVEAUDIT_CATK_OFFICIAL=official)
    subprocess.run([cfg["catk_python"], "-u", str(VENDOR / "catk/run_inference.py"), "--archive",
                    cfg["archive"], "--output", str(work / "traffic"), "--scene", *cfg["scenes"],
                    "--device", cfg.get("traffic_device", "cuda:0"), "--seed", str(cfg.get("traffic_seed", 20260914))],
                   env=env, cwd=VENDOR / "catk", check=True)
    return work / "traffic/COMPLETED.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write a run configuration for the native GPU engines")
    for key in ("engine-code-root", "conda-envs", "archive", "work-dir", "output"):
        init.add_argument("--" + key, required=True, type=Path)
    init.add_argument("--weights-root", type=Path, help="Build an editable inline model registry from installed checkpoints")
    init.add_argument("--scene", nargs="+", required=True)
    init.add_argument("--cuda-prefix", type=Path, help="Directory containing cuda-11.3, cuda-11.7, cuda-11.8 and cuda-12.1")
    for name in ("traffic", "stage", "render"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "render":
            command.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        code = args.engine_code_root.resolve()
        envs = args.conda_envs.resolve()
        config = dict(archive=str(args.archive.resolve()), work_dir=str(args.work_dir.resolve()),
                      scenes=scene_ids(args.scene), engine_code_root=str(code), models_json=str(code.parent / "configs/models.json"),
                      catk_python=str(envs / "catk/bin/python"), v2_python=str(envs / "mddit/bin/python"))
        if args.weights_root:
            config.pop("models_json")
            config["models"] = model_settings(envs, args.weights_root.resolve())
        if args.cuda_prefix:
            versions = dict(basedreamer="11.3", dreamforge="11.8", magicdrive="11.3", opendwm="12.1",
                            panacea="11.7", magicdrivev2="12.1")
            config["cuda_homes"] = {model: str(args.cuda_prefix.resolve() / ("cuda-" + version)) for model, version in versions.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        result = args.output
    else:
        cfg = load_config(args.config)
        if args.command == "traffic":
            result = traffic(cfg)
        elif args.command == "stage":
            result = stage(cfg)
        else:
            result = render(cfg, args.model)
    print(result)


if __name__ == "__main__":
    main()

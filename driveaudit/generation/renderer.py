"""Launch the native GPU renderers with the participant-specific adapters."""

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack

from . import VENDOR
from .staging import read, verify_stage, write

WORLD_DREAMER = Path("DriveAudit-video/WorldDreamer")
PROJECTS = {
    "basedreamer": WORLD_DREAMER / "BaseDreamer",
    "dreamforge": WORLD_DREAMER / "DreamForge",
    "horizondrive": Path("HorizonDrive-main"),
    "magicdrive": Path("MagicDrive-main"),
    "opendwm": Path("OpenDWM-main"),
    "panacea": Path("panacea-main"),
}
INDEPENDENT = {"basedreamer", "dreamforge", "magicdrive"}


def overlay(native, patches, destination):
    """Expand patched branches, link untouched native libraries and large assets."""
    destination.mkdir()
    native_files = {p.name: p for p in native.iterdir()} if native.is_dir() else {}
    patch_files = {p.name: p for p in patches.iterdir()}
    for name in sorted(native_files.keys() | patch_files.keys()):
        if name.startswith(".git") or name == "__pycache__":
            continue
        dst = destination / name
        if name not in patch_files:
            dst.symlink_to(native_files[name])
        elif patch_files[name].is_dir():
            overlay(native / name, patch_files[name], dst)
        else:
            shutil.copy2(patch_files[name], dst)


def runtime_for(engine, model, logs):
    """Preserve native code/config identities when resuming an interrupted run."""
    roots = ["gen-benchmark"]
    if model == "magicdrivev2":
        roots.append("MagicDrive-V2")
    identity = hashlib.sha256(os.fsencode(engine.resolve()))
    for name in roots:
        for file in sorted((VENDOR / name).rglob("*")):
            if "__pycache__" in file.parts or not file.is_file():
                continue
            identity.update(file.relative_to(VENDOR).as_posix().encode() + b"\0")
            identity.update(hashlib.sha256(file.read_bytes()).digest())
    runtime = logs / ("runtime-" + identity.hexdigest())
    if runtime.is_dir():
        return runtime
    with tempfile.TemporaryDirectory(prefix=".build-", dir=logs) as staging:
        build = Path(staging) / "code"
        build.mkdir()
        for name in roots:
            overlay(engine / name, VENDOR / name, build / name)
        build.rename(runtime)
    return runtime


def launch_plan(cfg, model, runtime, root, attempt, env):
    bench = runtime / "gen-benchmark"
    prediction = root / "predictions"
    indices = ",".join(str(i) for i in range(len(cfg["scenes"])))
    gpu_count = 4 if model == "magicdrivev2" else 2
    seed = "3407"
    env = dict(env)
    for name in list(env):
        if name.startswith("GENBENCH_") or name.startswith("CHALLENGE_"):
            env.pop(name)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="6",
               MKL_NUM_THREADS="6", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", GENBENCH_GT2_ROOT=str(root),
               GENBENCH_PREDICTION_ROOT=str(prediction), GENBENCH_GENERATION_SET="participant_joint",
               GENBENCH_SCENE_INDICES=indices, GENBENCH_LIMIT="0", GENBENCH_USE_CATK_EGO_PREDICTION="0",
               DEMO_ROOT=str(root))
    paths = [VENDOR / "guard"]
    if model == "magicdrivev2":
        project = runtime / "MagicDrive-V2"
        python = cfg["v2_python"]
        paths.extend([project / "rd", bench])
    else:
        project = bench / PROJECTS[model]
        settings = cfg["models"][model] if "models" in cfg else read(cfg["models_json"])[model]
        python = settings["python"]["renderer"]
        paths.extend([project, bench])
        if model in {"basedreamer", "dreamforge"}:
            world = bench / WORLD_DREAMER
            paths.extend([world / ".build/dit/dreamforgedit", world / "scripts"])
            native_packages = {
                "basedreamer": ["base/bevfusion", "base/diffusers/src", "base/xformers", "base/nuplan-devkit"],
                "dreamforge": ["forge/mmdetection3d", "forge/diffusers/src"],
            }
            paths.extend(world / ".build" / name for name in native_packages[model])
        if model == "magicdrive":
            paths = [VENDOR / "guard", bench / "MagicDrive-video", bench / "MagicDrive-bevfusion",
                     bench / "MagicDrive-video/third_party/diffusers/src", bench / "MagicDrive-xformers", bench]
        if model == "opendwm":
            paths.insert(1, project / "src")
        if model == "panacea":
            paths.append(project / "third_party/mmdetection3d")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    cuda_home = cfg.get("cuda_homes", {}).get(model)
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = os.pathsep.join([str(Path(cuda_home) / "bin"), env.get("PATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(Path(cuda_home) / "lib64"), env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    common = ["--data-root", root, "--prediction-root", prediction, "--generation-set", "participant_joint",
              "--seed", seed, "--scene-indices", indices]
    dist = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
    if model == "magicdrivev2":
        args = [python, project / "challenge_generation/genbench_gt2.py", "run", "--input-root", root,
                "--trajectory-root", root, "--output-root", prediction / model, "--scene-list",
                root / "metadata/scene_list.json", "--standard-layout", "--gpus-per-worker", str(gpu_count),
                "--temporal-mode", "full89"]
    elif model in INDEPENDENT:
        args = [python, "-u", project / "infer_genbench_gt2.py", *common, "--checkpoint",
                settings["render_checkpoint"], "--sd15-root", settings["sd15"], "--world-size", "2"]
        if model == "dreamforge":
            args.extend(["--single-checkpoint", settings["single_checkpoint"], "--num-inference-steps", "20"])
        if model == "magicdrive":
            args.extend(["--video-root", bench / "MagicDrive-video"])
        gpu_ids = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(gpu_ids) != gpu_count or not all(gpu_ids):
            raise ValueError("Select two GPU IDs in CUDA_VISIBLE_DEVICES for the independent workers")
        cwd = bench / "MagicDrive-video" if model == "magicdrive" else project
        base_port = 21000 + int(env["SLURM_JOB_ID"]) % 15000
        return [(list(map(str, [*args, "--rank", rank])), cwd,
                 dict(env, CUDA_VISIBLE_DEVICES=gpu_ids[rank], RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(base_port + rank)))
                for rank in range(min(len(cfg["scenes"]), gpu_count))]
    elif model == "opendwm":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        args = [*dist, project / "infer_genbench_gt2.py", *common, "--adapter-config",
                project / "examples/genbench_gt2_unimlvg.json", "--base-model", settings["base_model"],
                "--checkpoint", settings["render_checkpoint"]]
    elif model == "panacea":
        args = [*dist, project / "inference_genbench_gt2.py", *common, "--base", project / "configs/inference_nuscenes.yaml",
                "--ckptpath", settings["render_checkpoint"], "--clip_ckpt", settings["clip_checkpoint"],
                "--vae_decode_batch_size", "1", "--reference-mode", "autoregressive", "--ordered-targets"]
    else:
        args = [python, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_processes=2", "--num_machines=1",
                "--main_process_port=0", "--mixed_precision=bf16", "--dynamo_backend=no", "--",
                project / "horizondrive/eval.py", "--train_mode=unified", "--mixed_precision=bf16", "--seed=" + seed,
                "--crossview_attn_type=full", "--low_vram", "--config_path=" + str(project / "config/horizondrive_genbench_gt2.yaml"),
                "--pretrained_model_name_or_path=" + settings["wan_checkpoint"], "--output_dir=" + str(attempt / "horizondrive"),
                "--vae_path=" + str(Path(settings["render_checkpoint"]).with_name("horizondrive-vae.pkl")),
                "--transformer_path=" + settings["render_checkpoint"]]
    return [(list(map(str, args)), project, env)]


def render(cfg, model):
    root = verify_stage(cfg)
    if model != "magicdrivev2" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this native adapter inside a GPU Slurm allocation")
    engine = Path(cfg["engine_code_root"])
    native = engine / "MagicDrive-V2/rd" if model == "magicdrivev2" else engine / "gen-benchmark" / PROJECTS[model]
    if not native.is_dir():
        raise FileNotFoundError(f"Missing installed native project for {model}: {native}")
    (root / "predictions" / model).mkdir(parents=True, exist_ok=True)
    logs = Path(cfg["work_dir"]) / "logs" / model
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=logs))
        runtime = runtime_for(engine, model, logs)
        plans = launch_plan(cfg, model, runtime, root, attempt, os.environ)
        write(attempt / "commands.json", [dict(argv=cmd, cwd=str(cwd)) for cmd, cwd, _ in plans])
        before = {scene for scene in cfg["scenes"] if (
            root / "predictions" / model / "nuscenes/scenes" / scene / "status.json").is_file()}
        start = time.monotonic()
        workers = []
        with ExitStack() as files:
            try:
                for i, (argv, cwd, env) in enumerate(plans):
                    stream = files.enter_context((attempt / f"worker-{i}.log").open("w"))
                    workers.append(subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT))
                codes = [worker.wait() for worker in workers]
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                for worker in workers:
                    worker.wait()
        write(attempt / "process.json", dict(return_codes=codes, wall_seconds=time.monotonic()-start,
              existing_status_before_launch=sorted(before)))
        if any(codes):
            raise RuntimeError(f"{model} exited with {codes}. See {attempt}")
        command = [cfg["catk_python"], str(Path(__file__).with_name("verify_rgb.py")), str(root), model,
                   str(attempt / "VERIFIED.json"), *cfg["scenes"]]
        subprocess.run(command, check=True, env=dict(os.environ, PYTHONPATH=str(VENDOR / "gen-benchmark"),
                                                     PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"))
        return attempt / "VERIFIED.json"

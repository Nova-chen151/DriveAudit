"""Build a single-scene DriveAudit feedback run from the shared generation profile."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

from driveaudit.generation import MODELS
from driveaudit.generation.__main__ import load_config
from driveaudit.generation.renderer import PROJECTS, WORLD_DREAMER
from driveaudit.pipeline import Pipeline
from driveaudit.resources import resource_root

PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def configure(args):
    pipe = Pipeline(args.pipeline)
    cfg = load_config(pipe.generation)
    native = Path(cfg["engine_code_root"])
    bench = native / "gen-benchmark"
    world = bench / WORLD_DREAMER
    resources = pipe.runtime_environment()
    weights = Path(resources["DRIVEAUDIT_AGENT_ASSETS"]).parent
    public = args.participant_scene.resolve(strict=True)
    scene = args.render_scene.resolve(strict=True)
    if scene.name != public.name or json.loads((public/"task.json").read_text())["source_dataset"] != "nuscenes":
        raise ValueError("Use the same public nuScenes case for traffic and rendering")
    if (scene/"history/agents.csv").read_bytes() != (public/"history/agents.csv").read_bytes():
        raise ValueError("Public and renderer history differ")
    with (public/"conditioning/ego_trajectory.csv").open() as stream:
        last = list(csv.DictReader(stream))[-1]
    goal = [float(last[k]) for k in ("x", "y")]
    if int(last["frame_index"]) != 89 or last["valid"].lower() not in ("true","1") or not all(map(math.isfinite,goal)):
        raise ValueError("Valid public goal at frame 89 required")
    model = args.renderer
    v2 = model == "magicdrivev2"
    project = native/"MagicDrive-V2/rd" if v2 else bench/PROJECTS[model]
    settings = dict(cfg["models"].get(model, {}))
    python = cfg["v2_python"] if v2 else settings.pop("python")["renderer"]
    renderer_paths = [project, bench, world/"scripts"]
    groups = {"basedreamer":["base/bevfusion","base/diffusers/src","base/xformers","base/nuplan-devkit"],
              "dreamforge":["forge/mmdetection3d","forge/diffusers/src"]}
    if model in groups:
        renderer_paths.extend(world/".build"/x for x in ["dit/dreamforgedit", *groups[model]])
    if model == "magicdrive":
        video = bench/"MagicDrive-video"
        settings["video_root"] = str(video)
        renderer_paths.extend([video, video/"third_party/diffusers/src",bench/"MagicDrive-bevfusion",bench/"MagicDrive-xformers"])
    if model == "horizondrive":
        settings["vae_checkpoint"] = str(Path(settings["render_checkpoint"]).with_name("horizondrive-vae.pkl"))
        settings["render_config"] = str(project/"config/horizondrive_genbench_gt2.yaml")
    if model == "opendwm":
        renderer_paths.insert(0,project/"src")
        settings["render_config"] = str(project/"examples/genbench_gt2_unimlvg.json")
    if model == "panacea":
        settings["render_config"] = str(project/"configs/inference_nuscenes.yaml")
        renderer_paths.append(project/"third_party/mmdetection3d")
    if v2:
        renderer_paths.append(project.parent)
        settings.update(render_checkpoint=str(weights/"magicdrivev2/ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt"),
            vae_checkpoint=str(weights/"magicdrivev2/pretrained/CogVideoX-2b"),
            text_checkpoint=str(weights/"magicdrivev2/pretrained/t5-v1_1-xxl"),
            render_config=str(project/"configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py"))
    for key, value in settings.items():
        if key != "python" and not Path(value).exists():
            raise FileNotFoundError(value)
    workers = 4 if v2 or model == "opendwm" else 1
    role_paths = {"renderer":renderer_paths, "traffic":[native/"catk"],
                  "policy":[resource_root()/"models/agents/DrivoR",Path(resources["NUPLAN_DEVKIT_ROOT"])]}
    env = {r:{"PYTHONPATH":os.pathsep.join(map(str,[*paths,PACKAGE_PARENT]))} for r,paths in role_paths.items()}
    if model in cfg.get("cuda_homes", {}):
        cuda = cfg["cuda_homes"][model]
        env["renderer"]["CUDA_HOME"] = cuda
    env["policy"]["DRIVEAUDIT_AGENT_ASSETS"] = str(weights/"agents")
    return dict(settings,model=model, bench=str(bench), scene=str(scene), participant_scene=str(public),
        official=str(native/"catk/third_party/catk-official"),catk_adapter=str(native/"catk"),
        policy_checkpoint=str(weights/"agents/DrivoR/weights/drivor_Nav1_25epochs.pth"),
        python=dict(traffic=cfg["catk_python"],renderer=python,policy=str(pipe.python("policy"))),
        policy_image_protocol="generated_only_per_step_sha256",seed=3407, role_env=env,
        renderer_processes=workers, renderer_module=None if model=="basedreamer" else "render_"+model,
        gpu_layout=dict(traffic="0",policy="0",renderer=",".join(str(i) for i in range(1,workers+1))),
        role_cwd=dict(renderer=settings.get("video_root",str(project))),
        goal_navigation=dict(enabled=True,goal_anchor_xy=goal,arrival_radius_m=.5,turn_threshold_deg=30.))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline",type=Path,default=resource_root()/"configs/pipeline.json")
    p.add_argument("--renderer",choices=MODELS,default="basedreamer")
    p.add_argument("--participant-scene",type=Path,required=True)
    p.add_argument("--render-scene",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    profile=configure(a)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(profile,f,indent=2)


if __name__ == "__main__":
    main()

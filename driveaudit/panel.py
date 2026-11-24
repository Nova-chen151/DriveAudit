"""Run a native policy or the complete paired six-policy panel from a pipeline configuration."""
import os
import shlex
import subprocess

from .agents.registry import POLICIES
from .pipeline import Pipeline, ROOT, resolve


def run(file, action, policy=None, side=None, plan=False):
    pipeline = Pipeline(file)
    cfg = pipeline.config
    receipts = pipeline.output / "panel-input"
    settings = cfg["panel"]
    env = os.environ.copy()
    env.update(pipeline.runtime_environment())
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if action == "score":
        cmd = [pipeline.python("evaluation"), "-m", "driveaudit.evaluation.panel",
               "--ego-root", receipts, "--scenes", pipeline.paired / "scenes.json",
               "--quality", pipeline.output / "quality.json", "--trs", pipeline.output / "trs/result.json",
               "--output", pipeline.output / "panel", "--workers", str(settings.get("scoring_workers", 1))]
    else:
        model_name = POLICIES[policy][0]
        assets = env.get("DRIVEAUDIT_AGENT_DEPENDENCIES") or env["DRIVEAUDIT_AGENT_ASSETS"]
        env["NUPLAN_DEVKIT_ROOT"] = os.path.join(assets, model_name, "nuplan-devkit")
        public = pipeline.conditions / "nuscenes300_challenger_v1/scenes"
        source = (["--rgb-scenes", pipeline.conditions / "predictions" / pipeline.model / "nuscenes/scenes"]
                  if side == "generated" else ["--real-mapping", resolve(pipeline.file, cfg["organizer_mapping"])])
        cmd = [pipeline.python("policy"), "-m", "driveaudit.agents.replay", "--policy", policy,
               "--service-python", resolve(pipeline.file, settings["interpreters"][policy]), "--source", side,
               "--public-scenes", public, *source, "--cases", *pipeline.cases,
               "--command", cfg.get("command", "straight"), "--output", receipts / policy / f"{side}.jsonl"]
    cmd = list(map(str, cmd))
    print(shlex.join(cmd), flush=True)
    if not plan:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

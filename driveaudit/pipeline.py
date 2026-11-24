"""Dispatch a participant run and its paired organizer replay in native environments."""
import json
import os
from pathlib import Path

from driveaudit.resources import resource_root
import subprocess
import sys

from .generation import MODELS
from .generation.__main__ import load_config

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("traffic", "stage", "render", "policy-generated", "policy-real", "pair", "quality", "score")


def resolve(config_file, value):
    value = os.path.expandvars(str(value))
    if "${" in value:
        raise ValueError(f"Unset environment variable in {value}")
    # Keep the invoked interpreter inside its virtual environment. Resolving
    # bin/python through its symlink can silently select the base environment.
    return Path(os.path.abspath(config_file.parent / Path(value).expanduser()))


class Pipeline:
    def __init__(self, file):
        self.file = file.resolve()
        self.config = json.loads(self.file.read_text())
        self.generation = resolve(self.file, self.config["generation"])
        cfg = load_config(self.generation)
        self.cases = cfg["scenes"]
        self.model = self.config["renderer"]
        if self.model not in MODELS:
            raise ValueError(f"Unknown renderer {self.model}")
        self.conditions = resolve(self.file, self.config.get("conditions", Path(cfg["work_dir"]) / "conditions"))
        self.output = resolve(self.file, self.config["output"])
        self.paired = self.output / "staged"
        self.plans = self.output / "plans"
        self.profile = resolve(self.file, self.config.get("measurement_profile", resource_root() / "configs/quality.yaml"))

    def commands(self, step):
        if step in ("traffic", "stage", "render"):
            cmd = [sys.executable, "-m", "driveaudit.generation", step, "--config", self.generation]
            if step == "render":
                cmd.extend(["--model", self.model])
            return [cmd]
        # Participant generation need not resolve any organizer paths.
        scenes_root = self.conditions / "nuscenes300_challenger_v1/scenes"
        predictions = self.conditions / "predictions"
        if step.startswith("policy-"):
            side = step.split("-")[1]
            source = (["--rgb-scenes", predictions / self.model / "nuscenes/scenes"] if side == "generated"
                      else ["--real-mapping", resolve(self.file, self.config["organizer_mapping"])])
            return [[self.python("policy"), "-m", "driveaudit.agents.predict", "--source", side,
                     "--public-scenes", scenes_root, *source, "--cases", *self.cases,
                     "--command", self.config.get("command", "straight"), "--output", self.plans / f"{side}.jsonl"]]
        if step == "pair":
            cmd = [sys.executable, "-m", "driveaudit.organizer.stage", "--conditions", self.conditions,
                   "--predictions", predictions, "--renderer", self.model, "--mapping",
                   resolve(self.file, self.config["organizer_mapping"]), "--output", self.paired]
            for case in self.cases:
                cmd.extend(["--scene", case])
            return [cmd]
        scenes = self.paired / "scenes.json"
        if step == "quality":
            args = ["--scenes", scenes, "--output", self.output, "--profile", self.profile]
            module = ["-m", "driveaudit.organizer.measure"]
            cmds = [[self.python("evaluation"), *module, "trs", *args]]
            for case in self.cases:
                for side in ("generated", "real"):
                    for task, runtime in (("video", "vision"), ("tvc", "evaluation")):
                        cmds.append([self.python(runtime), *module, task, *args, "--case", case, "--side", side])
            cmds.append([self.python("evaluation"), "-m", "driveaudit.organizer.collect",
                         "--scenes", scenes, "--output", self.output])
            return cmds
        if step != "score":
            raise ValueError(step)
        return [[self.python("evaluation"), "-m", "driveaudit.evaluation", "--generated", self.plans / "generated.jsonl",
                 "--real", self.plans / "real.jsonl", "--scenes", scenes, "--quality", self.output / "quality.json",
                 "--trs", self.output / "trs/result.json", "--output", self.output / "paired"]]

    def runtime_environment(self):
        values = {}
        for key, value in self.config.get("runtime_env", {}).items():
            if value.startswith(("./", "../")):
                values[key] = str(resolve(self.file, value))
            else:
                values[key] = os.path.expanduser(os.path.expandvars(value))
        return values

    def python(self, runtime):
        return resolve(self.file, self.config["python"][runtime])

    def run(self, start, stop, plan=False):
        import shlex
        first, last = STEPS.index(start), STEPS.index(stop)
        if last < first:
            raise ValueError("End stage precedes start stage")
        if first <= STEPS.index("render") and "conditions" in self.config:
            raise ValueError("External conditions are for replay. Start at policy-generated or remove the override.")
        for step in STEPS[first:last+1]:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            if step not in ("traffic", "stage", "render"):
                env.update(self.runtime_environment())
                if step == "quality":
                    env["DRIVEAUDIT_VISION_PYTHON"] = str(self.python("vision"))
            for command in self.commands(step):
                command = list(map(str, command))
                print(step, shlex.join(command), flush=True)
                if not plan:
                    subprocess.run(command, cwd=ROOT, env=env, check=True)

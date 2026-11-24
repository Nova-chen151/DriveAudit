"""Run the native DriveAudit metrics on the organizer's paired scene trees."""
import argparse
import json
import logging
import os
from pathlib import Path

from .stage import read, write
from .runtime import native_evaluator


def measure(task, scenes_file, output, profile, case=None, side=None):
    native_evaluator()
    import yaml
    from driveaudit_eval.utils.io import jsonable

    scenes = read(scenes_file)
    profile_text = profile.read_text()
    if "DRIVEAUDIT_DINO_SOURCE" not in os.environ and "DRIVEAUDIT_METRIC_CACHE" in os.environ:
        profile_text = profile_text.replace("${DRIVEAUDIT_DINO_SOURCE}", str(
            Path(os.environ["DRIVEAUDIT_METRIC_CACHE"]) / "home/torch/hub/facebookresearch_dinov2_main"))
    settings = yaml.safe_load(os.path.expandvars(profile_text))["evaluation"]
    if task == "trs":
        from driveaudit_eval.trajectory.population import evaluate_paired

        base = scenes_file.parent.resolve()
        evaluate_paired([row["generated"] for row in scenes.values()], base / "generated",
                        base / "real", output / "trs", resume=False, bootstrap=False)
        return
    scene = Path(scenes[case][side]).resolve()
    dest = output / case / side / task
    dest.mkdir(parents=True, exist_ok=True)
    if task == "tvc":
        from driveaudit_eval.consistency.input_gt_full import evaluate_scene
        from driveaudit_eval.metrics.core import _tvc

        cfg = settings["consistency"]
        cfg.update(profile="core", candidate_role="generated" if side == "generated" else "ground_truth",
                   native_nuscenes_root=None, retain_intermediates=True)
        cfg.pop("metric_detail_cleanup", None)
        cfg.pop("metric_detail_retention", None)
        result = evaluate_scene(scene, cfg, dest, logging.getLogger(__name__))
        value = _tvc(result, {})
    else:
        from driveaudit_eval.video.evaluator import evaluate
        from driveaudit_eval.video.config import DEFAULT_CONFIG
        cfg = settings["video"]
        cfg.update(scene_paths=[str(scene)], reference_scene_paths=[scenes[case]["real"]],
                   mode="input_gt", universal_protocol=True, resume=False)
        for name in DEFAULT_CONFIG["metrics"].keys() | cfg["metrics"].keys():
            cfg["metrics"].setdefault(name, {})["enabled"] = name in ("loftr", "epipolar")
        for name, item in cfg["visual_quality"].items():
            if isinstance(item, dict):
                item["enabled"] = name in ("clipiqa", "dino_temporal")
        # CLIP-IQA+ is installed in the main visual environment; the DINO
        # subprocess retains its native checkpoint and interpreter from the profile.
        cfg["visual_quality"].pop("python", None)
        cfg["runtime"]["device"] = "cuda"
        value = evaluate(cfg, {"root": str(Path.cwd()), "output_dir": dest})
    write(dest / "measured.json", jsonable(value))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["video", "tvc", "trs"])
    for arg in ("scenes", "output", "profile"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--case")
    p.add_argument("--side", choices=["generated", "real"])
    a = p.parse_args()
    if a.task != "trs" and not (a.case and a.side):
        p.error("video and tvc require --case and --side")
    measure(a.task, a.scenes.resolve(), a.output.resolve(), a.profile, a.case, a.side)


if __name__ == "__main__":
    main()

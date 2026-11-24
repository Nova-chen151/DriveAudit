"""Selected-scene CATK preflight, provenance, and safe resume for HorizonDrive.

This module is CPU-only. Source trees are read-only; all artifacts live under
input-catk/predictions/horizondrive. The launcher holds its lock for the full run.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from genbench_gt2.constants import CAMERAS, HISTORY_FRAMES
from genbench_gt2.dataset import load_generation_set, load_scene_inputs
from genbench_gt2.output import ValidationResult, checkpoint_identity, validate_completed_scene


STATIC_FILES = (
    "generation_input.json", "scene_context.json", "timeline.json",
    "calibration/cameras.json", "calibration/reference_ego_poses.json",
    "calibration/history_camera_poses.json", "history/agents.csv", "history/ego.csv",
    "future/ego_trajectory.csv", "map/semantic_map.npz", "map/map_meta.json",
    "map/generation_vectors.json",
)
OVERRIDE = "future/future_trajectories.csv"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conditioning_provenance(scene):
    static_root = (scene.asset_root or scene.root).resolve()
    trajectory_root = (scene.trajectory_root or scene.root).resolve()
    if static_root == trajectory_root:
        raise ValueError("CATK agent-only conditioning requires distinct static and overlay roots")
    files = list(STATIC_FILES) + [
        "history/rgb/{}/{:06d}.jpg".format(camera, frame)
        for camera in CAMERAS for frame in HISTORY_FRAMES
    ]
    record = {
        "contract": "input-gt-static_catk-future-agents_v1",
        "input_relative_path": scene.relative_path,
        "static_root": str(static_root),
        "trajectory_root": str(trajectory_root),
        "overrides": [OVERRIDE],
        "uses_catk_ego_prediction": False,
        "static_sha256": {name: sha256_file(static_root / scene.relative_path / name) for name in files},
        "overlay_sha256": {OVERRIDE: sha256_file(trajectory_root / scene.relative_path / OVERRIDE)},
    }
    record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return record


def selected_scenes(data_root, scene_list, generation_set="balanced_500", expected_scenes=None):
    requested = json.loads(Path(scene_list).read_text())
    if (not isinstance(requested, list) or not requested or
            any(not isinstance(value, str) for value in requested)):
        raise ValueError("--scene-list must be a nonempty JSON list of relative scene paths")
    if len(set(requested)) != len(requested):
        raise ValueError("--scene-list contains duplicate scene paths")
    if expected_scenes is not None and len(requested) != expected_scenes:
        raise ValueError("expected {} selected scenes, found {}".format(expected_scenes, len(requested)))
    available = load_generation_set(Path(data_root), generation_set)
    by_path = {scene.relative_path: scene for scene in available}
    unknown = set(requested) - set(by_path)
    if unknown:
        raise ValueError("scene-list paths absent from static generation set: {}".format(sorted(unknown)))
    # Keep the canonical input-gt indices/order used by the dataset and sharding.
    return [scene for scene in available if scene.relative_path in set(requested)]


def prediction_path(scene):
    return scene.root / "predictions" / "horizondrive" / scene.dataset / "scenes" / scene.scene_id


def validate_prediction(path, scene, expected_conditioning=None, expected_runtime=None):
    if expected_conditioning is not None:
        try:
            status = json.loads((Path(path) / "status.json").read_text())
        except (OSError, ValueError):
            return ValidationResult(False, 0, "missing conditioning provenance/status")
        if status.get("conditioning_provenance") != expected_conditioning:
            return ValidationResult(False, 0, "conditioning provenance differs from selected CATK inputs")
        if expected_runtime is not None and status.get("runtime_provenance") != expected_runtime:
            return ValidationResult(False, 0, "runtime checkpoint/config/seed differs from current generation")
    return validate_completed_scene(path, "horizondrive", scene)


def archive_incompatible(scene, provenance, archive_tag, runtime=None):
    final = prediction_path(scene)
    if not final.exists() and not final.is_symlink():
        return None
    model_root = scene.root.resolve() / "predictions" / "horizondrive"
    try:
        final.resolve().relative_to(model_root)
    except ValueError:
        raise ValueError("prediction scene escapes model output root: {}".format(final))
    check = validate_prediction(final, scene, provenance, runtime)
    if check.valid:
        return None
    target = model_root / "_archives" / archive_tag / scene.dataset / "scenes" / scene.scene_id
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    final.rename(target)
    return {"scene": scene.relative_path, "path": str(target), "reason": check.reason}


def verify_campaign_scene(scene, provenance):
    campaign_path = os.environ.get("GENBENCH_CATK_CAMPAIGN")
    if not campaign_path:
        return
    campaign = json.loads(Path(campaign_path).read_text())
    expected = {item["relative_path"]: item["conditioning_provenance"] for item in campaign["scenes"]}
    if expected.get(scene.relative_path) != provenance:
        raise ValueError("scene conditioning changed after CATK preflight: {}".format(scene.relative_path))


def campaign_runtime():
    path = os.environ.get("GENBENCH_CATK_CAMPAIGN")
    if not path:
        raise ValueError("CATK agent-only dataset requires a prepared GENBENCH_CATK_CAMPAIGN")
    return json.loads(Path(path).read_text())["runtime_provenance"]


def runtime_provenance(config, checkpoints, seed):
    return {"checkpoint": checkpoint_identity(checkpoints), "seed": seed,
            "config_path": str(config.resolve()), "config_sha256": sha256_file(config),
            "settings": {"mixed_precision": "bf16", "crossview_attn_type": "full",
                         "num_processes": 2, "num_condition_images": 11,
                         "num_unroll_steps": 8, "latent_rollout": True},
            "implementation_sha256": {
                name: sha256_file(Path(__file__).parent / name)
                for name in ("catk_generation.py", "trainer.py", "datasets/genbench_gt2_dataset.py")
            }}


def _check_mode(data_root):
    if os.environ.get("GENBENCH_CATK_AGENT_ONLY") != "1":
        raise ValueError("requires GENBENCH_CATK_AGENT_ONLY=1")
    if os.environ.get("GENBENCH_USE_CATK_EGO_PREDICTION") != "0":
        raise ValueError("requires GENBENCH_USE_CATK_EGO_PREDICTION=0")
    if not os.environ.get("GENBENCH_STATIC_ASSET_ROOT"):
        raise ValueError("requires GENBENCH_STATIC_ASSET_ROOT")
    if not (data_root / "CATK_INFERENCE_MANIFEST.csv").is_file():
        raise ValueError("CATK root lacks CATK_INFERENCE_MANIFEST.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--generation-set", default="balanced_500")
    parser.add_argument("--expected-scenes", type=int, default=80)
    parser.add_argument("--required-path", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--selection-only", action="store_true", help="read-only launch planning; defer input/image audit to the job")
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    _check_mode(data_root)
    runtime = runtime_provenance(args.runtime_config, args.checkpoint_path, args.seed)
    if args.verify and runtime != campaign_runtime():
        raise ValueError("runtime checkpoint/config/seed changed after CATK preflight")
    for path in args.required_path:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing or empty required file: {}".format(path))
    scenes = selected_scenes(data_root, args.scene_list, args.generation_set, args.expected_scenes)
    if args.manifest:
        args.manifest.resolve().relative_to(data_root / "predictions" / "horizondrive")

    def inspect(scene):
        if args.selection_only:
            overlay = (scene.trajectory_root / scene.relative_path / OVERRIDE)
            if not overlay.is_file():
                raise ValueError("required CATK future agent overlay is missing: {}".format(overlay))
            return {"relative_path": scene.relative_path, "index": scene.index,
                    "ready": False, "readiness_reason": "not audited in selection-only mode"}
        # Validate only selected inputs; never open evaluation future RGB.
        load_scene_inputs(scene)
        provenance = conditioning_provenance(scene)
        if args.verify:
            verify_campaign_scene(scene, provenance)
        check = validate_prediction(prediction_path(scene), scene, provenance, runtime)
        return {"relative_path": scene.relative_path, "index": scene.index,
                "conditioning_provenance": provenance,
                "ready": check.valid, "readiness_reason": check.reason}

    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for count, record in enumerate(executor.map(inspect, scenes), 1):
            records.append(record)
            if count % 10 == 0 or count == len(scenes):
                print("CATK selected input audit {}/{}".format(count, len(scenes)), flush=True)
    report = {
        "model": "horizondrive", "scope": "explicit_scene_list_only",
        "input_audit_performed": not args.selection_only,
        "runtime_provenance": runtime,
        "scene_list": str(args.scene_list.resolve()), "scene_list_sha256": sha256_file(args.scene_list),
        "selected_count": len(scenes), "dataset_counts": dict(Counter(scene.dataset for scene in scenes)),
        "static_root": str(Path(os.environ["GENBENCH_STATIC_ASSET_ROOT"]).resolve()),
        "trajectory_root": str(data_root), "overrides": [OVERRIDE],
        "scene_indices_csv": ",".join(str(scene.index) for scene in scenes), "scenes": records,
        "ready_count": sum(record["ready"] for record in records),
    }
    if args.prepare:
        if os.environ.get("GENBENCH_CATK_LOCK_HELD") != "1":
            raise ValueError("--prepare must run under the launcher model lock")
        if args.manifest is None:
            raise ValueError("--prepare requires --manifest")
        # Validate every selected input before moving any prior output.
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report["archived"] = []
        for scene, record in zip(scenes, records):
            archived = archive_incompatible(scene, record["conditioning_provenance"], tag, runtime)
            if archived:
                report["archived"].append(archived)
    if args.manifest:
        model_root = data_root / "predictions" / "horizondrive"
        args.manifest.resolve().relative_to(model_root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.manifest)
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, sort_keys=True), flush=True)
    if args.verify and report["ready_count"] != len(scenes):
        raise ValueError("only {}/{} selected scenes have complete current-condition outputs".format(report["ready_count"], len(scenes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

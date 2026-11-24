"""Bind each measured quality dimension to the scene and side used for policy scoring."""
import argparse
from pathlib import Path

from .stage import read, write
from .runtime import native_evaluator


def collect(scenes_file, directory):
    native_evaluator()
    from driveaudit_eval.metrics.population_testing import FIELDS
    from driveaudit_eval.quality import scene_quality
    from driveaudit_eval.testing import scene_evidence, scene_key
    from driveaudit_eval.utils.io import file_sha256

    trajectory_manifest = read(directory / "trs/result.json")["source_manifest"]
    records = {}
    for case, sides in read(scenes_file).items():
        records[case] = {}
        for side, path in sides.items():
            scene = Path(path)
            scene_id = scene_key(str(scene))
            base = directory / case / side
            video = read(base / "video/measured.json")
            tvc = read(base / "tvc/measured.json")
            mtpc = read(base / "tvc/chain_progress.json")["mtpc"]
            if mtpc.get("execution_status") != "completed":
                raise RuntimeError(f"MTPC tracking did not complete: {case}/{side}")

            summary = {"core_metrics": {"tvc": [tvc]}, "video": video}
            context = {scene_id: {"dataset": "nuscenes", "scene_path": str(scene)}}
            row = scene_quality(summary, base / "video", context, fields=FIELDS)[scene_id]
            # Distinguish measured low support from skipped/failed model execution.
            statuses = video.get("visual_quality", {}).get("metrics", {})
            for name in ("clipiqa", "dino_temporal"):
                if statuses.get(name, {}).get("status") != "valid":
                    raise RuntimeError(f"{case}/{side}/{name}: {statuses.get(name, {}).get('reason')}")
            if video.get("geometry_consistency", {}).get("skipped", False):
                raise RuntimeError(f"Epipolar matching did not run: {case}/{side}")
            evidence = scene_evidence({"scene_path": str(scene), "frames": []})
            row.update({key: evidence[key] for key in ("scene_hash", "rgb_hash", "history_hash")})
            expected = trajectory_manifest[scene_id]["candidate" if side == "generated" else "reference"]
            row["trajectory_inputs"] = {name: file_sha256(scene / name) for name in expected}
            if row["trajectory_inputs"] != expected:
                raise ValueError(f"{case}/{side}: trajectory evidence changed since TRS extraction")
            row["computation_status"] = "completed"
            records[case][side] = row
    write(directory / "quality.json", records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    collect(args.scenes, args.output)


if __name__ == "__main__":
    main()

"""Join anonymized predictions with organizer references strictly after inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def component(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError(f"Invalid path component {value!r}")
    return value


def link(source, target):
    target.symlink_to(source.resolve(strict=True), target_is_directory=source.is_dir())


def stage(mapping, conditions, predictions, renderer, cases, output):
    """Build the native paired scene format in an isolated organizer directory."""
    renderer = component(renderer)
    records = read(mapping)
    by_case = {r["public_scene"]: r for r in records}
    if len(by_case) != len(records) or not cases or len(set(cases)) != len(cases):
        raise ValueError("Expected unique scene IDs in selection and mapping")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Choose a fresh organizer directory: {output}")
    scenes = {}
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage_root = Path(temporary) / "staged"
        for case in cases:
            row = by_case[component(case)]
            original = Path(row["source_path"]).resolve(strict=True)
            dataset = original.parent.parent.name
            relative = Path(dataset) / "scenes" / case
            public = conditions / relative
            prediction = predictions / renderer / "nuscenes" / "scenes" / case
            status = read(prediction / "status.json")
            if (status["state"], status["model"], status["scene_id"], status["observed_image_count"]) != (
                "complete", renderer, case, 480
            ) or status["input_relative_path"] != relative.as_posix():
                raise ValueError(f"Unexpected generation receipt for {case}")
            # Native history fields and timestamps must match before restoring actor IDs.
            for scene in (original, public):
                times = [f["reference_time_us"] for f in read(scene / "timeline.json")["frames"]]
                if scene == original:
                    original_times = times
                elif times != original_times:
                    raise ValueError(f"Reference time mismatch: {case}")
            if read(original / "calibration/reference_ego_poses.json")["ego_to_anchor"] != read(
                public / "calibration/reference_ego_poses.json"
            )["ego_to_anchor"]:
                raise ValueError(f"Reference ego pose mismatch: {case}")
            reverse = {alias: real for real, alias in row["agent_ids"].items()}
            if len(reverse) != len(row["agent_ids"]):
                raise ValueError(f"Ambiguous actor identity: {case}")
            scenes[case] = {}
            for kind in ("generated", "real"):
                dst = stage_root / kind / relative
                (dst / "future").mkdir(parents=True)
                for name in ("history", "timeline.json", "calibration", "scene_context.json", "map", "generation_input.json"):
                    link(original / name, dst / name)
                future = dst / "future"
                # Both policies receive the same public route derived from the released SE(3) poses.
                link(public / "future/ego_trajectory.csv", future / "ego_trajectory.csv")
                if kind == "real":
                    for name in ("rgb", "rgb_manifest.json", "future_trajectories.csv"):
                        link(original / "future" / name, future / name)
                else:
                    with (public / "future/future_trajectories.csv").open() as inp, (future / "future_trajectories.csv").open("w") as out:
                        rows = csv.DictReader(inp)
                        writer = csv.DictWriter(out, fieldnames=rows.fieldnames)
                        writer.writeheader()
                        for actor in rows:
                            actor["agent_id"] = reverse[actor["agent_id"]]
                            writer.writerow(actor)
                    rgb = (prediction / "future/rgb").resolve(strict=True)
                    link(rgb, future / "rgb")
                    manifest = read(original / "future/rgb_manifest.json")
                    for frame in manifest["images"]:
                        name = component(frame["camera"])
                        path = rgb / name / f"{int(frame['frame_index']):06d}.jpg"
                        frame.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     file=str(path), source_file=str(path), raw_path=str(path))
                        frame.pop("source_sha256", None)
                    manifest.update(source_kind="generated", renderer=renderer)
                    write(future / "rgb_manifest.json", manifest)
                scenes[case][kind] = str(output / kind / relative)
        write(stage_root / "scenes.json", scenes)
        os.rename(stage_root, output)
    return scenes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("mapping", "conditions", "predictions", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--renderer", required=True)
    p.add_argument("--scene", action="append", required=True)
    a = p.parse_args()
    stage(a.mapping, a.conditions, a.predictions, a.renderer, a.scene, a.output)


if __name__ == "__main__":
    main()

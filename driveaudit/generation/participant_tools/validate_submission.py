import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image

CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]
FIELDS = [
    "frame_index",
    "agent_id",
    "category",
    "is_dynamic",
    "x",
    "y",
    "z",
    "width",
    "length",
    "height",
    "yaw",
    "vx",
    "vy",
    "valid",
]
CATEGORIES = {
    "car",
    "vehicle",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "animal",
    "traffic_cone",
    "barrier",
    "other",
}


def read(p):
    return json.loads(p.read_text())


def rows(p, fields):
    with p.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames != fields:
            raise ValueError(str(p) + " columns mismatch")
        return list(r)


def boolean(s):
    if s not in ["True", "False", "true", "false", "0", "1"]:
        raise ValueError("invalid boolean")
    return s in ["True", "true", "1"]


def check(inputs, submission, scene=None, trajectories_only=False):
    if any(p.is_symlink() for p in submission.rglob("*")):
        raise ValueError("Symlinks forbidden")
    meta = read(submission / "submission.json")
    assert meta["protocol"] == "nuscenes_ego_conditioned_v1"
    for k in ["team", "method", "checkpoint", "training_data", "external_data"]:
        assert isinstance(meta[k], str) and meta[k].strip()
    assert type(meta["seed"]) is int
    names = read(inputs / "dataset.json")["scenes"]
    assert len(names) == len(set(names))
    if scene is not None:
        assert scene in names
        names = [scene]
    else:
        assert {p.name for p in (submission / "scenes").iterdir()} == set(names)
    for name in names:
        root = submission / "scenes" / name / "future"
        inp = inputs / "scenes" / name
        task = read(inp / "task.json")
        target = task["target_frame_indices"]
        assert target == list(range(10, 90))
        assert (
            not (root / "ego_trajectory.csv").exists()
            and not (root / "ego_poses.json").exists()
        ), "Ego motion is provided, not a prediction submission"
        ego = rows(
            inp / "conditioning/ego_trajectory.csv",
            ["frame_index", "x", "y", "yaw", "valid"],
        )
        assert [int(r["frame_index"]) for r in ego] == target
        poses = read(inp / "conditioning/ego_poses.json")
        assert (
            poses["coordinate_frame"] == "history_frame_9_ego"
            and poses["frame_indices"] == target
        )
        transforms = np.asarray(poses["ego_to_anchor"], dtype=float)
        assert transforms.shape == (80, 4, 4) and np.isfinite(transforms).all()
        for r, t in zip(ego, transforms):
            assert boolean(r["valid"])
            vals = [float(r[k]) for k in ["x", "y", "yaw"]]
            assert all(map(math.isfinite, vals))
            assert (
                np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
                and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
                and abs(np.linalg.det(t[:3, :3]) - 1) < 1e-4
            )
            assert np.allclose(t[:2, 3], vals[:2], atol=1e-3)
            dy = np.arctan2(t[1, 0], t[0, 0]) - vals[2]
            assert abs(np.arctan2(np.sin(dy), np.cos(dy))) < 1e-3
        hist = list(
            csv.DictReader((inp / "history/agents.csv").read_text().splitlines())
        )
        known = {r["agent_id"] for r in hist}
        knowncat = {r["agent_id"]: r["category"] for r in hist}
        seen = {}
        cats = {}
        for r in rows(root / "future_trajectories.csv", FIELDS):
            f = int(r["frame_index"])
            a = r["agent_id"]
            assert f in target
            assert a in known or re.fullmatch(r"new_[0-9]+", a)
            assert r["category"] in CATEGORIES or (
                a in knowncat and r["category"] == knowncat[a]
            )
            if a in knowncat:
                assert r["category"] == knowncat[a]
            assert a not in cats or cats[a] == r["category"]
            cats[a] = r["category"]
            assert f not in seen.setdefault(a, set())
            seen[a].add(f)
            boolean(r["valid"])
            boolean(r["is_dynamic"])
            assert all(
                math.isfinite(float(r[k]))
                for k in ["x", "y", "z", "width", "length", "height", "yaw", "vx", "vy"]
            )
            assert all(float(r[k]) > 0 for k in ["width", "length", "height"])
        assert known <= set(seen)
        for a, ff in seen.items():
            assert ff == set(target if a in known else range(min(ff), 90))
        if trajectories_only:
            continue
        assert {p.name for p in (root / "rgb").iterdir()} == set(CAMS)
        for cam in CAMS:
            folder = root / "rgb" / cam
            assert {p.name for p in folder.iterdir()} == {
                f"{f:06d}.jpg" for f in target
            }
            for p in folder.iterdir():
                with Image.open(p) as im:
                    im.load()
                    assert (
                        im.format == "JPEG"
                        and im.mode == "RGB"
                        and im.size == tuple(task["image_size"])
                    )
    print(
        "TRAJECTORY_FORMAT_PASS" if trajectories_only else "FORMAT_PASS",
        len(names),
        "scenes; not a metric or validity score",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--scene")
    p.add_argument("--trajectories-only", action="store_true")
    a = p.parse_args()
    check(a.inputs, a.submission, a.scene, a.trajectories_only)

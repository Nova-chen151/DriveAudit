"""Stage only participant conditions from a completed CAT-K run."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def condition_digest(scene):
    """Hash the relative filenames and contents of every published scene condition."""
    identity = hashlib.sha256()
    for file in sorted(scene.rglob("*")):
        if file.is_file():
            identity.update(file.relative_to(scene).as_posix().encode() + b"\0")
            identity.update(bytes.fromhex(digest(file)))
    return identity.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_scene(scene):
    return Path("nuscenes300_challenger_v1", "scenes", scene)


def stage(cfg):
    work = Path(cfg["work_dir"])
    catk = Path(cfg.get("catk_run", work / "traffic"))
    report = read(catk / "COMPLETED.json")
    scenes = cfg["scenes"]
    if (report["status"] != "PASS" or report["future_actor_gt_used"] is not False
            or report["future_rgb_used"] is not False or not set(scenes) <= set(report["input"]["scene_ids"])):
        raise ValueError("CAT-K report does not cover the requested participant scenes")
    archive_hash = digest(cfg["archive"])
    if archive_hash != report["input"]["archive_sha256"]:
        raise ValueError("Archive differs from the completed CAT-K run")
    checks = {row["scene_id"]: row for row in report["renderer_loader_checks"]}
    work.mkdir(parents=True, exist_ok=True)
    final = work / "conditions"
    if final.exists():
        raise FileExistsError(f"Existing staged conditions at {final}. Use a new work directory to restage.")
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=work) as temp:
        root = Path(temp) / "conditions"
        records = []
        refs = []
        for scene in scenes:
            rel = relative_scene(scene)
            source = catk / "render_inputs" / scene / rel
            check = checks[scene]
            if check["status"] != "PASS" or check["public_ego_unchanged"] is not True:
                raise ValueError(f"Failed CAT-K loader verification for {scene}")
            if (source / "future/rgb").exists():
                raise ValueError(f"Unexpected future RGB inside {source}")
            target = root / rel
            shutil.copytree(source, target)
            actor_hash = digest(target / "future/future_trajectories.csv")
            ego_hash = digest(target / "future/ego_trajectory.csv")
            public = catk / "inputs/participant/scenes" / scene / "conditioning/ego_trajectory.csv"
            if actor_hash != check["generated_actor_csv_sha256"] or ego_hash != digest(public):
                raise ValueError(f"CAT-K traffic or public ego trajectory changed for {scene}")
            # Older verified CAT-K bridges predate the explicit local-coordinate declaration.
            write(target / "participant_render_frame.json", {
                "world_frame": "history_frame_9_ego_local_origin",
                "native_rig": "explicit_virtual_nuscenes_height_lidar_rig",
                "sensor2ego_translation_m": [0.985793, 0.0, 1.84019],
                "sensor2ego_rotation_wxyz": [2**-0.5, 0, 0, -2**-0.5],
                "reason": "The participant archive supplies camera-to-ego calibration and a local origin."
            })
            refs.append(rel.as_posix())
            records.append({"scene": scene, "actor_sha256": actor_hash, "public_ego_sha256": ego_hash,
                            "conditions_sha256": condition_digest(target)})
        write(root / "metadata/generation_sets.json", {"participant_joint": refs})
        write(root / "metadata/scene_list.json", refs)
        write(root / "STAGING_COMPLETE.json", {"status": "PASS", "archive_sha256": archive_hash,
              "traffic_run": str(catk), "traffic_report_sha256": digest(catk / "COMPLETED.json"),
              "scenes": records})
        os.rename(root, final)
    return final


def verify_stage(cfg):
    root = Path(cfg["work_dir"]) / "conditions"
    marker = read(root / "STAGING_COMPLETE.json")
    scenes = cfg["scenes"]
    refs = [relative_scene(scene).as_posix() for scene in scenes]
    if (marker["status"] != "PASS" or marker["archive_sha256"] != digest(cfg["archive"])
            or scenes != [row["scene"] for row in marker["scenes"]]
            or refs != read(root / "metadata/scene_list.json")
            or refs != read(root / "metadata/generation_sets.json")["participant_joint"]):
        raise ValueError("Staged input differs from the requested archive or scene selection")
    for ref, row in zip(refs, marker["scenes"]):
        future = root / ref / "future"
        if ((future / "rgb").exists() or digest(future / "future_trajectories.csv") != row["actor_sha256"]
                or digest(future / "ego_trajectory.csv") != row["public_ego_sha256"]
                or ("conditions_sha256" in row and condition_digest(root / ref) != row["conditions_sha256"])):
            raise ValueError(f"Invalid staged conditions for {ref}")
    return root

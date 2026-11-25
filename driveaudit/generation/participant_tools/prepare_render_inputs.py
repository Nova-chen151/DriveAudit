"""Stage-1 predictions -> renderer inputs. Public ego conditions + generated actors; no hidden target root accepted."""

import argparse, json, shutil, hashlib
from pathlib import Path
import numpy as np
from validate_submission import check, read

ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]
CHANNELS = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
    "road_block",
]
SLUGS = {
    "nuscenes": "nuscenes300_challenger_v1",
    "nuplan": "nuplan_challenger_v1",
    "av2": "av2_challenger_v1",
    "pandaset": "pandaset_challenger_v1",
}


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")


def build(inputs, trajectories, output, scene=None):
    for root in [inputs, trajectories]:
        a, b = root.resolve(), output.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("Output must be separate from inputs and predictions")
    if output.exists():
        raise ValueError("Refuse to overwrite output")
    check(inputs, trajectories, scene, trajectories_only=True)
    names = [scene] if scene else read(inputs / "dataset.json")["scenes"]
    output.mkdir(parents=True)
    entries = []
    audit = []
    for name in names:
        source = inputs / "scenes" / name
        pred = trajectories / "scenes" / name / "future"
        task = read(source / "task.json")
        ds = task["source_dataset"]
        rel = Path(SLUGS[ds]) / "scenes" / name
        dst = output / rel
        dst.mkdir(parents=True)
        shutil.copytree(source / "history", dst / "history")
        shutil.copytree(source / "map", dst / "map")
        cameras = read(source / "calibration/cameras.json")
        cameras["camera_order"] = ORDER
        cameras["cameras"] = [
            next(c for c in cameras["cameras"] if c["camera"] == k) for k in ORDER
        ]
        write(dst / "calibration/cameras.json", cameras)
        shutil.copyfile(
            source / "conditioning/camera_poses.json",
            dst / "calibration/history_camera_poses.json",
        )
        past = read(source / "calibration/history_ego_poses.json")["ego_to_anchor"]
        future = read(source / "conditioning/ego_poses.json")["ego_to_anchor"]
        assert len(past) == 10 and len(future) == 80
        write(
            dst / "calibration/reference_ego_poses.json",
            dict(
                coordinate_frame="history_frame_9_ego",
                ego_to_anchor=past + future,
                pose_origin="history + publicly provided reference ego motion",
            ),
        )
        (dst / "future").mkdir()
        shutil.copyfile(
            source / "conditioning/ego_trajectory.csv",
            dst / "future/ego_trajectory.csv",
        )
        shutil.copyfile(
            pred / "future_trajectories.csv", dst / "future/future_trajectories.csv"
        )
        # Existing loader's legacy filename is required; its future values are predictions.
        write(
            dst / "generation_input.json",
            dict(
                reference_frame=9,
                target_frames=[0, 89],
                future_rgb=False,
                conditioning="public_reference_ego_motion_and_generated_actors",
            ),
        )
        write(
            dst / "scene_context.json",
            dict(dataset=ds, generation_prompt=task["prompt"]),
        )
        write(
            dst / "participant_render_frame.json",
            dict(
                world_frame="history_frame_9_ego_local_origin",
                native_rig="explicit_virtual_nuscenes_height_lidar_rig",
                reason="Public camera-to-ego calibration is retained; no physical top-LiDAR rig or global origin is supplied. Use a fixed virtual rig for model tokens and a local origin for CAN bus; do not recover private metadata.",
                sensor2ego_translation_m=[0.985793, 0.0, 1.84019],
                sensor2ego_rotation_wxyz=[2**-0.5, 0, 0, -(2**-0.5)],
            ),
        )
        shutil.copyfile(source / "timeline.json", dst / "timeline.json")
        if not task["map_available"]:
            write(
                dst / "map/map_meta.json",
                dict(
                    channels=CHANNELS,
                    array_axes="channel,x_forward,y_left",
                    coordinate_frame="history_last_ego_x_forward_y_left_meters",
                    available_channels=[],
                    unavailable_channels=CHANNELS,
                    extent_m=[-256, 256, -256, 256],
                    resolution_m=0.5,
                    status="UNAVAILABLE",
                    unknown_policy="All layers unknown; zero values are storage placeholders",
                ),
            )
            np.savez_compressed(
                dst / "map/semantic_map.npz",
                mask=np.zeros((8, 1024, 1024), np.uint8),
                ground_height_m=np.full((1024, 1024), np.nan, np.float32),
            )
        entries.append(rel.as_posix())
        audit.append(
            dict(
                scene=name,
                generated_trajectory_hashes={
                    f: hashlib.sha256((pred / f).read_bytes()).hexdigest()
                    for f in ["future_trajectories.csv"]
                },
            )
        )
    write(output / "metadata/generation_sets.json", {"participant_joint": entries})
    write(
        output / "BRIDGE_COMPLETE.json",
        dict(
            status="PASS",
            scene_count=len(entries),
            future_state_source="provided_reference_ego_motion_and_stage_one_actor_predictions",
            reference_future_rgb_used=False,
            records=audit,
        ),
    )
    print("RENDER_INPUTS_READY", len(entries))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene")
    a = p.parse_args()
    build(a.inputs, a.trajectories, a.output, a.scene)

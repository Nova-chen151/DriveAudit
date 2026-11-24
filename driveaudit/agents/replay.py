"""Run one native ego service and emit scorer-compatible participant predictions."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from .predict import CAMERAS, COMMANDS, ego_state, hash_file, model_input, scene_frames, selected_rgb
from .registry import POLICIES, resources
from .temporal import WORLD_DREAMER_CAMERA_ORDER, build_http_request, parse_output, planning_to_unified_ego


TEMPORAL = {"uniad", "vad"}
FRONT_THREE = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def rpc(endpoint, path, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(endpoint + path, data=body, headers={"Content-Type": "application/json"})
    # Services are local to the allocated node; ignore system-wide proxies.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as reply:
        return reply.read().decode()


@contextmanager
def service(args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    process = subprocess.Popen([args.service_python, "-m", "driveaudit.agents.service",
                                args.policy, "--port", str(port)])
    try:
        deadline = time.monotonic() + args.start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{args.policy} service exited with code {process.returncode}")
            try:
                rpc(endpoint, "/driver-clean/", timeout=3)
                break
            except URLError:
                time.sleep(1)
        else:
            raise TimeoutError(f"{args.policy} service startup timed out")
        yield endpoint
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def initial_state(frame, times, poses):
    if frame >= 2:
        return ego_state(frame, times, poses)
    state = {"timestamp": times[frame], "velocity": [0., 0., 0.], "acceleration": [0., 0., 0.]}
    if frame == 1:
        dt = (times[1]-times[0])/1e6
        dx, dy = (np.asarray(poses[1][:2])-poses[0][:2])/dt
        c, s = math.cos(poses[1][2]), math.sin(poses[1][2])
        state["velocity"] = [float(c*dx+s*dy), float(-s*dx+c*dy), 0.]
    return state


def prepare_request(policy, case, frame, rgb, cameras, times, poses, rig, command):
    from scipy.spatial.transform import Rotation
    channels = WORLD_DREAMER_CAMERA_ORDER if policy in TEMPORAL else (CAMERAS if policy == "drivor" else FRONT_THREE)
    state = initial_state(frame, times, poses)
    visible, image_hashes = model_input(rgb, frame, cameras, state, command, channels=channels)
    if policy in TEMPORAL or policy == "sparsedrivev2":
        for name in channels:
            info = cameras[name]
            matrix = np.asarray(info["camera_to_ego_4x4"])
            xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            # Both legacy adapters expect K on a 1600x900 canvas. They handle
            # resizing to the native policy's actual image canvas internally.
            width, height = info["challenge_image_size"]
            K = np.asarray(info["challenge_intrinsics"], dtype=float)
            K[0] *= 1600/width
            K[1] *= 900/height
            visible["camera_calibration"][name].update(
                sensor2ego_rotation=np.roll(xyzw, 1).tolist(), sensor2ego_translation=matrix[:3, 3].tolist(),
                camera_intrinsics=K.tolist())
    if policy in TEMPORAL:
        x, y, yaw = poses[frame]
        rate = 0.
        if frame:
            delta = (yaw-poses[frame-1][2]+math.pi) % (2*math.pi)-math.pi
            rate = delta*1e6/(times[frame]-times[frame-1])
        state.update(position=[x, y, 0.], rotation=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
                     yaw_rate=rate, lidar2ego_rotation=rig["sensor2ego_rotation_wxyz"],
                     lidar2ego_translation=rig["sensor2ego_translation_m"])
        return build_http_request(visible, case, frame), image_hashes
    return {"model_input": visible}, image_hashes


def result(endpoint, policy, payload):
    rpc(endpoint, "/driver-api/", payload)
    deadline = time.monotonic()+180
    while time.monotonic() < deadline:
        data = rpc(endpoint, "/driver-get/")
        if data:
            points, mode, score, native = parse_output(policy, data)
            if policy in {"goalflow", "diffusiondrivev2"} and (native.get("sensor_mode") != "zero_lidar"
                    or native.get("uses_lidar") is not False or native.get("lidar_feature_nonzero") != 0):
                raise ValueError("Expected verified all-zero LiDAR compatibility input")
            points = planning_to_unified_ego(points) if policy in TEMPORAL else np.asarray(points)
            if points.shape != (6, 2) or not np.isfinite(points).all():
                raise ValueError("Expected a finite six-point ego plan")
            return {"trajectory_xy": points.tolist(), "mode_index": mode,
                    "mode_score": score, "request_seed": native["request_seed"]}
        time.sleep(.1)
    raise TimeoutError("Native service did not return a plan")


def run(args):
    if len(set(args.cases)) != len(args.cases) or any(Path(c).name != c or c in (".", "..") for c in args.cases):
        raise ValueError("Expected unique case directory names")
    config, paths = resources(args.policy)
    asset_hashes = {name: hash_file(path) for name, path in paths.items()}
    config_hash = hash_file(config)
    adapter_hash = hash_file(Path(__file__).parent / "native" / (args.policy+".py"))
    roots = selected_rgb(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with service(args) as endpoint, tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent,
            prefix=args.output.name+".", suffix=".partial", delete=False) as output:
        partial = Path(output.name)
        try:
            for case, rgb in roots.items():
                scene = args.public_scenes / case
                cameras = {c["camera"]: c for c in json.loads((scene/"calibration/cameras.json").read_text())["cameras"]}
                future, times, poses = scene_frames(scene)
                rig = json.loads((scene/"participant_render_frame.json").read_text()) if args.policy in TEMPORAL else None
                rpc(endpoint, "/driver-clean/")
                history_inputs = []
                frames = list(range(future[0])) if args.policy in TEMPORAL else []
                decisions = future[:args.max_decisions] if args.max_decisions else future
                for frame in frames + decisions:
                    history = frame < future[0]
                    selected = scene/"history/rgb" if history else rgb
                    payload, digests = prepare_request(args.policy, case, frame, selected, cameras, times, poses, rig, args.command)
                    plan = result(endpoint, args.policy, payload)
                    if history:
                        history_inputs.append({"frame_index": frame, "rgb_sha256": digests})
                    else:
                        history_bytes = json.dumps(history_inputs, sort_keys=True, separators=(",", ":")).encode()
                        row = dict(history_rgb_sha256=hashlib.sha256(history_bytes).hexdigest(),
                                   sensor_protocol="camera_only", sensor_leakage=False, real_lidar_used=False,
                                   adapter_sha256=adapter_hash,
                                   lidar_mode="zero" if args.policy in {"goalflow", "diffusiondrivev2"} else "not_requested",
                                   case_id=case, source=args.source, policy=args.policy,
                                   frame_index=frame, frame_id=frame-future[0], reference_time_us=times[frame],
                                   coordinate_system="ego_x_forward_y_left", dt_s=.5, rgb_sha256=digests,
                                   command=args.command, checkpoint_sha256=next(iter(asset_hashes.values())),
                                   asset_sha256=asset_hashes, config_sha256=config_hash,
                                   ego_state=initial_state(frame, times, poses), **plan)
                        if args.policy == "drivor":
                            row["backbone_sha256"] = asset_hashes["weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"]
                        output.write(json.dumps(row, allow_nan=False)+"\n")
                print(f"{case}: {len(decisions)} decisions", flush=True)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    os.replace(partial, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--service-python", default=sys.executable)
    parser.add_argument("--start-timeout", type=int, default=600)
    parser.add_argument("--public-scenes", type=Path, required=True)
    parser.add_argument("--rgb-scenes", type=Path)
    parser.add_argument("--real-mapping", type=Path)
    parser.add_argument("--max-decisions", type=int, help="Optional diagnostic prefix for GPU smoke tests")
    parser.add_argument("--source", choices=["generated", "real"], required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", choices=tuple(COMMANDS), default="straight")
    args = parser.parse_args()
    if args.max_decisions is not None and args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    run(args)


if __name__ == "__main__":
    main()

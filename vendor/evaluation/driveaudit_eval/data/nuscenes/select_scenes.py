from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from .build_camera_timeline import build as build_timeline
from .schema import CAMERAS, write_csv, write_json


def _tags(scene: dict) -> list[str]:
    text = scene.get("description", "").lower(); tags = []
    for key, words in {"night": ("night",), "rain": ("rain", "wet"), "intersection": ("intersection",),
                       "dense_traffic": ("traffic", "busy"), "turning": ("turn",)}.items():
        if any(word in text for word in words): tags.append(key)
    return tags or ["nominal"]


def _dynamic_count(nusc, scene: dict) -> int:
    tokens = set(); sample = nusc.get("sample", scene["first_sample_token"])
    while sample:
        for token in sample["anns"]:
            ann = nusc.get("sample_annotation", token)
            if ann["category_name"].startswith(("vehicle.", "human.pedestrian")):
                tokens.add(ann["instance_token"])
        sample = nusc.get("sample", sample["next"]) if sample["next"] else None
    return len(tokens)


def _affordances(samples: list[dict]) -> list[str]:
    values = []
    if any(0 <= row["x"] <= 40 and abs(row["y"]) <= 4.5 for row in samples): values.append("front_follow_or_brake")
    if any(-15 <= row["x"] <= 40 and 2.5 <= abs(row["y"]) <= 10 for row in samples): values.append("adjacent_lane_cut_in")
    if any(-15 <= row["x"] <= 5 and 1.5 <= abs(row["y"]) <= 7 for row in samples): values.append("blind_spot_or_overtake")
    if samples and max(row["y"] for row in samples) - min(row["y"] for row in samples) >= 6: values.append("crossing_or_lane_change")
    return values or ["proximity_interaction"]


def _adversarial_profile(nusc, scene: dict, timeline: list[dict], config: dict) -> dict:
    settings = config.get("adversarial_selection", {})
    square_half = float(settings.get("ego_square_half_extent_m", 50.0))
    max_distance = float(settings.get("max_interaction_distance_m", 35.0))
    min_keyframes = int(settings.get("min_candidate_keyframes", 4))
    min_visibility = int(settings.get("min_visibility_token", 2))
    tracks = defaultdict(list); start = timeline[0]["reference_timestamp_us"]; end = timeline[-1]["reference_timestamp_us"]
    sample = nusc.get("sample", scene["first_sample_token"])
    while sample:
        if start <= sample["timestamp"] <= end:
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"]); ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            rotation = Quaternion(ego["rotation"]).inverse; origin = np.asarray(ego["translation"], dtype=float)
            for token in sample["anns"]:
                ann = nusc.get("sample_annotation", token)
                if not ann["category_name"].startswith("vehicle."): continue
                relative = rotation.rotate(np.asarray(ann["translation"], dtype=float) - origin); x, y = map(float, relative[:2])
                if abs(x) > square_half or abs(y) > square_half: continue
                tracks[ann["instance_token"]].append({"x": x, "y": y, "distance": math.hypot(x, y),
                    "visibility": int(ann["visibility_token"]), "sensor_points": int(ann["num_lidar_pts"] + ann["num_radar_pts"]),
                    "category": ann["category_name"]})
        sample = nusc.get("sample", sample["next"]) if sample["next"] else None
    candidates = []
    for instance_token, values in tracks.items():
        if len(values) < min_keyframes or min(row["distance"] for row in values) > max_distance: continue
        if max(row["visibility"] for row in values) < min_visibility or max(row["sensor_points"] for row in values) <= 0: continue
        candidates.append({"instance_token": instance_token, "category": values[0]["category"], "keyframe_count": len(values),
            "minimum_ego_distance_m": min(row["distance"] for row in values), "maximum_visibility_token": max(row["visibility"] for row in values),
            "affordances": _affordances(values)})
    candidates.sort(key=lambda row: (-len(row["affordances"]), row["minimum_ego_distance_m"], -row["keyframe_count"], row["instance_token"]))
    return {"adversarial_candidate_count": len(candidates), "minimum_candidate_distance_m": min((x["minimum_ego_distance_m"] for x in candidates), default=None),
        "candidate_affordance_counts": dict(Counter(a for row in candidates for a in row["affordances"])), "adversarial_candidates": candidates}


def select(nusc, config: dict, root: Path, limit: int | None = None) -> dict:
    requested = limit or int(config["dataset"]["scene_count"])
    official = create_splits_scenes(); by_name = {scene["name"]: scene for scene in nusc.scene}
    accepted, rejected = [], []
    for split in config["dataset"]["official_split_priority"]:
        if split == "test": raise ValueError("nuScenes test split is forbidden for GT conversion")
        for name in official[split]:
            scene = by_name.get(name)
            if not scene: continue
            timeline, quality = build_timeline(nusc, scene, config["challenge"]["total_frames"], int(config["timeline"]["max_camera_offset_ms"] * 1000))
            if timeline is None:
                rejected.append({"scene_name": name, "scene_token": scene["token"], "official_split": split, "reason": quality.get("reason", "insufficient_camera_frames")}); continue
            log = nusc.get("log", scene["log_token"])
            dynamic = _dynamic_count(nusc, scene)
            if not dynamic:
                rejected.append({"scene_name": name, "scene_token": scene["token"], "official_split": split, "reason": "no_dynamic_agent"}); continue
            profile = _adversarial_profile(nusc, scene, timeline, config)
            if config.get("adversarial_selection", {}).get("enabled") and not profile["adversarial_candidate_count"]:
                rejected.append({"scene_name": name, "scene_token": scene["token"], "official_split": split, "reason": "no_challenger_eligible_vehicle"}); continue
            accepted.append({"scene_token": scene["token"], "scene_name": name, "log_token": scene["log_token"], "location": log["location"],
                             "official_split": split, "description": scene["description"], "heuristic_scene_tag": _tags(scene),
                             "first_timestamp": timeline[0]["reference_timestamp_us"], "last_timestamp": timeline[-1]["reference_timestamp_us"],
                             "dynamic_agent_count": dynamic, "camera_frame_counts": {c: 90 for c in CAMERAS},
                             "selection_window_start": quality["start_index"], "selection_score": quality,
                             "source_file_hash_digest": hashlib.sha256("".join(frame["cameras"]["CAM_FRONT"]["sample_data_token"] for frame in timeline).encode()).hexdigest(),
                             **profile, "timeline": timeline})
        if len(accepted) >= requested: break
    # Deterministic split-priority and location-balanced ordering.  Val is exhausted
    # before train when more scenes than the official val split are requested.
    seed = int(config["dataset"]["seed"]); selected = []
    counts = Counter()
    for split in config["dataset"]["official_split_priority"]:
        pool = [row for row in accepted if row["official_split"] == split]
        while pool and len(selected) < requested:
            pool.sort(key=lambda row: (counts[row["location"]], -row["adversarial_candidate_count"],
                row["minimum_candidate_distance_m"] if row["minimum_candidate_distance_m"] is not None else float("inf"),
                row["selection_score"]["maximum_offset_us"], hashlib.sha256(f"{seed}:{row['scene_token']}".encode()).hexdigest()))
            row = pool.pop(0); selected.append(row); counts[row["location"]] += 1
        if len(selected) >= requested: break
    if len(selected) != requested:
        raise RuntimeError(f"requested {requested} valid scenes, found {len(selected)}")
    sizes = config.get("splits", {}); calibration = int(sizes.get("calibration", min(60, requested))); validation = int(sizes.get("validation", min(20, max(0, requested - calibration))))
    if calibration + validation > requested: raise ValueError("calibration + validation split exceeds requested scene count")
    split_names = ["calibration"] * calibration + ["validation"] * validation + ["test"] * (requested - calibration - validation)
    for index, row in enumerate(selected):
        row["challenge_split"] = split_names[index]
        scene_root = root / "scenes" / row["scene_name"]
        write_json(scene_root / "timeline.json", {"scene_id": row["scene_name"], "camera_order": CAMERAS, "frames": row.pop("timeline")})
    compact = [{k: v for k, v in row.items() if k != "timeline"} for row in selected]
    result = {"dataset": config["dataset"]["name"], "official_split_priority": config["dataset"]["official_split_priority"],
              "seed": seed, "requested_scene_count": requested, "selected_scene_count": len(compact),
              "split_counts": dict(Counter(row["challenge_split"] for row in compact)), "location_counts": dict(counts), "scenes": compact}
    write_json(root / "scene_selection.json", result); write_csv(root / "scene_selection.csv", compact); write_csv(root / "rejected_candidates.csv", rejected)
    return result

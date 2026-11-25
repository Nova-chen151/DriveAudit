"""Score a camera policy on paired generated and recorded observations."""
import csv
import hashlib
import json
import re
from pathlib import Path

from . import EVALUATION_ROOT, enable_engine_imports
from statistics import mean

import numpy as np
import yaml

enable_engine_imports()

from driveaudit_eval.agents.core import ego_to_world
from driveaudit_eval.agents.input_gt import load_scene, sample, score_window
from driveaudit_eval.agents.scoring import overlap
from driveaudit_eval.episode_rollout import evaluate_terminal_responsibility, run_episode
from driveaudit_eval.metrics.core import compute_rpdms
from driveaudit_eval.metrics.population_testing import local_retention
from driveaudit_eval.metrics.testing_effectiveness import content_hash, testing_effectiveness, unit
from driveaudit_eval.responsibility.attribution import classify
from driveaudit_eval.responsibility.cars_adapter import evaluate_collision_responsibility
from driveaudit_eval.responsibility.evidence import evaluate_event
from driveaudit_eval.responsibility.ratios import raw_evidence_tracks
from driveaudit_eval.traffic_compliance import load_map_context as load_traffic_map
from driveaudit_eval.traffic_evaluator import ego_track
from driveaudit_eval.trajectory.input_gt import load_scene as load_participants
from driveaudit_eval.utils.io import write_json
from driveaudit_eval.utils.map_provider.canonical_map import load_map_context
from driveaudit_eval.utils.map_provider.geometry import trajectory_headings

OFFSETS = np.arange(1, 7) * 0.5
IDENTITY = ("policy", "config_sha256", "checkpoint_sha256")
DIGEST = re.compile(r"[0-9a-f]{64}")
DRIVOR_CAMERAS = {"CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_identity(row):
    identity = {k: row[k] for k in IDENTITY}
    if not identity["policy"] or any(not DIGEST.fullmatch(row[k]) for k in IDENTITY[1:]):
        raise ValueError("Missing policy, checkpoint or config provenance")
    if "asset_sha256" in row:
        assets = row["asset_sha256"]
        if not assets or any(not name or not DIGEST.fullmatch(h) for name, h in assets.items()):
            raise ValueError("Invalid policy asset digest mapping")
        identity["asset_sha256"] = assets
    if "backbone_sha256" in row:
        if not DIGEST.fullmatch(row["backbone_sha256"]):
            raise ValueError("Invalid visual backbone digest")
        identity["backbone_sha256"] = row["backbone_sha256"]
    if "asset_sha256" not in identity and "backbone_sha256" not in identity:
        raise ValueError("Missing inference asset provenance")
    return identity


def read_records(path, side):
    by_scene = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case, index = row["case_id"], row["frame_index"]
        rows = by_scene.setdefault(case, {})
        if row["source"] != side or index in rows or type(index) is not int:
            raise ValueError(f"{path}:{number}: duplicate frame or source mismatch")
        if row["coordinate_system"] != "ego_x_forward_y_left" or row["dt_s"] != 0.5:
            raise ValueError("Expected decision-relative +x forward, +y left, 0.5 s steps")
        xy = np.asarray(row["trajectory_xy"], dtype=float)
        if xy.shape != (6, 2) or not np.isfinite(xy).all():
            raise ValueError("Expected six finite decision-relative XY positions")
        policy_identity(row)
        images = row["rgb_sha256"]
        if not images or any(not DIGEST.fullmatch(h) for h in images.values()):
            raise ValueError("Missing sampled RGB input hashes")
        front = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}
        channels = {"drivor": DRIVOR_CAMERAS, "uniad": front | {"CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"},
                    "vad": front | {"CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"},
                    **{name: front for name in ("sparsedrivev2", "diffusiondrivev2", "goalflow")}}
        if row["policy"] in channels and set(images) != channels[row["policy"]]:
            raise ValueError("Wrong policy camera set")
        rows[index] = row
    return by_scene


def validate_pair(case, paths, pairs):
    left, right = pairs["generated"], pairs["real"]
    timeline = json.loads((paths["real"] / "timeline.json").read_text())
    ref = int(json.loads((paths["real"] / "generation_input.json").read_text())["reference_frame"])
    expected = {r["frame_index"]: r["reference_time_us"] for r in timeline["frames"] if r["frame_index"] > ref}
    if not expected or set(left) != set(expected) or set(right) != set(expected):
        raise ValueError(f"{case}: each branch must cover all future observations")
    if any(path.name != case for path in paths.values()):
        raise ValueError(f"{case}: scene directory mismatch")
    for f in ("timeline.json", "map/map_meta.json", "map/semantic_map.npz",
              "calibration/reference_ego_poses.json", "history/ego.csv", "future/ego_trajectory.csv"):
        if digest(paths["real"] / f) != digest(paths["generated"] / f):
            raise ValueError(f"{case}: mismatched paired timeline, route or map: {f}")
    if len({content_hash(policy_identity(row)) for rows in pairs.values() for row in rows.values()}) != 1:
        raise ValueError(f"{case}: mixed policy or weight/config provenance")
    for idx, stamp in expected.items():
        a, b = left[idx], right[idx]
        if a["request_seed"] != b["request_seed"] or a["rgb_sha256"].keys() != b["rgb_sha256"].keys():
            raise ValueError(f"{case}/{idx}: unmatched policy seed or camera channels")
        for side, row in (("generated", a), ("real", b)):
            for camera, observed_hash in row["rgb_sha256"].items():
                if Path(camera).name != camera or camera in (".", ".."):
                    raise ValueError("Invalid sensor name")
                image = paths[side]/"future/rgb"/camera/f"{idx:06d}.jpg"
                if digest(image) != observed_hash:
                    raise ValueError(f"{case}/{side}/{idx}/{camera}: sampled image hash mismatch")
            if row["reference_time_us"] != stamp or row["frame_id"] != idx-ref-1:
                raise ValueError(f"{case}/{idx}: decision timestamp/index mismatch")
        if a.get("ego_state") != b.get("ego_state") or a.get("command") != b.get("command"):
            raise ValueError(f"{case}/{idx}: non-visual policy inputs differ")


def window_attribution(scene, window, actors, timestamps, times, poses, lookup, traffic_map, cfg):
    """Replay each contiguous overlap with the original CARS, V2 and V3 modules."""
    start = window["decision_time"]
    idx = window["decision_frame_id"]
    knots = window["timestamps"]
    values = np.column_stack([window["positions"], window["headings"]])
    past = times <= start
    pred = ego_track(scene, np.r_[times[past], knots[1:]], np.vstack([poses[past], values[1:]]),
                     int(past.sum())-1, "generated")
    pred.metadata.update(replay_max_gap_s=.5, history_policy="logged past plus one decision's plan")
    tracks = dict(lookup, ego=pred)
    traffic_cfg = dict(cfg["traffic_compliance"], max_gap_s=.5)
    output = []
    for ident, actor in actors.items():
        other, valid = sample(actor[:, 0], actor[:, 1:], knots[1:])
        hits = np.flatnonzero(valid & overlap(values[1:, :2], values[1:, 2],
            other[:, :2], other[:, 2], other[:, 3], other[:, 4]))
        if not len(hits):
            continue
        for group in np.split(hits, np.flatnonzero(np.diff(hits) > 1)+1):
            hit_time = float(knots[1+group[0]])
            event = dict(scene_id=scene, collision_event_id=f"{scene}/{idx}/{ident}/{group[0]}",
                         decision_frame_id=idx, collision_pair=["ego", ident], first_collision_time=hit_time,
                         first_collision_frame=min(timestamps, key=lambda f: abs(timestamps[f]-hit_time)),
                         collision_sample_steps=group.tolist(), collision_duration=len(group)*.5)
            cars = evaluate_collision_responsibility([event], tracks, traffic_map,
                dict(cfg["collision_responsibility"], traffic_config=traffic_cfg), {"root": EVALUATION_ROOT})["events"][0]
            evidence = evaluate_event(cars, tracks, traffic_map, cfg["responsibility_v2"], cars_event=cars,
                traffic_config=traffic_cfg, kinematic_thresholds=cfg["evaluation"]["trajectory"].get("kinematics", {}))
            final = classify(evidence, cfg["collision_responsibility_v3"])
            output.append(dict(event=event, attribution=final))
    if not output:
        raise ValueError("Frozen collision flag disagrees with responsibility event reconstruction")
    return output


def evaluate_branch(case, path, records):
    cfg = yaml.safe_load((EVALUATION_ROOT / "driveaudit_eval/defaults.yaml").read_text())
    timestamps, times, poses, actors, _ = load_scene(path)
    raster = load_map_context(path)
    traffic_map = load_traffic_map(path)
    windows, rows, tail = [], {}, []
    tracks, provenance = load_participants(path)
    lookup = raw_evidence_tracks(case, actors, provenance)
    lookup.update({track.agent_id: track for track in tracks})
    for idx, record in sorted(records.items()):
        start = timestamps[idx]
        if start + OFFSETS[-1] > times[-1] + 1e-8:
            tail.append(idx)
            continue
        plan = np.asarray(record["trajectory_xy"])
        row = score_window(start, times, poses, actors, None, {"dac_protocol": "observed_common_v1"},
                           prediction=plan, canonical_map=raster)
        rows[idx] = row
        if not row["valid_trajectory"]:
            continue
        origin, ok = sample(times, poses, [start])
        if not ok.all():
            raise ValueError("Logged decision anchor unavailable")
        pos = np.vstack([origin[0, :2], ego_to_world(plan, [*origin[0, :2], 0], origin[0, 2])])
        yaw = np.r_[origin[0, 2], trajectory_headings(pos, origin[0, 2])[1:]]
        first = row["first_collision_time"]
        w = dict(decision_frame_id=idx, decision_time=start, timestamps=np.r_[start, start+OFFSETS],
                 positions=pos, headings=yaw, open_loop_collision=first is not None,
                 open_loop_first_collision_time=None if first is None else start+first)
        windows.append(w)
        events = []
        if first is not None:
            events = window_attribution(case, w, actors, timestamps, times, poses, lookup, traffic_map, cfg)
        ratios = [event["attribution"].get("ego_responsibility_ratio") for event in events]
        ratio = max(ratios) if ratios and all(r is not None for r in ratios) else None
        row.update(compute_rpdms(row, ratio, require_responsibility=True))
        row["events"] = events
    if not windows:
        raise ValueError(f"{case}: no complete three-second decision windows")
    policy = next(iter(records.values()))["policy"]
    def terminal_attribution(event, window, executed_history):
        return evaluate_terminal_responsibility(event, window, executed_history, case, times, poses,
            tracks, lookup, traffic_map, cfg, {"root": EVALUATION_ROOT}, timestamps)

    options = dict(cfg["episode_rollout"], traffic_config=cfg["traffic_compliance"])
    replay = run_episode(case, policy, windows, actors, poses[:, :2], options,
                         map_context=traffic_map, terminal_callback=terminal_attribution, scene_end_time=times[-1],
                         route_provenance={"source": "logged_ego_path_proxy"})
    return dict(rows=rows, episode=replay, unscored_tail=tail,
                source_sha256={f: digest(path / f) for f in ("timeline.json", "history/agents.csv",
                    "future/future_trajectories.csv", "map/semantic_map.npz", "future/ego_trajectory.csv")})


def paired_summary(case, outputs):
    left, right = (outputs[k]["rows"] for k in ("generated", "real"))
    if set(left) != set(right):
        raise ValueError("Candidate and reference decision supports do not match")
    common = sorted(k for k in left if left[k].get("rpdms") is not None and right[k].get("rpdms") is not None)
    row = dict(case_id=case, legal_windows=len(left), paired_windows=len(common),
               common_window_coverage=len(common)/len(left), common_frame_indices=common)
    for side, data in outputs.items():
        selected = [data["rows"][idx] for idx in common]
        ep = data["episode"]["episode"]
        row[side] = dict(rpdms=mean(v["rpdms"] for v in selected) if selected else None,
                         route_completion=ep["route_completion_ratio"] if ep["route_completion_status"] == "valid" else None,
                         route_status=ep["route_completion_status"], route_projection_coverage=ep["route_projection_coverage"],
                         termination=ep["termination_reason"], unscored_tail=len(data["unscored_tail"]))
    enough = bool(common) and row["common_window_coverage"] >= .8
    row["single_policy_TE"] = testing_effectiveness(row["real"]["rpdms"], row["generated"]["rpdms"]) if enough else None
    return row


def summarize_population(rows):
    """Require complete paired support before reporting a scene-equal mean."""
    means = {"scene_count": len(rows)}
    for metric in ("rpdms", "route_completion"):
        selected = [row for row in rows if all(row[side][metric] is not None for side in ("generated", "real"))]
        means[metric] = {"paired_scenes": len(selected)}
        for side in ("generated", "real"):
            means[metric][side] = mean(row[side][metric] for row in selected) if len(selected) == len(rows) else None
    values = [row["single_policy_TE"] for row in rows]
    means["single_policy_TE"] = mean(values) if all(v is not None for v in values) else None
    return means


def merge_quality(report, quality_file, trs_file, manifests):
    scenes = report["scenes"]
    pop = json.loads(trs_file.read_text()) if trs_file else None
    trs = None
    if pop is not None:
        source = pop["source_manifest"]
        expected = sorted(path["real"].parents[1].name+"/"+case for case, path in manifests.items())
        if sorted(source) != expected or pop["population_hash"] != content_hash(expected):
            raise ValueError("TRS provenance or population mismatch")
        for case, paths in manifests.items():
            sid = paths["real"].parents[1].name+"/"+case
            for side, name in (("real", "reference"), ("generated", "candidate")):
                bound = source[sid][name]
                if not bound or any(digest(paths[side]/k) != v for k, v in bound.items()):
                    raise ValueError(f"{sid}/{side}: TRS input hashes mismatch")
        if pop.get("status", "").lower() in ("available", "valid", "completed", "complete"):
            trs = unit(pop["TRS"], "TRS")
    report["TRS_population"] = trs
    if quality_file:
        quality = json.loads(quality_file.read_text())
        if set(quality) != set(manifests):
            raise ValueError("Quality scene selection mismatch")
        for row in scenes:
            pair = quality[row["case_id"]]
            for side in ("generated", "real"):
                identity = pair[side]["scene_id"].replace("__", "/").split("/")[-1]
                if identity != row["case_id"]:
                    raise ValueError("Quality/decision scene mismatch")
                row[side]["quality_raw"] = pair[side]["scores"]
            try:
                result = local_retention(pair["generated"], pair["real"])
            except (ValueError, KeyError) as exc:
                result = {"B": None, "invalid_reason": [str(exc)]}
            row["quality_retention"] = result
    for row in scenes:
        b = row.get("quality_retention", {}).get("B")
        te = row["single_policy_TE"]
        row["single_policy_contribution"] = te*b if b is not None and te is not None else None
    scores = [row["single_policy_contribution"] for row in scenes]
    reasons = []
    if trs is None:
        reasons.append("population_TRS_missing_or_unsupported")
    if not quality_file:
        reasons.append("paired_local_quality_not_supplied")
    for row in scenes:
        if row["single_policy_TE"] is None:
            reasons.append(row["case_id"]+":insufficient_paired_policy_support")
        for reason in row.get("quality_retention", {}).get("invalid_reason", []):
            reasons.append(row["case_id"]+":"+reason)
    report["single_policy_QTES_diagnostic"] = (100*trs**.25*mean(scores)
        if not reasons and all(v is not None for v in scores) else None)
    report["single_policy_QTES_status"] = "available" if report["single_policy_QTES_diagnostic"] is not None else "unavailable"
    report["single_policy_QTES_reasons"] = reasons


def run(args):
    manifests = {case: {s: Path(r[s]) for s in ("real", "generated")}
                 for case, r in json.loads(args.scenes.read_text()).items()}
    captures = {s: read_records(getattr(args, s), s) for s in ("generated", "real")}
    if not manifests or any(set(branch) != set(manifests) for branch in captures.values()):
        raise ValueError("Prediction and scene selections differ")
    report = dict(protocol="paired_logged_replay_one_policy", official_QTES=None,
                  official_status="requires_complete_formal_policy_population",
                  observation="generated RGB + generated traffic versus recorded RGB + recorded traffic",
                  implementation_sha256={"adapter": digest(Path(__file__)),
                      **{k: digest(EVALUATION_ROOT/"driveaudit_eval"/k) for k in ("defaults.yaml", "agents/input_gt.py",
                          "episode_rollout.py", "route_projection.py", "metrics/core.py",
                          "metrics/population_testing.py", "metrics/testing_effectiveness.py")}},
                  hashes={k: digest(getattr(args, k)) for k in ("real", "generated", "scenes", "quality", "trs")
                          if getattr(args, k, None)}, scenes=[])
    identities = set()
    for case, paths in manifests.items():
        pairs = {s: captures[s][case] for s in captures}
        validate_pair(case, paths, pairs)
        first = next(iter(pairs["real"].values()))
        identities.add(content_hash(policy_identity(first)))
        if len(identities) > 1:
            raise ValueError("Different policy identities across scenes")
        outputs = {}
        for side in ("generated", "real"):
            data = evaluate_branch(case, paths[side], pairs[side])
            write_json(args.output/case/(side+".json"), data)
            outputs[side] = data
        report["scenes"].append(paired_summary(case, outputs))
    report["policy_identity"] = policy_identity(first)
    report["scene_equal_means"] = summarize_population(report["scenes"])
    merge_quality(report, args.quality, args.trs, manifests)
    write_json(args.output/"summary.json", report)
    table = []
    for row in report["scenes"]:
        for side in ("real", "generated"):
            metrics = dict(row[side])
            raw = metrics.pop("quality_raw", {})
            table.append(dict(case_id=row["case_id"], side=side, **metrics, **raw,
                paired_windows=row["paired_windows"], legal_windows=row["legal_windows"],
                single_policy_TE=row["single_policy_TE"]))
    fields = list(dict.fromkeys(k for row in table for k in row))
    with (args.output/"paired.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    return report

"""Evaluate the complete frozen six-policy cohort on a paired submission."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from .score import content_hash, digest, policy_identity, read_records, run, write_json, local_retention
from driveaudit_eval.agents.registry import FORMAL_AGENTS
from driveaudit_eval.metrics.population_testing import aggregate
from driveaudit_eval.metrics.testing_effectiveness import paired_comparison
from driveaudit_eval.testing import scene_evidence

POLICIES = {name: "sparsedrivev2" if name == "sparsedrivev2_v2" else name for name in FORMAL_AGENTS}
SIDES = ("real", "generated")


def measured_history(scene, row):
    frames = []
    if row["policy"] in ("uniad", "vad"):
        ref = json.loads((scene/"generation_input.json").read_text())["reference_frame"]
        for frame in range(ref+1):
            images = {name: digest(scene/"history/rgb"/name/f"{frame:06d}.jpg") for name in row["rgb_sha256"]}
            frames.append(dict(frame_index=frame, rgb_sha256=images))
    return content_hash(frames)


def require_receipts(policy, scene, rows):
    first = next(iter(rows.values()))
    mode = "zero" if policy in ("goalflow", "diffusiondrivev2") else "not_requested"
    expected = dict(sensor_protocol="camera_only", real_lidar_used=False, sensor_leakage=False,
                    history_rgb_sha256=measured_history(scene, first), lidar_mode=mode, policy=policy)
    adapter_file = Path(__file__).resolve().parents[1]/"agents/native"/(policy+".py")
    expected["adapter_sha256"] = digest(adapter_file)
    for row in rows.values():
        if any(k not in row or type(row[k]) is not type(v) or row[k] != v for k, v in expected.items()):
            raise ValueError(f"{scene.name}/{policy}: camera protocol, adapter or history receipt mismatch")
        if not row.get("ego_state") or not row.get("command") or type(row.get("request_seed")) is not int:
            raise ValueError(f"{scene.name}/{policy}: missing nonvisual input/seed receipt")
    return expected


def measure_one(args):
    run(args)
    return str(args.output)


def common_frames(windows):
    """Freeze one support mask across both branches of all six policies."""
    all_rows = [scores for sides in windows.values() for scores in sides.values()]
    if len(all_rows) != 12:
        raise ValueError("Expected six complete policy pairs")
    expected = set(all_rows[0])
    if not expected or any(set(rows) != expected for rows in all_rows):
        raise ValueError("Policy branches do not cover the same native decision set")
    common = sorted(i for i in expected if all(row[i].get("rpdms") is not None for row in all_rows))
    if len(common) < 10 or len(common)/len(expected) < .8:
        raise ValueError(f"Insufficient common native windows: {len(common)}/{len(expected)}")
    return common, len(expected)


def collect_panel(paths, policy_dirs, quality, population, output, input_hashes):
    refs, candidates, locals_, evidence_index, support, table = [], [], [], {}, {}, []
    for case, sidepaths in paths.items():
        scene = sidepaths["real"]
        sid = scene.parent.parent.name+"/"+case
        branches, raw = {}, {}
        pair = quality[case]
        local = local_retention(pair["generated"], pair["real"])
        if local["scene_id"] != sid:
            raise ValueError("Quality population ID mismatch")
        evidence_index[sid] = {}
        for side in SIDES:
            label = "reference" if side == "real" else "generated"
            src = "reference" if side == "real" else "candidate"
            evidence = scene_evidence({"scene_path": str(sidepaths[side]), "frames": []})
            evidence_index[sid][label] = evidence
            for key in ("scene_hash", "rgb_hash", "history_hash"):
                if pair[side].get(key) != evidence[key]:
                    raise ValueError(f"{sid}/{side}: quality/policy {key} mismatch")
            bound = population["source_manifest"][sid][src]
            if pair[side].get("trajectory_inputs") != bound or any(digest(sidepaths[side]/k) != v for k,v in bound.items()):
                raise ValueError(f"{sid}/{side}: quality/traffic/trajectory identity mismatch")
            local["reference_trajectory_inputs" if side == "real" else "trajectory_inputs"] = bound
            local.setdefault("evidence_identity", {})[label] = {k: pair[side][k] for k in ("scene_hash", "rgb_hash")}
        locals_.append(local)
        for agent, policy in POLICIES.items():
            raw[agent] = {}
            branches[agent] = {}
            for side in SIDES:
                raw[agent][side] = read_records(policy_dirs[policy]/(side+".jsonl"), side)[case]
                require_receipts(policy, sidepaths[side], raw[agent][side])
                data = json.loads((output/policy/case/(side+".json")).read_text())
                branches[agent][side] = {int(f): v for f,v in data["rows"].items()}
        common, total = common_frames(branches)
        support[sid] = dict(indices=common, paired_windows=len(common), legal_windows=total)
        for agent, policy in POLICIES.items():
            report = json.loads((output/policy/"summary.json").read_text())
            single = next(r for r in report["scenes"] if r["case_id"] == case)
            for side, dest in (("generated", candidates), ("real", refs)):
                capture = raw[agent][side]
                first = capture[common[0]]
                evidence = evidence_index[sid]["reference" if side == "real" else "generated"]
                if report["hashes"][side] != digest(policy_dirs[policy]/(side+".jsonl")):
                    raise ValueError("Policy evidence changed during scoring")
                identity = policy_identity(first)
                rpdms = mean(branches[agent][side][f]["rpdms"] for f in common)
                table.append(dict(scene=case, agent=agent, side=side, rpdms=rpdms,
                                  common_windows=len(common), legal_windows=total,
                                  route_completion=single[side]["route_completion"], route_status=single[side]["route_status"]))
                record = dict(dataset=local["dataset"], scene=sid, agent=agent,
                    checkpoint_hash=content_hash(identity["asset_sha256"]),
                    agent_config_hash=content_hash([identity["config_sha256"],first["adapter_sha256"]]),
                    camera_only_protocol_hash=content_hash([sorted(first["rgb_sha256"]),first["lidar_mode"],first["adapter_sha256"]]),
                    scene_hash=evidence_index[sid]["reference"]["scene_hash"], scoring_hash=content_hash(report["implementation_sha256"]),
                    conditioning_hash=content_hash([evidence["history_hash"],
                        [(capture[f]["ego_state"],capture[f]["command"]) for f in common]]),
                    decision_frames=common, decision_timestamps=[capture[f]["reference_time_us"] for f in common],
                    request_seeds=[capture[f]["request_seed"] for f in common],
                    required_rgb_complete=evidence["required_rgb_complete"], six_camera_valid=evidence["six_camera_valid"],
                    timestamps_valid=evidence["timestamps_valid"], inference_valid=True, rpdms=rpdms,
                    **{k: first[k] for k in ("sensor_protocol","real_lidar_used","sensor_leakage","lidar_mode")})
                dest.append(record)
    comparison = paired_comparison(refs, candidates, [(row["dataset"],row["scene_id"]) for row in locals_])
    comparison["input_identity"] = evidence_index
    qualities = dict(scenes=locals_, invalid_reason=[r["scene_id"]+":"+reason for r in locals_ for reason in r["invalid_reason"]])
    final = aggregate(population, comparison, qualities, list(evidence_index))
    by_pair = {(r["scene"], r["agent"], r["side"]): r for r in table}
    paired_table = []
    for measured in comparison["rows"]:
        case = measured["scene"].rsplit("/", 1)[-1]
        agent = measured["agent"]
        ref, gen = (by_pair[case, agent, side] for side in SIDES)
        paired_table.append(dict(case_id=case, policy=agent,
            common_windows=ref["common_windows"], legal_windows=ref["legal_windows"],
            rpdms_real=ref["rpdms"], rpdms_generated=gen["rpdms"],
            route_real=ref["route_completion"], route_generated=gen["route_completion"],
            route_real_status=ref["route_status"], route_generated_status=gen["route_status"],
            TE=measured["testing_effectiveness"]))
    write_json(output/"panel.json", dict(result=final, input_sha256=input_hashes,
               panel_code_sha256=digest(Path(__file__)), paired_windows=support, comparison=comparison,
               local_quality=qualities, TRS=population["TRS"], policy_scores=paired_table,
               execution="logged_state_prefix_with_terminal_commit", population="selected_scenes"))
    with (output/"panel.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(paired_table[0]))
        writer.writeheader()
        writer.writerows(paired_table)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("scenes","quality","trs","ego-root","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--capture-dir", action="append", default=[], metavar="POLICY=PATH",
                        help="Override one policy capture directory")
    parser.add_argument("--workers", type=int, default=1, choices=range(1,7), help="Parallel CPU policy scorers")
    args=parser.parse_args()
    directory={name: args.ego_root/name for name in POLICIES.values()}
    for value in args.capture_dir:
        name,sep,path=value.partition("=")
        if name not in directory or not sep or not path:
            parser.error("Unknown policy or invalid directory override")
        directory[name]=Path(path)
    for policy,path in directory.items():
        for side in SIDES:
            if not (path/(side+".jsonl")).is_file():
                parser.error(f"Missing {policy}/{side} capture at {path}")
    inputs = {"scenes": args.scenes, "quality": args.quality, "TRS": args.trs}
    inputs.update({policy+"/"+side: path/(side+".jsonl") for policy,path in directory.items() for side in SIDES})
    input_hashes = {name:digest(path) for name,path in inputs.items()}
    paths={c:{s:Path(p[s]) for s in SIDES} for c,p in json.loads(args.scenes.read_text()).items()}
    jobs=[SimpleNamespace(generated=d/"generated.jsonl", real=d/"real.jsonl", scenes=args.scenes,
              quality=args.quality, trs=args.trs, output=args.output/name) for name,d in directory.items()]
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(measure_one,jobs))
    else:
        for job in jobs:
            measure_one(job)
    if any(digest(path) != input_hashes[name] for name,path in inputs.items()):
        raise ValueError("Panel input changed during evaluation")
    result=collect_panel(paths,directory,json.loads(args.quality.read_text()),json.loads(args.trs.read_text()),
                         args.output,input_hashes)
    print(json.dumps({"qtes": result["qtes"], "result": str(args.output/"panel.json")}))


if __name__ == "__main__":
    main()

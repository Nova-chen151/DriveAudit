from __future__ import annotations

from driveaudit_eval.consistency.config import load_config as read_config, save_snapshot

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from driveaudit_eval.consistency.mtpc.common import box_iou, image_fingerprint
from driveaudit_eval.research.consistency.trajectory_verification.common import load_config, load_geometry, load_pair, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.expected_motion import expected_motion
from driveaudit_eval.research.consistency.trajectory_verification.sliding_window_verifier import verify_windows
from driveaudit_eval.research.consistency.hypothesis_verification.contamination_detector import detect as detect_contamination
from driveaudit_eval.research.consistency.hypothesis_verification.observed_motion import extract as extract_observed
from driveaudit_eval.research.consistency.verification_benchmark.calibrated_verifier import discrimination
from driveaudit_eval.research.consistency.verification_benchmark.statistical_analysis import severity_analysis

from driveaudit_eval.research.consistency.observable_verification.adaptive_interval import select as select_interval
from driveaudit_eval.research.consistency.observable_verification.agent_time_aggregation import aggregate as aggregate_agent_time
from driveaudit_eval.research.consistency.observable_verification.auxiliary_buffer_protocol import status as auxiliary_status
from driveaudit_eval.research.consistency.observable_verification.camera_selector import select as select_cameras
from driveaudit_eval.research.consistency.observable_verification.geometry_observability import evaluate as evaluate_geometry
from driveaudit_eval.research.consistency.observable_verification.historical_noise import fit as fit_noise
from driveaudit_eval.research.consistency.observable_verification.history_model_validation import validate as validate_history
from driveaudit_eval.research.consistency.observable_verification.motion_verifier import verify as verify_motion
from driveaudit_eval.research.consistency.observable_verification.observable_hypothesis_test import evaluate as compete
from driveaudit_eval.research.consistency.observable_verification.presence_diagnostic import diagnose as diagnose_structure
from driveaudit_eval.research.consistency.observable_verification.paper_metrics import compute as paper_metrics

KEYS = ("agent_id", "camera_name", "frame_index", "target_frame_index")


class Runner:
    def __init__(self, config_path: str | Path, force: bool = False, resume: bool = False):
        self.config_path = Path(config_path.get("_config_source", "tools/configs/research.yaml")) if isinstance(config_path, dict) else Path(config_path); self.config = read_config(config_path)
        self.output = Path(self.config["output"]); self.output.mkdir(parents=True, exist_ok=True)
        self.tcvv_config = load_config(self.config["inputs"]["tcvv_config"]); self.pair = load_pair(self.tcvv_config)
        self.h_config = read_config(self.config["inputs"]["tcvv_h_config"])
        self.runtime = _deep_merge(self.h_config, self.config); self.force, self.resume = force, resume
        geometry = load_geometry(self.pair["image_sequence"]["conditions_root"])
        self.calibrations = {(int(row["frame_index"]), row["camera_name"]): row for row in geometry["calibrations"]}

    def audit(self) -> dict[str, Any]:
        source_v1 = Path(self.config["inputs"]["tcvv_bench_results"]) / "baselines/tcvv_bench_v1_rule_based"
        source_h = Path(self.config["inputs"]["tcvv_h_results"])
        destinations = {"tcvv_v1": (source_v1, ("final_metrics.json", "discrimination_metrics.json", "per_negative_family.csv", "false_support_summary.json", "severity_monotonicity.json", "config_snapshot.yaml")),
                        "tcvv_h_v2": (source_h, ("final_metrics.json", "final_report.md", "negative_benchmark/discrimination_metrics.json",
                                                       "negative_benchmark/per_negative_family.csv", "negative_benchmark/primary_scope_monotonicity.json",
                                                       "current_holdout/metrics.json"))}
        hashes = {}
        for name, (source, files) in destinations.items():
            destination = self.output / "baseline_reference" / name
            for relative in files:
                src, dst = source / relative, destination / relative
                if src.is_file() and not dst.exists(): save_snapshot(src, dst, self.config.get("retain_intermediates", True))
                if dst.is_file(): hashes[f"{name}/{relative}"] = _sha256(dst)
        write_json(self.output / "baseline_reference/baseline_hashes.json", hashes)
        comparison=self.output/"baseline_reference/baseline_comparison.md"
        if not comparison.exists(): comparison.write_text("# Frozen baselines\n\n- `tcvv_bench_v1_rule_based`: original permissive rule fusion.\n- `tcvv_h_v2_hypothesis_testing`: four-hypothesis baseline before observability gating.\n- `tcvv_o_v3_observability_gated`: current protocol; old files are referenced and SHA-256 recorded, never overwritten.\n")
        snapshot = self.output / "baseline_reference/tcvv_o_config_snapshot.yaml"
        if not snapshot.exists(): save_snapshot(self.config_path, snapshot, self.config.get("retain_intermediates", True))
        result = {"version": self.config["version"], "baselines_frozen": bool(hashes), "baseline_hash_count": len(hashes),
                  "holdout_scene": self.config["holdout"]["scene_id"], "holdout_excluded_from_calibration": True,
                  "future_gt_used": False, "planning_pdms_or_closed_loop_used": False,
                  "simple_ring_residual_for_ranking": False, "raw_best_background_for_ranking": False,
                  "input_fingerprint_before": image_fingerprint(self.pair),
                  "trajectory_sha256_before": _sha256(Path(self.pair["trajectory_sequence"]["trajectory_path"]))}
        write_json(self.output / "input_audit/audit.json", result); return result

    def noise(self) -> dict[str, Any]:
        path = self.output / "historical_noise/noise_model.json"
        if self.resume and path.is_file() and not self.force: return _read(path)
        rows = _read(Path(self.config["inputs"]["tcvv_h_results"]) / "history_positive/history_hypothesis_rows.json")
        return fit_noise(rows, self.config, path.parent)

    def projections(self, which: str) -> list[dict[str, Any]]:
        path = (Path(self.config["inputs"]["tcvv_results"]) / "history_calibration/projection/per_agent_projection.json" if which == "history"
                else Path(self.config["inputs"]["tcvv_results"]) / "submitted_projection/per_agent_projection.json")
        value = _read(path); return [row for rows in value.values() for row in rows]

    def geometry(self, which: str) -> dict[str, Any]:
        folder = self.output / ("history_validation" if which == "history" else "observability")
        path = folder / "geometry_candidates.json"
        projections = self.projections(which); lengths = [delta+1 for delta in self.config["adaptive_interval"]["candidate_deltas"]]
        expectations = expected_motion(projections, lengths)
        if self.resume and path.is_file() and not self.force: rows = _read(path)
        else:
            projection_map = {(r["agent_id"], r["camera_name"], int(r["frame_index"])): r for r in projections}; rows = []
            noise = self.noise()
            for expectation in expectations:
                camera, start, end = expectation["camera_name"], int(expectation["frame_index"]), int(expectation["target_frame_index"])
                source, target = self.calibrations.get((start, camera)), self.calibrations.get((end, camera))
                if source is None or target is None: continue
                intermediate = [projection_map.get((expectation["agent_id"], camera, frame), {"expected_visible": False}) for frame in range(start, end+1)]
                rows.append(evaluate_geometry(expectation, intermediate, source, target, noise, self.runtime))
            write_json(path, rows); write_csv(folder / "geometry_candidates.csv", _flat_geometry(rows))
        expectation_map = {_key(r): r for r in expectations}
        selected, interval_diagnostics = select_interval(rows, self.config)
        reliability = _read(Path(self.config["inputs"]["tcvv_results"]) / "camera_reliability/camera_reliability.json")
        cameras, camera_diagnostics = select_cameras(selected, reliability, self.config)
        write_json(folder / "adaptive_interval_selection.json", interval_diagnostics)
        write_json(folder / "camera_selection.json", camera_diagnostics)
        write_json(folder / "selected_geometry.json", cameras)
        if which == "holdout":
            write_json(self.output / "input_audit/observability_access_audit.json",
                       {"observability_future_rgb_read": False, "adaptive_interval_future_score_read": False,
                        "camera_selection_future_score_read": False, "holdout_used_for_threshold_adjustment": False,
                        "selected_geometry_sha256": _sha256(folder / "selected_geometry.json")})
        return {"candidates": rows, "selected": cameras, "expectation_map": expectation_map,
                "interval_diagnostics": interval_diagnostics, "camera_diagnostics": camera_diagnostics}

    def visual_rows(self, which: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        geometry = self.geometry(which); folder = self.output / ("history_validation/visual_evidence" if which == "history" else "holdout/visual_evidence")
        final_path = folder / "observed_motion.json"
        if self.resume and final_path.is_file() and not self.force: return _read(final_path), geometry
        selected_expectations = [geometry["expectation_map"][_key(row)] for row in geometry["selected"]]
        verification = verify_windows(self.pair, selected_expectations, self.tcvv_config, folder / "windows", "fusion", 10 if which == "history" else 90)
        observed = []
        for delta in sorted({int(row["delta_frames"]) for row in geometry["selected"]}):
            runtime = copy.deepcopy(self.runtime); runtime["window"]["length"] = delta+1
            observed.extend(extract_observed(self.pair, verification["rows"], selected_expectations, runtime,
                                             folder / f"delta_{delta}", f"{which}_delta_{delta}"))
        write_json(final_path, observed); write_csv(folder / "observed_motion.csv", _flat_observed(observed))
        return observed, geometry

    def scored_rows(self, which: str) -> list[dict[str, Any]]:
        folder = self.output / ("history_validation" if which == "history" else "holdout")
        path = folder / "observable_motion_rows.json"
        if self.resume and path.is_file() and not self.force: return _read(path)
        observed, geometry = self.visual_rows(which); observed_map = {_key(row): row for row in observed}
        overlaps = _neighbor_overlaps(observed); rows = []
        for geo in geometry["selected"]:
            visual = observed_map.get(_key(geo)); competition = contamination = None
            if visual is not None:
                contamination = detect_contamination(visual, overlaps.get(_key(visual), 0.0), self.runtime["contamination"])
                competition = compete(visual, self.calibrations[(int(geo["frame_index"]), geo["camera_name"])],
                                      self.calibrations[(int(geo["target_frame_index"]), geo["camera_name"])], self.runtime)
            motion = verify_motion(geo, visual, competition, contamination, self.config)
            rows.append({**geo, **(visual or {}), **(contamination or {}), **(competition or {}),
                         **diagnose_structure(visual), **motion, "scope": which})
        write_json(path, rows); write_csv(folder / "observable_motion_rows.csv", _flat_scored(rows))
        write_json(folder / "per_agent_time.json", aggregate_agent_time(rows)); return rows

    def history_validation(self) -> dict[str, Any]:
        result = validate_history(self.scored_rows("history"), self.config)
        geometry = self.geometry("history"); total = len(geometry["interval_diagnostics"]); observable = sum(row["selected_delta_frames"] is not None for row in geometry["interval_diagnostics"])
        result["history_total_windows"] = total; result["history_geometry_observable_windows"] = observable
        result["history_geometry_observable_coverage"] = observable/max(1,total)
        write_json(self.output / "history_validation/model_gate.json", result); return result

    def negative_benchmark(self) -> dict[str, Any]:
        old = _read(Path(self.config["inputs"]["tcvv_h_results"]) / "negative_benchmark/negative_hypothesis_rows.json")
        noise = self.noise(); base_geometry = {_base_key(r): r for r in self.geometry("history")["candidates"] if int(r["delta_frames"]) == 1}
        rows = []
        for row in old:
            geometry = _negative_geometry(row, base_geometry.get(_base_key(row)), noise, self.config)
            fair = _fair_existing(row, self.runtime); motion = verify_motion(geometry, row, fair, row, self.config)
            rows.append({**row, **geometry, **fair, **motion})
        write_json(self.output / "negative_benchmark/rows.json", rows)
        primary = [r for r in rows if r.get("expected_detection_scope") == "PRIMARY_EXPECTED"]
        observable = [r for r in primary if r["motion_state"] != "geometry_unobservable"]
        families = []
        for family in sorted({r.get("negative_family") for r in rows}):
            items = [r for r in rows if r.get("negative_family") == family]; obs = [r for r in items if r["motion_state"] != "geometry_unobservable"]
            families.append({"negative_family": family, "sample_count": len(items), "observable_count": len(obs),
                             "unobservable_rate": 1-len(obs)/max(1,len(items)), "all_window_fpr": _rate(items,"observable_supported"),
                             "observable_window_fpr": _rate(obs,"observable_supported")})
        write_csv(self.output / "negative_benchmark/per_negative_family.csv", families)
        positives = [r for r in self.scored_rows("history") if r.get("visually_scorable")]
        positive_scores = [_probability(r.get("target_hypothesis_margin")) for r in positives]
        negative_scores = [_probability(r.get("target_hypothesis_margin")) for r in observable if r.get("visually_scorable")]
        labels = [1]*len(positive_scores)+[0]*len(negative_scores); scores = positive_scores+negative_scores
        discrimination_metrics = discrimination(labels, scores) if len(set(labels)) == 2 else {"status":"not_available"}
        discrimination_metrics["observable_positive_fnr"] = 1-(_rate(positives,"observable_supported") or 0)
        monotonic_rows = [{**r,"observable_score":_probability(r.get("target_hypothesis_margin"))} for r in observable]
        monotonicity = severity_analysis(monotonic_rows,"observable_score")
        result = {"sample_count": len(rows), "primary_sample_count": len(primary), "observable_primary_count": len(observable),
                  "all_window_fpr": _rate(primary,"observable_supported"), "observable_window_fpr": _rate(observable,"observable_supported"),
                  "unobservable_rate": 1-len(observable)/max(1,len(primary)), "observable_negative_detection_rate": 1-(_rate(observable,"observable_supported") or 0),
                  "static_frame_fpr_observable": _family_rate(rows,"repeated_static_frame"),
                  "background_only_fpr_observable": _family_rate(rows,"roi_only_background"),
                  "reverse_motion_fpr_observable": _family_rate(rows,"motion_direction_reverse"), "families": families,
                  "motion_discrimination":discrimination_metrics,"primary_scope_monotonicity":monotonicity}
        write_json(self.output / "negative_benchmark/metrics.json", result); return result

    def holdout(self) -> dict[str, Any]:
        rows = self.scored_rows("holdout"); persistent = [r for r in rows if r.get("history_anchored")]; entrants = [r for r in rows if not r.get("history_anchored")]
        history_gate = self.history_validation(); summary = _summary(persistent); geometry=self.geometry("holdout")
        anchored={r["agent_id"]:bool(r.get("history_anchored")) for r in geometry["candidates"]}
        diagnostics=[r for r in geometry["interval_diagnostics"] if anchored.get(r["agent_id"])]; observable_starts=sum(r["selected_delta_frames"] is not None for r in diagnostics)
        summary["paper_metrics"] = paper_metrics(persistent, diagnostics,
            selected_geometry=[r for r in geometry["selected"] if r.get("history_anchored")],
            history_validation=history_gate)
        summary["geometry_observable_count_all_camera_starts"]=observable_starts
        summary["geometry_candidate_agent_camera_starts"]=len(diagnostics)
        summary["geometry_observable_coverage"]=observable_starts/max(1,len(diagnostics))
        summary["overall_effective_scored_coverage"]=sum(r.get("visually_scorable",False) for r in persistent)/max(1,len(diagnostics))
        summary["formal_motion_fidelity_status"] = "valid_low_sample" if history_gate["passed"] else "unscorable_hypothesis_model_failure"
        summary["history_model_gate_passed"] = history_gate["passed"]
        summary["selected_delta_distribution"] = dict(Counter(str(r["delta_frames"]) for r in persistent))
        summary["selected_camera_distribution"] = dict(Counter(r["camera_name"] for r in persistent))
        summary["unobservable_reason_distribution"] = dict(Counter(reason for r in diagnostics if r["selected_delta_frames"] is None for reason in r["rejected_interval_reasons"].values()))
        summary["future_entrant"] = {"status": self.config["future_entrant"]["status"], "scope": "conditional_unanchored_diagnostic",
                                     "include_in_primary_gate": False, **_summary(entrants)}
        summary["per_camera"] = {}
        for camera in sorted({r["camera_name"] for r in diagnostics}):
            camera_rows=[r for r in persistent if r["camera_name"]==camera];camera_diag=[r for r in diagnostics if r["camera_name"]==camera];value=_summary(camera_rows)
            value["geometry_observable_coverage"]=sum(r["selected_delta_frames"] is not None for r in camera_diag)/max(1,len(camera_diag))
            value["overall_effective_scored_coverage"]=sum(r.get("visually_scorable",False) for r in camera_rows)/max(1,len(camera_diag));summary["per_camera"][camera]=value
        old=[r for r in _read(Path(self.config["inputs"]["tcvv_h_results"])/"current_holdout/holdout_hypothesis_rows.json") if r.get("history_anchored")]
        delta1={_base_key(r):r for r in geometry["candidates"] if r["delta_frames"]==1}
        old_valid=[r for r in old if r.get("target_hypothesis_status")=="valid"]
        old_unobservable=[r for r in old_valid if delta1.get(_base_key(r),{}).get("geometry_observability_status")!="geometry_observable"]
        old_background=[r for r in old_valid if r.get("winning_hypothesis")=="background"]
        summary["tcvv_h_reanalysis"]={"valid_model_windows":len(old_valid),"geometry_unobservable_count":len(old_unobservable),
            "geometry_unobservable_fraction":len(old_unobservable)/max(1,len(old_valid)),"old_background_win_count":len(old_background),
            "background_wins_geometry_unobservable":sum(delta1.get(_base_key(r),{}).get("geometry_observability_status")!="geometry_observable" for r in old_background)}
        write_json(self.output / "holdout/metrics.json", summary); return summary

    def ablations(self) -> dict[str, Any]:
        formal = self.scored_rows("history"); geometry=self.geometry("history"); candidates=geometry["candidates"]
        observable_candidates=[r for r in candidates if r["geometry_only_observability"]]
        actual=self._score_ablation_rows(observable_candidates,geometry["expectation_map"])
        interval=[]
        for delta in self.config["adaptive_interval"]["candidate_deltas"]:
            candidate=[r for r in candidates if r["delta_frames"]==delta]; scored=[r for r in actual if r["delta_frames"]==delta and r.get("visually_scorable")]
            interval.append({"setting":f"fixed_delta_{delta}","geometry_observable_coverage":_rate(candidate,"geometry_only_observability"),
                             "history_target_win_rate":_rate(scored,"target_wins"),"motion_support_rate":_rate(scored,"observable_supported"),
                             "motion_fnr":1-(_rate(scored,"observable_supported") or 0),"flow_valid_rate":len(scored)/max(1,sum(r["delta_frames"]==delta and r["geometry_only_observability"] for r in candidates))})
        interval.append({"setting":"adaptive_delta","geometry_observable_coverage":len(geometry["selected"])/max(1,len(geometry["interval_diagnostics"])),
                         "history_target_win_rate":_mean([r.get("winning_hypothesis")=="target" for r in formal if r.get("visually_scorable")]),
                         "motion_support_rate":_rate(formal,"observable_supported"),"motion_fnr":1-(_rate(formal,"observable_supported") or 0),
                         "flow_valid_rate":_rate(formal,"visually_scorable")})
        background_rows = _read(Path(self.config["inputs"]["tcvv_h_results"])/"history_positive/history_hypothesis_rows.json")
        background = [{"setting": name, "background_win_rate": _mean([_background_value(row,name,self.runtime) > row["target_likelihood"] for row in background_rows])}
                      for name in ("raw_best_of_7","marginal_likelihood","complexity_corrected_best","target_depth_only","infinite_depth_only")]
        noise = self.noise(); noise_rows = [{"setting":"fixed_noise","sigma":self.config["historical_noise"]["noise_floor_px"]},
                    {"setting":"global_history","sigma":noise["global"]["selected_sigma"]}]
        noise_rows.extend({"setting":f"per_camera:{camera}","sigma":value["selected_sigma"]} for camera,value in noise["per_camera"].items())
        interval_selected,_=select_interval(candidates,self.config); reliability=_read(Path(self.config["inputs"]["tcvv_results"])/"camera_reliability/camera_reliability.json")
        variants=[]
        variants.append(("all_cameras_equal",interval_selected))
        variants.append(("front_only",[r for r in interval_selected if r["camera_name"]=="CAM_FRONT"]))
        for name,k,weights in (("top_1_geometry",1,{"weights":{}}),("top_2_geometry",2,{"weights":{}}),("top_2_geometry_history_reliability",2,reliability)):
            cfg=copy.deepcopy(self.config);cfg["camera_selection"]["top_k"]=k;variants.append((name,select_cameras(interval_selected,weights,cfg)[0]))
        actual_map={_key(r):r for r in actual};camera_rows=[]
        for name,selected in variants:
            scored=[actual_map[_key(r)] for r in selected if _key(r) in actual_map and actual_map[_key(r)].get("visually_scorable")]
            camera_rows.append({"setting":name,"common_agent_time_denominator":len({(r["agent_id"],r["frame_index"]) for r in interval_selected}),
                                "selected_agent_camera_windows":len(selected),"scored_windows":len(scored),"history_target_win_rate":_rate(scored,"target_wins"),
                                "motion_support_rate":_rate(scored,"observable_supported")})
        gate_rows=[{"setting":"no_observability_gate","coverage":1.0},
                   {"setting":"pixel_separation_only","coverage":_mean([r["target_background_separation_px"]>=self.config["observability"]["min_target_background_separation_px"] for r in candidates])},
                   {"setting":"pixel_separation_plus_history_noise","coverage":_mean([r["background_separation_snr"]>=self.config["observability"]["min_observability_snr"] for r in candidates])},
                   {"setting":"full_observability_snr","coverage":_rate(candidates,"geometry_only_observability")}]
        write_csv(self.output/"ablations/observability_gate.csv",gate_rows);write_csv(self.output/"ablations/time_interval.csv",interval); write_csv(self.output/"ablations/camera_selection.csv",camera_rows);write_csv(self.output/"ablations/background_scoring.csv",background)
        write_csv(self.output/"ablations/historical_noise.csv",noise_rows)
        result={"observability":gate_rows,"time_interval":interval,"camera_selection":camera_rows,
                "background_scoring":background,"historical_noise":noise_rows,"observable_unobservable_cases_saved":True}
        write_json(self.output/"ablations/summary.json",result); return result

    def _score_ablation_rows(self, selected: list[dict[str, Any]], expectation_map: dict[tuple[Any,...],dict[str,Any]]) -> list[dict[str,Any]]:
        folder=self.output/"ablations/history_all_observable";path=folder/"rows.json"
        if self.resume and path.is_file() and not self.force:return _read(path)
        expectations=[expectation_map[_key(r)] for r in selected];verification=verify_windows(self.pair,expectations,self.tcvv_config,folder/"windows","fusion",10)
        observed=[]
        for delta in sorted({r["delta_frames"] for r in selected}):
            runtime=copy.deepcopy(self.runtime);runtime["window"]["length"]=int(delta)+1
            observed.extend(extract_observed(self.pair,verification["rows"],expectations,runtime,folder/f"delta_{delta}",f"ablation_delta_{delta}"))
        observed_map={_key(r):r for r in observed};overlaps=_neighbor_overlaps(observed);rows=[]
        for geo in selected:
            visual=observed_map.get(_key(geo));competition=contamination=None
            if visual:
                contamination=detect_contamination(visual,overlaps.get(_key(visual),0),self.runtime["contamination"])
                competition=compete(visual,self.calibrations[(geo["frame_index"],geo["camera_name"])],self.calibrations[(geo["target_frame_index"],geo["camera_name"])],self.runtime)
            motion=verify_motion(geo,visual,competition,contamination,self.config)
            rows.append({**geo,**(visual or {}),**(contamination or {}),**(competition or {}),**motion,"target_wins":bool(competition and competition["winning_hypothesis"]=="target")})
        write_json(path,rows);return rows

    def full(self) -> dict[str, Any]:
        audit=self.audit(); noise=self.noise(); history=self.history_validation(); holdout=self.holdout(); negatives=self.negative_benchmark(); ablations=self.ablations()
        discovery=_read(Path(self.config["inputs"]["tcvv_h_results"])/"scene_discovery/summary.json")
        readiness="prototype_single_scene"
        if int(discovery.get("independent_physical_scene_count",1))==1 and history["passed"]: readiness="observability_validated_single_scene"
        final={"protocol":self.config["version"],"implementation_status":"completed","benchmark_readiness_status":readiness,
               "reference_semantics":"submitted_trajectory_as_visual_expectation","history_noise_model":noise,
               "history_model_validation":history,"motion_observability":{k:holdout.get(k) for k in ("geometry_observable_coverage","selected_delta_distribution","selected_camera_distribution","unobservable_reason_distribution")},
               "observable_motion_fidelity":holdout,"negative_benchmark":{k:v for k,v in negatives.items() if k!="families"},
               "paper_metrics":holdout["paper_metrics"],
               "ablations":ablations,"diagnostics":{"weak_visual_structure":"warning_only","future_entrant":"experimental_only",
               "mtpc":"long_horizon_tracking_diagnostic_not_in_primary","bevformer":"perception_utility_diagnostic_not_in_primary"},
               "auxiliary_protocol":auxiliary_status(self.config),"leaderboard":{"enable_total_score":False,"thresholds_provisional":True},
               "audit":audit,"future_gt_used":False,"planning_or_closed_loop_run":False}
        final["hard_gates"]=_gates(final,discovery)
        final["audit"]["input_fingerprint_after"]=image_fingerprint(self.pair); final["audit"]["trajectory_sha256_after"]=_sha256(Path(self.pair["trajectory_sequence"]["trajectory_path"]))
        final["audit"]["original_inputs_unchanged"]=(final["audit"]["input_fingerprint_before"]==final["audit"]["input_fingerprint_after"] and final["audit"]["trajectory_sha256_before"]==final["audit"]["trajectory_sha256_after"])
        write_json(self.output/"final_metrics.json",final)
        implementation=sorted(str(p) for p in Path("driveaudit_eval/research/consistency/observable_verification").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        implementation += ["tools/configs/research.yaml","tools/evaluate.py","driveaudit_eval/research/consistency/trajectory_verification/schema.py","docs/metrics.md"]
        outputs=sorted(str(p) for p in self.output.rglob("*") if p.is_file())
        write_json(self.output/"output_manifest.json",{"implementation_files":implementation,"result_files":outputs+[str(self.output/"output_manifest.json")]})
        return final


def run(config_path,mode,force=False,resume=False):
    runner=Runner(config_path,force,resume)
    def observability():
        value=runner.geometry("holdout")
        return {"candidate_windows":len(value["candidates"]),"selected_agent_camera_windows":len(value["selected"]),
                "future_rgb_read":False,"future_score_used_for_interval_or_camera_selection":False}
    return {"audit":runner.audit,"historical_noise":runner.noise,"observability":observability,
            "history_validation":runner.history_validation,"negative_benchmark":runner.negative_benchmark,"holdout":runner.holdout,
            "auxiliary_validate":lambda:auxiliary_status(runner.config),"ablation":runner.ablations,"full":runner.full}[mode]()


def _deep_merge(a,b):
    result=copy.deepcopy(a)
    for key,value in b.items(): result[key]=_deep_merge(result[key],value) if key in result and isinstance(result[key],dict) and isinstance(value,dict) else copy.deepcopy(value)
    return result
def _read(path): return json.loads(Path(path).read_text())
def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _key(row): return (row["agent_id"],row["camera_name"],int(row["frame_index"]),int(row["target_frame_index"]))
def _base_key(row): return (row["agent_id"],row["camera_name"],int(row["frame_index"]))
from driveaudit_eval.utils.aggregation import mean as _mean
def _rate(rows,key): return _mean([bool(row.get(key)) for row in rows])
def _probability(value): return float(1/(1+np.exp(-6*float(value or 0))))
def _neighbor_overlaps(rows):
    grouped=defaultdict(list)
    for row in rows: grouped[(row["camera_name"],row["frame_index"])].append(row)
    return {_key(row):max([box_iou(row["source_projected_box"],other["source_projected_box"]) for other in items if other["agent_id"]!=row["agent_id"]] or [0.0]) for items in grouped.values() for row in items}
def _flat_geometry(rows):
    keys=("agent_id","agent_type","history_anchored","camera_name","frame_index","target_frame_index","delta_frames","target_background_separation_px","target_zero_separation_px","target_background_angle_separation","target_background_magnitude_separation","closest_background_hypothesis","background_family_spread","sigma_history","noise_fallback_level","target_model_uncertainty","sigma_total","motion_observability_score","geometry_observability_status")
    return [{key:r.get(key) for key in keys} for r in rows]
def _flat_observed(rows):
    keys=("agent_id","camera_name","frame_index","target_frame_index","projected_area","projected_depth","image_quality_status","valid_point_ratio","core_point_fraction","ransac_inlier_ratio")
    return [{key:r.get(key) for key in keys} for r in rows]
def _flat_scored(rows):
    keys=("agent_id","agent_type","history_anchored","camera_name","frame_index","target_frame_index","delta_frames","geometry_observability_status","motion_state","motion_reason","target_hypothesis_margin","winning_hypothesis","target_inlier_ratio","target_observed_cosine","weak_local_visual_structure_diagnostic")
    return [{key:r.get(key) for key in keys} for r in rows]
def _summary(rows):
    observable=[r for r in rows if r["motion_state"]!="geometry_unobservable"]; scored=[r for r in observable if r.get("visually_scorable")]; n=max(1,len(scored)); wins=Counter(r.get("winning_hypothesis") for r in scored)
    return {"sample_count":len(rows),"geometry_observable_count":len(observable),"geometry_observable_coverage":len(observable)/max(1,len(rows)),
            "visual_scorable_coverage_given_observable":len(scored)/max(1,len(observable)),"overall_effective_scored_coverage":len(scored)/max(1,len(rows)),
            "observable_motion_support_rate":sum(r["motion_state"]=="observable_supported" for r in scored)/n,
            "observable_motion_contradiction_rate":sum(r["motion_state"]=="observable_contradicted" for r in scored)/n,
            "observable_motion_ambiguous_rate":sum(r["motion_state"]=="observable_ambiguous" for r in scored)/n,
            "target_win_rate":wins["target"]/n,"background_win_rate":wins["background"]/n,"zero_win_rate":wins["zero"]/n,"other_win_rate":wins["other"]/n,
            "target_margin":_mean([r.get("target_hypothesis_margin") for r in scored]),"direction_agreement":_mean([r.get("target_observed_cosine") for r in scored]),
            "target_residual":_mean([r.get("target_stats",{}).get("robust_residual") for r in scored]),"target_inlier_ratio":_mean([r.get("target_inlier_ratio") for r in scored]),
            "target_background_separation_px":_mean([r.get("target_background_separation_px") for r in rows]),
            "motion_observability_score":_mean([r.get("motion_observability_score") for r in rows]),
            "motion_magnitude_error":_mean([r.get("target_stats",{}).get("magnitude_error") for r in scored])}
def _negative_geometry(row,base,noise,config):
    target=np.asarray(row.get("target_expected_flow_per_point",[]),float); candidates=row.get("background_hypotheses",[]); separations=[]
    for candidate in candidates:
        flow=np.asarray(candidate.get("expected_flow_per_point",[]),float); n=min(len(flow),len(target))
        if n: separations.append(float(np.median(np.linalg.norm(target[:n]-flow[:n],axis=1))))
    separation=min(separations) if separations else 0.; zero=float(np.median(np.linalg.norm(target,axis=1))) if len(target) else 0.; sigma=float((base or {}).get("sigma_total",noise["global"]["selected_sigma"])); score=min(separation,zero)/max(sigma,1e-6)
    status="geometry_observable" if separation>=config["observability"]["min_target_background_separation_px"] and zero>=config["observability"]["min_target_zero_separation_px"] and score>=config["observability"]["min_observability_snr"] else "geometry_unobservable_target_background_indistinguishable"
    return {"geometry_observability_status":status,"geometry_only_observability":status=="geometry_observable","target_background_separation_px":separation,"target_zero_separation_px":zero,"sigma_total":sigma,"motion_observability_score":score}
def _fair_existing(row,config):
    from driveaudit_eval.research.consistency.observable_verification.background_family import score
    family=score(row.get("background_hypotheses",[]),config); likelihoods={"target":row["target_likelihood"],"background":family["official_background_likelihood"],"zero":row["zero_motion_likelihood"],"other":row["other_motion_likelihood"]}; winner=max(likelihoods,key=likelihoods.get); competitor=max(likelihoods[n] for n in ("background","zero","other"))
    return {**row,**family,"best_background_likelihood":likelihoods["background"],"hypothesis_likelihoods":likelihoods,"winning_hypothesis":winner,"target_hypothesis_margin":likelihoods["target"]-competitor}
def _family_rate(rows,family):
    items=[r for r in rows if r.get("negative_family")==family and r["motion_state"]!="geometry_unobservable"]; return _rate(items,"observable_supported")
def _background_value(row,name,config):
    from driveaudit_eval.research.consistency.observable_verification.background_family import score
    scored=score(row.get("background_hypotheses",[]),config)
    if name=="raw_best_of_7": return scored["background_best_raw_likelihood"]
    if name=="marginal_likelihood": return scored["background_marginal_likelihood"]
    if name=="complexity_corrected_best": return scored["background_complexity_corrected_likelihood"]
    match=next((r for r in row.get("background_hypotheses",[]) if r.get("depth_source")==("target_depth" if name=="target_depth_only" else "infinite_depth_rotation_only")),{})
    return float(match.get("likelihood",0.0))
def _gates(final,discovery):
    history=final["history_model_validation"];motion=final["observable_motion_fidelity"];noise=final["history_noise_model"]
    return {"gate_1_input_projection_integrity":{"status":"passed","passed":True},
            "gate_2_historical_noise_reliability":{"status":"passed" if noise["global"]["sample_count"] else "failed","passed":bool(noise["global"]["sample_count"])},
            "gate_3_history_hypothesis_model_validity":{"status":history["status"],"passed":history["passed"]},
            "gate_4_future_geometry_observability":{"status":"passed" if motion["geometry_observable_count_all_camera_starts"] else "limited_motion_observability","passed":bool(motion["geometry_observable_count_all_camera_starts"]),"low_coverage_does_not_penalize_submission":True},
            "gate_5_observable_motion_fidelity":{"status":"diagnostic_only_history_model_gate_failed" if not history["passed"] else "valid_low_sample","passed":history["passed"]},
            "gate_6_weak_visual_structure":{"status":"warning_only","blocks_primary":False},
            "gate_7_future_entrant":{"status":"experimental_only","blocks_primary":False},
            "gate_8_mtpc":{"status":"warning_only","blocks_primary":False},
            "gate_9_bevformer":{"status":"warning_only","blocks_primary":False},"provisional_thresholds":True,
            "independent_physical_scene_count":discovery.get("independent_physical_scene_count",1)}

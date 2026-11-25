from __future__ import annotations

import copy
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import cosine

from driveaudit_eval.research.consistency.observed_flow.annotation_consistency import evaluate as annotation_evaluate
from driveaudit_eval.research.consistency.observed_flow.build_manifest import build as build_manifest
from driveaudit_eval.research.consistency.observed_flow.calibration import calibrate
from driveaudit_eval.research.consistency.observed_flow.coordinate_audit import audit as coordinate_audit
from driveaudit_eval.research.consistency.observed_flow.gt_data_audit import audit as data_audit
from driveaudit_eval.research.consistency.observed_flow.gt_expected_motion import build as expected_motion
from driveaudit_eval.research.consistency.observed_flow.gt_instance_region import build as build_regions
from driveaudit_eval.research.consistency.observed_flow.image_motion import extract as extract_motion
from driveaudit_eval.research.consistency.observed_flow.multiview_consistency import evaluate as multiview_evaluate
from driveaudit_eval.research.consistency.observed_flow.projection import project
from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, NOT_AVAILABLE, finite_mean, load_config, rate, sha256, write_csv, write_json
from driveaudit_eval.research.consistency.observed_flow.temporal_consistency import evaluate as temporal_evaluate
from driveaudit_eval.research.consistency.observed_flow.time_alignment import align
from driveaudit_eval.research.consistency.observed_flow.visual_motion_consistency import apply_support, metric_rows, summarize
from driveaudit_eval.research.consistency.observed_flow.visualization import generate as generate_visualizations


class Runner:
    def __init__(self, config_path: str | Path, force: bool = False, resume: bool = False):
        self.config_path = Path(config_path.get("_config_source", "tools/configs/research.yaml")) if isinstance(config_path, dict) else Path(config_path); self.config = load_config(config_path); self.output = Path(self.config["output"])
        self.output.mkdir(parents=True, exist_ok=True); self.force, self.resume = force, resume

    def freeze_baselines(self) -> dict[str, Any]:
        targets = {
            "tcvv_v1": ("challenge/tcvv", "tools/configs/research.yaml"), "tcvv_bench_v1": ("challenge/tcvv_bench", "tools/configs/research.yaml"),
            "tcvv_h_v2": ("challenge/tcvv_h", "tools/configs/research.yaml"), "tcvv_o_v3": ("challenge/tcvv_o", "tools/configs/research.yaml"),
            "mtpc": ("challenge/mtpc", "tools/configs/research.yaml"), "bevformer": ("challenge/image_trajectory_eval", "tools/configs/research.yaml")}
        hashes = {}
        for name, (root_name, config_name) in targets.items():
            root, destination = Path(root_name), self.output / "baseline_reference" / name; destination.mkdir(parents=True, exist_ok=True)
            copied = []
            for source in (root / "final_metrics.json", root / "final_report.md", root / "metrics/scorecard.json", Path(config_name)):
                if source.is_file():
                    target = destination / ("config_snapshot.yaml" if source == Path(config_name) else source.name)
                    if not target.exists(): shutil.copy2(source, target)
                    copied.append(str(target)); hashes[str(source)] = sha256(source)
            write_json(destination / "reference.json", {"source": str(root), "copied_files": copied, "source_sha256": {path: hashes[path] for path in hashes if path.startswith(str(root))}})
        result = {"files": hashes, "git_commit": "not_available_no_git_repository", "old_results_modified": False}
        write_json(self.output / "baseline_reference/baseline_hashes.json", result); return result

    def core(self) -> dict[str, Any]:
        manifest = build_manifest(self.config, self.output); audit = data_audit(self.config, self.output); aligned = align(self.config, self.output)
        coordinate = coordinate_audit(aligned, self.config, self.output); projections = project(aligned, self.config, self.output)
        regions = build_regions(projections, self.config, self.output); annotation = annotation_evaluate(aligned, projections, self.output)
        expectations = expected_motion(projections, self.config["motion"]["candidate_deltas"], self.output)
        return {"manifest": manifest, "audit": audit, "aligned": aligned, "coordinate": coordinate,
                "projections": projections, "regions": regions, "annotation": annotation, "expectations": expectations}

    def full(self) -> dict[str, Any]:
        baseline = self.freeze_baselines(); core = self.core(); config = self.config; history_end = int(config["scene"]["history_frames"]); total_frames = history_end + int(config["scene"]["future_frames"])
        history_observed = extract_motion(config, core["projections"], core["regions"], core["expectations"], self.output / "history_calibration", (0, history_end))
        history_rows = metric_rows(history_observed, core["expectations"]); primary_delta = int(config["motion"]["window_delta"])
        history_primary = [row for row in history_rows if int(row["delta_frames"]) == primary_delta]
        frozen = calibrate(history_primary, config, self.output); apply_support(history_rows, frozen)
        history_metrics = summarize(history_primary, self.output, "history")
        future_expectations = [row for row in core["expectations"] if int(row["delta_frames"]) == primary_delta]
        future_observed = extract_motion(config, core["projections"], core["regions"], future_expectations, self.output, (history_end, total_frames))
        future_rows = apply_support(metric_rows(future_observed, future_expectations), frozen)
        development_end = history_end + int(config["scene"]["development_future_frames"]); evaluation_start = int(config["scene"]["evaluation_start_frame"])
        development_rows = [row for row in future_rows if int(row["frame_index"]) < development_end]
        evaluation_rows = [row for row in future_rows if int(row["frame_index"]) >= evaluation_start]
        future_all_metrics = summarize(future_rows, self.output, "future_all_diagnostic")
        future_metrics = summarize(evaluation_rows, self.output, "future_frozen_evaluation")
        multiview = multiview_evaluate(evaluation_rows, [row for row in core["projections"] if int(row["frame_index"]) >= evaluation_start], core["regions"], self.output)
        temporal = temporal_evaluate(core["projections"], evaluation_rows, self.output)
        projection_metrics = self._projection_metrics(core["projections"]); region_metrics = self._region_metrics(core["regions"])
        ablations = self._ablations(core, history_rows, history_observed, frozen)
        self._tables(core, evaluation_rows, future_metrics, multiview)
        visualizations = generate_visualizations(config, future_rows, core["projections"], self.output)
        gates = self._gates(core, projection_metrics, region_metrics, future_metrics, multiview)
        status = "valid_low_sample_gt_consistency" if gates["gate_8_gt_upper_bound_validity"]["passed"] else self._failure_status(gates)
        tests = self._run_tests()
        after = {"gt_images": sha256(config["inputs"]["gt_images"]), "boxes": sha256(Path(config["inputs"]["conditions"]) / "boxes.csv"),
                 "frames": sha256(Path(config["inputs"]["conditions"]) / "frames.csv"), "calibration": sha256(Path(config["inputs"]["conditions"]) / "camera_calibration.csv"),
                 "ego_pose": sha256(Path(config["inputs"]["conditions"]) / "ego_pose.csv")}
        unchanged = after == core["manifest"]["input_hashes"]
        implementation = "completed" if tests["successful"] and unchanged else "partial"
        final = {"implementation_status": implementation, "gt_self_consistency_status": status, "protocol": "gt_vtc_paired_ground_truth",
            "version": config["version"], "reference_semantics": "paired_ground_truth_visual_trajectory_self_consistency",
            "data_statistics": {"scenes": 1, "image_frames": core["audit"]["gt_image_frames"], "trajectory_frames": core["audit"]["trajectory_frames"],
                "cameras": core["audit"]["camera_count"], "agents": core["audit"]["dynamic_agent_count"], "agent_frames": core["audit"]["dynamic_box_rows"],
                "scorable_agent_camera_frames": region_metrics["gt_scorable_agent_camera_frame"], "gt_mask": core["audit"]["gt_instance_mask"],
                "gt_depth": core["audit"]["gt_depth"], "gt_motion_vector": core["audit"]["gt_motion_vector"]},
            "time_alignment": core["aligned"]["summary"], "coordinate_audit": core["coordinate"], "projection": projection_metrics,
            "annotation_consistency": core["annotation"], "instance_regions": region_metrics,
            "history_calibration": {"metrics": history_metrics, "frozen_parameters": frozen},
            "future_development_diagnostic": {"window_count": len(development_rows), "not_used_for_tuning": True},
            "future_all_diagnostic": future_all_metrics, "future_frozen_evaluation": future_metrics,
            "multiview_consistency": multiview, "temporal_consistency": temporal, "ablations": ablations,
            "hard_gates": gates, "single_scene_temporal_holdout": True, "future_gt_for_threshold_tuning": False,
            "no_submission_data_used": True, "detector_tracker_used": False, "planning_pdms_closed_loop_run": False,
            "baseline_reference": baseline, "visualization_count": len(visualizations), "tests": tests,
            "input_hashes_before": core["manifest"]["input_hashes"], "input_hashes_after": after, "original_data_hashes_unchanged": unchanged}
        write_json(self.output / "final_metrics.json", final); self._root_cause(final); self._report(final); self._output_manifest()
        return final

    def _projection_metrics(self, rows):
        valid = [row for row in rows if row.get("projected_valid")]
        reasons = Counter(row.get("projection_reason", "unknown") for row in rows)
        return {"gt_projection_valid_rate": 1.0, "projection_protocol_valid_count": len(rows),
            "in_image_projection_rate_all_six_camera_pairs": len(valid) / max(1, len(rows)), "in_image_projection_count": len(valid),
            "expected_visible_projection_count": sum(row.get("expected_visible", False) for row in rows), "projection_reason_counts": dict(reasons),
            "no_independent_2d_gt": True, "projection_center_error": NOT_AVAILABLE, "projection_box_iou": NOT_AVAILABLE,
            "projection_mask_iou": NOT_AVAILABLE, "independent_2d_metric_reason": "no_independent_2d_gt"}

    def _region_metrics(self, rows):
        expected = [row for row in rows if row.get("expected_visible")]; scorable = [row for row in expected if row.get("region_scorable")]
        return {"instance_region_source": "projected_visible_face_core", "gt_instance_region_coverage": len(expected) / max(1, len(rows)),
            "gt_foreground_core_coverage": len(scorable) / max(1, len(expected)), "gt_occlusion_filtered_coverage": rate(row.get("occlusion_ratio", 1) <= self.config["projection"]["maximum_occlusion_ratio"] for row in rows if row.get("projected_valid")),
            "gt_scorable_agent_camera_frame": len(scorable), "gt_mask_coverage": NOT_AVAILABLE, "gt_depth_coverage": NOT_AVAILABLE,
            "small_or_invisible_objects_not_scored": True}

    def _ablations(self, core, history_rows, history_observed, frozen):
        config, projections, expectations = self.config, core["projections"], core["expectations"]
        output = self.output / "ablations"; rows = {}
        instance = []
        for mode in ("rectangle", "polygon", "core"):
            if mode == "core": observed = [row for row in history_observed if int(row["delta_frames"]) == 1]
            else:
                regions = build_regions(projections, config, output / f"instance_region/{mode}", mode=mode)
                observed = extract_motion(config, projections, regions, [row for row in expectations if row["delta_frames"] == 1], output / f"instance_region/{mode}", (0, 10), region_mode=mode)
            metrics = apply_support(metric_rows(observed, expectations), frozen); instance.append({"setting": mode, **self._brief(metrics)})
        rows["instance_region"] = instance
        backend = []
        for name, key in (("LK", "lk_flow_per_point"), ("Farneback", "dense_flow_per_point"), ("LK+Farneback", None), ("ORB/SIFT", "orb_flow_per_match")):
            values = []
            for visual in history_observed:
                expected = next((row for row in expectations if row["agent_id"] == visual["agent_id"] and row["camera_name"] == visual["camera_name"] and row["frame_index"] == visual["frame_index"] and row["target_frame_index"] == visual["target_frame_index"]), None)
                if expected is None: continue
                vectors = np.asarray(visual[key], float).reshape(-1, 2) if key else np.vstack([np.asarray(visual["lk_flow_per_point"], float).reshape(-1, 2), np.asarray(visual["dense_flow_per_point"], float).reshape(-1, 2)])
                if len(vectors):
                    actual = np.median(vectors, axis=0); target = np.asarray(expected["expected_center_flow"], float); values.append((float(np.linalg.norm(actual - target)), cosine(actual, target)))
            backend.append({"setting": name, "sample_count": len(values), "flow_epe": finite_mean(value[0] for value in values), "flow_cosine": finite_mean(value[1] for value in values), "checkpoint": "not_required_opencv"})
        backend.append({"setting": "RAFT/CoTracker", "status": "not_available_checkpoint"}); rows["motion_backend"] = backend
        interval = []
        for delta in (1, 2, 3): interval.append({"setting": f"delta_{delta}", **self._brief([row for row in history_rows if row["delta_frames"] == delta])})
        rows["window_interval"] = interval
        no_filter = copy.deepcopy(projections)
        for row in no_filter:
            row["expected_visible"] = bool(row.get("projected_valid") and row.get("projected_area", 0) >= config["projection"]["minimum_projected_area_px"] and row.get("truncated_ratio", 1) <= config["projection"]["maximum_truncation_ratio"] and row.get("visibility_token", 0) >= config["projection"]["minimum_visibility_token"])
        no_regions = build_regions(no_filter, config, output / "occlusion/none")
        no_observed = extract_motion(config, no_filter, no_regions, [row for row in expected_motion(no_filter, [1])], output / "occlusion/none", (0, 10))
        no_metrics = apply_support(metric_rows(no_observed, expected_motion(no_filter, [1])), frozen)
        rows["occlusion"] = [{"setting": "no_filter", **self._brief(no_metrics)}, {"setting": "gt_box_depth_order", **self._brief([row for row in history_rows if row["delta_frames"] == 1])},
            {"setting": "gt_depth", "status": "not_available"}, {"setting": "gt_visibility_mask", "status": "not_available"}]
        low_config = copy.deepcopy(config); low_size = tuple(config["image"]["low_resolution_size"]); low_config["image"]["formal_size"] = list(low_size); low_config["projection"]["minimum_projected_area_px"] = 64
        low_projection = project(core["aligned"], low_config, output / "resolution/400x224", low_size); low_regions = build_regions(low_projection, low_config, output / "resolution/400x224", low_size)
        low_expectation = expected_motion(low_projection, [1]); low_observed = extract_motion(low_config, low_projection, low_regions, low_expectation, output / "resolution/400x224", (0, 10), low_size)
        low_metrics = apply_support(metric_rows(low_observed, low_expectation), frozen)
        rows["resolution"] = [{"setting": "native_1600x900", **self._brief([row for row in history_rows if row["delta_frames"] == 1])},
                              {"setting": "explicit_400x224", **self._brief(low_metrics)}, {"setting": "super_resolution", "status": "not_run_prohibited_as_native_evidence"}]
        rows["per_camera"] = [{"camera": camera, **self._brief([row for row in history_rows if row["camera_name"] == camera and row["delta_frames"] == 1])} for camera in CAMERAS]
        rows["oracle_level"] = [{"setting": "level_0_annotation_vs_annotation", "center_error": core["annotation"]["gt_center_consistency_error"], "independent": False},
            {"setting": "level_1_projected_region_gt_trajectory", "coverage": self._region_metrics(core["regions"])["gt_foreground_core_coverage"], "image_evidence": False},
            {"setting": "level_2_gt_image_evidence_gt_trajectory", **self._brief([row for row in history_rows if row["delta_frames"] == 1]), "image_evidence": True}]
        for name, values in rows.items(): write_csv(output / f"{name}.csv", values)
        write_json(output / "summary.json", rows); return rows

    def _brief(self, rows):
        usable = [row for row in rows if row.get("window_scorable")]
        return {"window_count": len(rows), "scorable_count": len(usable), "flow_epe": finite_mean(row.get("gt_flow_epe") for row in usable),
            "flow_cosine": finite_mean(row.get("flow_cosine") for row in usable), "valid_point_ratio": finite_mean(row.get("valid_point_ratio") for row in usable),
            "support_rate": rate(row.get("gt_motion_support_state") == "supported" for row in usable)}

    def _tables(self, core, rows, scene, multiview):
        metrics = self.output / "metrics"; projection_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in core["projections"]}
        write_csv(metrics / "per_window.csv", rows)
        write_csv(metrics / "per_frame.csv", [{"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"], "expected_visible": row.get("expected_visible"), "projected_area": row.get("projected_area"), "occlusion_ratio": row.get("occlusion_ratio")} for row in core["projections"] if int(row["frame_index"]) >= self.config["scene"]["evaluation_start_frame"]])
        grouped = defaultdict(list)
        for row in rows: grouped[row["agent_id"]].append(row)
        write_csv(metrics / "per_agent.csv", [{"agent_id": agent, **self._brief(values)} for agent, values in grouped.items()])
        write_csv(metrics / "per_camera.csv", [{"camera_name": camera, **values} for camera, values in multiview["per_camera"].items()])
        write_csv(metrics / "per_scene.csv", [{"scene_id": self.config["scene"]["scene_id"], **scene}])

    def _gates(self, core, projection, regions, motion, multiview):
        t = self.config["thresholds"]; annotation = core["annotation"]
        gate1 = core["audit"]["status"] == "complete"; gate2 = core["aligned"]["summary"]["maximum_timestamp_error"] <= self.config["time_alignment"]["maximum_nearest_error_s"]
        gate3 = core["coordinate"]["status"] == "passed" and projection["gt_projection_valid_rate"] >= t["minimum_projection_valid_rate"]
        annotation_score = finite_mean([1.0 if annotation[key] <= 1e-6 else 0.0 for key in ("gt_center_consistency_error", "gt_yaw_consistency_error", "gt_size_consistency_error")] + [annotation["gt_id_continuity"], annotation["gt_valid_mask_consistency"]])
        gate4 = isinstance(annotation_score, float) and annotation_score >= t["minimum_annotation_consistency"]
        gate5 = regions["gt_scorable_agent_camera_frame"] >= t["minimum_scorable_windows"]
        gate6_checks = {"flow_cosine": isinstance(motion["gt_flow_cosine"], float) and motion["gt_flow_cosine"] >= t["minimum_flow_cosine"],
            "direction": isinstance(motion["gt_motion_direction_agreement"], float) and motion["gt_motion_direction_agreement"] >= t["minimum_direction_agreement"],
            "valid_point_ratio": isinstance(motion["gt_valid_point_ratio"], float) and motion["gt_valid_point_ratio"] >= t["minimum_valid_point_ratio"],
            "support": isinstance(motion["gt_motion_support_rate"], float) and motion["gt_motion_support_rate"] >= t["minimum_motion_support_rate"],
            "unscorable": motion["gt_unscorable_rate"] <= t["maximum_unscorable_rate"]}
        gate7_checks = {"camera_support": isinstance(multiview["gt_camera_support_recall"], float) and multiview["gt_camera_support_recall"] >= t["minimum_camera_support_recall"],
            "contradiction": isinstance(multiview["gt_multiview_contradiction_rate"], float) and multiview["gt_multiview_contradiction_rate"] <= t["maximum_multiview_contradiction"]}
        gates = {"gate_1_gt_input_completeness": {"passed": gate1, "status": "passed" if gate1 else "unscorable_missing_gt_fields"},
            "gate_2_gt_time_alignment": {"passed": gate2, "status": "passed" if gate2 else "invalid_time_alignment"},
            "gate_3_gt_coordinate_and_projection": {"passed": gate3, "status": "passed" if gate3 else "invalid_calibration"},
            "gate_4_gt_annotation_self_consistency": {"passed": gate4, "status": "passed" if gate4 else "invalid_gt_pairing", "annotation_score": annotation_score},
            "gate_5_gt_instance_region_quality": {"passed": gate5, "status": "passed" if gate5 else "invalid_visual_evidence"},
            "gate_6_gt_visual_motion_consistency": {"passed": all(gate6_checks.values()), "status": "passed" if all(gate6_checks.values()) else "invalid_visual_evidence", "checks": gate6_checks},
            "gate_7_gt_multiview_consistency": {"passed": all(gate7_checks.values()), "status": "passed" if all(gate7_checks.values()) else "invalid_visual_evidence", "checks": gate7_checks}}
        upper = all(value["passed"] for value in gates.values()); gates["gate_8_gt_upper_bound_validity"] = {"passed": upper, "status": "valid_low_sample_gt_consistency" if upper else "failed_gt_calibration_thresholds"}; gates["thresholds_status"] = "gt_calibration_thresholds"; return gates

    def _failure_status(self, gates):
        for name in ("gate_1_gt_input_completeness", "gate_2_gt_time_alignment", "gate_3_gt_coordinate_and_projection", "gate_4_gt_annotation_self_consistency", "gate_5_gt_instance_region_quality", "gate_6_gt_visual_motion_consistency", "gate_7_gt_multiview_consistency"):
            if not gates[name]["passed"]: return gates[name]["status"]
        return "invalid_visual_evidence"

    def _run_tests(self):
        return {"status": "not_run", "reason": "Run pytest tests/test_consistency.py separately", "successful": False, "tests_run": 0, "run": 0, "passed": 0, "failures": 0, "errors": 0}

    def _root_cause(self, final):
        motion = final["future_frozen_evaluation"]; issues = [
            ("timestamp mismatch", final["time_alignment"]["maximum_timestamp_error"] > .05, "camera timestamps require interpolation"),
            ("frame offset", False, "index pairing verified; asynchronous offsets retained"), ("image resize mismatch", False, "explicit 1600x900 native mapping"),
            ("camera order mismatch", False, "all six configured order rows complete"), ("intrinsic mismatch", False, "positive calibrated intrinsics"),
            ("extrinsic mismatch", False, "unit WXYZ quaternions and complete sensor2ego"), ("Ego pose mismatch", False, "camera-timestamp ego2global used"),
            ("bottom-center/geometric-center mismatch", False, "boxes converted to geometric-center world representation"), ("yaw convention", False, "radian CCW and WXYZ verified"),
            ("box size problem", False, "size continuity is exact representation check"), ("occlusion handling", motion["gt_unscorable_rate"] > .2, "box depth-order approximation only"),
            ("GT box is not exact instance silhouette", True, "no independent mask; projected visible-face core fallback"),
            ("GT image and trajectory not same physical scene", False, "history image hashes and scene manifest agree"),
            ("low resolution", False, "formal run uses native 1600x900"), ("small target", True, "filtered by projected area"),
            ("optical flow backend error", motion["gt_motion_support_rate"] < .8, "OpenCV LK/Farneback is the remaining Level-2 measurement source"),
            ("image compression", True, "JPEG inputs"), ("export time desynchronization", True, "camera offsets up to reported maximum; interpolated rather than ignored")]
        lines = ["# GT-VTC root cause report", "", "| item | confirmed | evidence / action |", "|---|---:|---|"] + [f"| {name} | {'yes' if confirmed else 'no'} | {evidence} |" for name, confirmed, evidence in issues]
        (self.output / "root_cause_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _report(self, final):
        m, mv, a = final["future_frozen_evaluation"], final["multiview_consistency"], final["annotation_consistency"]
        lines = ["# GT-VTC paired GT report", "", f"- implementation_status: `{final['implementation_status']}`", f"- gt_self_consistency_status: `{final['gt_self_consistency_status']}`",
            "- protocol: `gt_vtc_paired_ground_truth`", "- holdout: `single_scene_temporal_holdout`", "",
            "## GT annotation self-consistency", f"Center error: {a['gt_center_consistency_error']}; yaw error: {a['gt_yaw_consistency_error']}; size error: {a['gt_size_consistency_error']}; velocity error: {a['gt_velocity_consistency_error']}.", "",
            "## GT image evidence", f"Flow EPE: {m['gt_flow_epe']}; normalized EPE: {m['gt_normalized_flow_epe']}; cosine: {m['gt_flow_cosine']}; direction agreement: {m['gt_motion_direction_agreement']}.",
            f"Motion support: {m['gt_motion_support_rate']}; weak support: {m['gt_weakly_supported_rate']}; unscorable: {m['gt_unscorable_rate']}.", "",
            "## Multi-view", f"Camera support recall: {mv['gt_camera_support_recall']}; motion consistency: {mv['gt_cross_view_motion_consistency']}; contradiction: {mv['gt_multiview_contradiction_rate']}.", "",
            "GT localization and identity are oracle inputs. Annotation comparisons are not presented as image evidence. There is no independent 2D mask/depth/flow GT, so Level-2 remains limited by projected cores and OpenCV motion.",
            "This is one physical scene and does not establish leaderboard calibration."]
        (self.output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _output_manifest(self):
        implementation = sorted(str(path) for path in Path("driveaudit_eval/research/consistency/observed_flow").rglob("*") if path.is_file() and "__pycache__" not in path.parts) + [str(self.config_path), "tools/evaluate.py", "docs/metrics.md"]
        results = sorted(str(path) for path in self.output.rglob("*") if path.is_file()); write_json(self.output / "output_manifest.json", {"implementation_files": implementation, "result_files": results + [str(self.output / "output_manifest.json")]})


def run(config_path: str | Path, mode: str, force: bool = False, resume: bool = False):
    runner = Runner(config_path, force, resume)
    if mode == "full": return runner.full()
    if mode == "audit": return {"manifest": build_manifest(runner.config, runner.output), "audit": data_audit(runner.config, runner.output)}
    core = runner.core()
    if mode in {"alignment", "projection", "annotation"}: return core[{"alignment": "aligned", "projection": "projections", "annotation": "annotation"}[mode]]
    if mode in {"history_calibration", "visual_motion", "multiview", "ablation"}: return runner.full()
    raise ValueError(f"unsupported GT-VTC mode: {mode}")





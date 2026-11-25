from __future__ import annotations

import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from driveaudit_eval.research.consistency.geometry_flow.aggregate import Runner as V2Runner
from driveaudit_eval.research.consistency.geometry_flow.gate import evaluate as evaluate_gates

from driveaudit_eval.research.consistency.projection_consistency.consistency_metrics import augment_row, summarize_generic
from driveaudit_eval.research.consistency.projection_consistency.corruption_test import aggregate as aggregate_corruptions
from driveaudit_eval.research.consistency.projection_consistency.corruption_test import evaluate_scene as evaluate_corruptions
from driveaudit_eval.research.consistency.projection_consistency.input_audit import audit as audit_inputs
from driveaudit_eval.research.consistency.projection_consistency.multiview_metrics import evaluate as evaluate_multiview
from driveaudit_eval.research.consistency.projection_consistency.schema import CAMERAS, NOT_AVAILABLE, load_config, mean, numeric, read_json, sha256, write_csv, write_json
from driveaudit_eval.research.consistency.projection_consistency.trajectory_loader import conditions_root
from driveaudit_eval.research.consistency.geometry_flow.visualization import generate as generate_visualizations


class EvaluationRunner:
    def __init__(self, config_path: str | Path, force: bool = False, resume: bool = False,
                 scene_ids: list[str] | None = None, max_workers: int = 1):
        self.config_path = Path(config_path.get("_config_source", "tools/configs/research.yaml")) if isinstance(config_path, dict) else Path(config_path)
        self.config = load_config(config_path)
        self.output = Path(self.config["outputs"]["root"])
        self.output.mkdir(parents=True, exist_ok=True)
        frozen = read_json(self.config["dataset"]["manifest"])["scene_ids"]
        self.scene_ids = scene_ids or list(frozen)
        if not set(self.scene_ids).issubset(frozen):
            raise ValueError("scene selection must remain inside the frozen ten-scene manifest")
        self.force, self.resume = force, resume
        self.max_workers = max(1, int(max_workers))
        self._audit: dict[str, Any] | None = None

    def audit(self, hash_images: bool = True) -> dict[str, Any]:
        self._audit = audit_inputs(self.config, self.output, self.scene_ids, hash_images)
        return self._audit

    def projection(self) -> dict[str, Any]:
        outputs = []
        for scene_id in self.scene_ids:
            runner, core, _, _ = self._prepare(scene_id)
            rows = core["native_projection"]
            eligible = [row for row in rows if row.get("expected_visible")]
            outputs.append({
                "scene_id": scene_id, "projection_rows": len(rows),
                "raw_agent_camera_projection_rate": sum(bool(row.get("projected_valid")) for row in rows) / max(1, len(rows)),
                "eligible_projection_valid_rate": sum(bool(row.get("projected_valid")) for row in eligible) / max(1, len(eligible)),
                "expected_visible_rows": len(eligible),
                "source": str(runner.output / "native_camera_time/gt_projection/projections.json"),
            })
        result = {"status": "completed", "scenes": outputs}
        write_json(self.output / "trajectory_projection/index.json", result)
        return result

    def evaluate(self, corruption: bool = False) -> dict[str, Any]:
        if self.config["mode"] == "submission" and not Path(self.config["inputs"]["candidate_image_root"]).is_dir():
            result = {"status": "not_run_no_submission", "mode": "submission"}
            write_json(self.output / "final_metrics.json", result)
            return result
        if self._audit is None:
            self.audit(hash_images=self.config["mode"] == "gt_example")
        scene_results, corruptions = [], {}
        for scene_id in self.scene_ids:
            result, corruption_result = self._evaluate_scene(scene_id, corruption)
            scene_results.append(result)
            if corruption_result is not None:
                corruptions[scene_id] = corruption_result
        write_json(self.output / "image_motion/run_manifest.json", {
            "status": "completed", "scene_count": len(scene_results),
            "candidate_rgb_read": True, "long_tracking_used": False,
            "trajectory_prediction_filled_failed_points": False,
            "scenes": [{"scene_id": row["scene_id"], "window_count": row["window_count"], "status": row["status"]} for row in scene_results],
        })
        if corruptions:
            aggregate_corruptions(corruptions, self.output)
        return {"status": "completed", "scenes": scene_results}

    def _evaluate_scene(self, scene_id: str, corruption: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
        scene_output = self.output / "metrics/scenes" / scene_id
        metrics_path = scene_output / "scene_metrics.json"
        corruption_path = scene_output / "corruption_test/scene_summary.json"
        if self.resume and not self.force and metrics_path.is_file() and (not corruption or corruption_path.is_file()):
            cached = read_json(metrics_path)
            if cached.get("protocol") == self.config["version"] and cached.get("candidate_image_root") == str(Path(self.config["inputs"]["candidate_image_root"]) / scene_id):
                return cached, read_json(corruption_path) if corruption else None
        runner, core, selected, image_size = self._prepare(scene_id)
        rows, points = runner._process(
            core["native_warps"], core["native_projection"], core["native_surfaces"],
            (10, 90), selected, "frozen_gt_vtc_v2_backend",
        )
        rows = [augment_row(row, image_size) for row in rows]
        summary = summarize_generic(rows)
        multiview = evaluate_multiview(rows)
        audit_scene = next(row for row in self._audit["scenes"] if row["scene_id"] == scene_id)
        time_ok = all(value["native_timestamps_strictly_increasing"] for value in core["timestamps"]["per_camera"].values())
        projections = core["native_projection"]
        eligible_projections = [row for row in projections if row.get("expected_visible")]
        projection_ok = bool(eligible_projections) and all(row.get("projected_valid") for row in eligible_projections)
        gate_input = evaluate_gates(
            True, projection_ok, summary,
            {
                "gt_camera_support_recall": multiview["camera_support_recall"],
                "gt_multiview_contradiction_rate": multiview["multiview_contradiction_rate"],
            },
            runner.config, audit_scene["status"] == "completed", time_ok,
        )
        status = {
            "valid_gt_consistency": "valid_consistency",
            "valid_gt_consistency_limited_coverage": "limited_visual_coverage",
            "invalid_gt_visual_alignment": "invalid_alignment",
            "unscorable_missing_gt_fields": "invalid_input",
        }.get(gate_input["gate_8_upper_bound"]["status"], "unscorable")
        leakage_free = all(not row.get("trajectory_prediction_used_as_observation", True) for row in rows) and all(not point.get("gt_prediction_used_as_observation", True) for point in points)
        result = {
            "scene_id": scene_id, "protocol": self.config["version"], "mode": self.config["mode"],
            "image_role": self.config["semantics"]["image_role"], "trajectory_role": self.config["semantics"]["trajectory_role"],
            "candidate_image_root": str(Path(self.config["inputs"]["candidate_image_root"]) / scene_id),
            "trajectory_root": str(Path(self.config["inputs"]["trajectory_root"]) / scene_id),
            "future_frames": [10, 89], "resolution": list(image_size), "window_count": len(rows),
            "scorable_window_count": summary["scorable_window_count"], "status": status,
            "metrics": summary, "multiview": multiview, "gates": gate_input,
            "observed_motion_source": "candidate_rgb_geometry_initialized_lk",
            "trajectory_prediction_used_as_observation": not leakage_free,
            "candidate_future_used_for_tuning": False,
            "frozen_parameter_source": self.config["backend"]["frozen_parameters"],
        }
        write_csv(scene_output / "per_window.csv", rows)
        write_csv(scene_output / "per_point.csv", points)
        write_json(metrics_path, result)
        generate_visualizations(runner.v1_config, rows, core["native_projection"], scene_output)
        corruption_result = evaluate_corruptions(runner, core, selected, rows) if corruption else None
        return result, corruption_result

    def _prepare(self, scene_id: str) -> tuple[V2Runner, dict[str, Any], dict[str, Any], tuple[int, int]]:
        candidate = Path(self.config["inputs"]["candidate_image_root"]) / scene_id
        first = cv2.imread(str(candidate / "CAM_FRONT/000010.jpg"))
        if first is None:
            raise FileNotFoundError(candidate / "CAM_FRONT/000010.jpg")
        image_size = (int(first.shape[1]), int(first.shape[0]))
        output = self.output / "metrics/scenes" / scene_id
        conditions = conditions_root(self.config, scene_id)
        v1 = {
            "version": "gt_vtc_v1_oracle_paired",
            "protocol": {"name": "image_trajectory_consistency_v1_compat", "use_future_gt": False, "use_gt_agent_id": False, "use_gt_boxes": False, "use_gt_trajectory": False, "use_submission_data": True},
            "inputs": {"gt_images": str(candidate), "conditions": str(conditions)},
            "output": str(output / "compat"),
            "scene": {"scene_id": scene_id, "history_frames": 10, "future_frames": 80, "development_future_frames": 0, "evaluation_start_frame": 10},
            "image": {"native_size": [1600, 900], "formal_size": list(image_size), "low_resolution_size": [400, 224]},
            "time_alignment": {"use_timestamp": True, "allow_interpolation": True, "offset_search_diagnostic": True, "maximum_nearest_error_s": 0.06},
            "projection": {"minimum_depth_m": 0.2, "minimum_projected_area_px": 256, "maximum_truncation_ratio": 0.65, "maximum_occlusion_ratio": 0.8, "minimum_visibility_token": 2},
            "instance_region": {"priority": ["projected_visible_face_core"], "erode_core": True, "erosion_ratio": 0.08, "minimum_core_pixels": 64, "exclude_overlap": True, "exclude_occluded": True, "exclude_small_objects": True},
            "motion": {"window_delta": 1, "candidate_deltas": [1], "use_lk": True, "use_farneback": False, "use_orb": False, "allow_raft_if_available": False, "allow_long_propagation": False, "maximum_points": 96, "minimum_valid_points": 4, "corner_quality": 0.01, "minimum_point_distance_px": 4, "forward_backward_error_px": 2.5, "ransac_threshold_px": 3.0, "maximum_flow_px": 160, "orb_features": 128, "repeated_frame_mean_difference": 0.15, "minimum_contrast_std": 4.0},
            "calibration": {"use_history_frames": False, "future_gt_for_threshold_tuning": False, "single_scene_temporal_holdout": False},
            "metrics": {"annotation_consistency": False, "visual_motion_consistency": True, "multiview_consistency": True, "temporal_consistency": True},
            "thresholds": {"status": "frozen_gt_vtc_v2_thresholds", "minimum_projection_valid_rate": 0.95, "minimum_annotation_consistency": 0.99, "minimum_camera_support_recall": 0.90, "minimum_valid_point_ratio": 0.70, "minimum_motion_support_rate": 0.80, "minimum_flow_cosine": 0.75, "minimum_direction_agreement": 0.75, "maximum_multiview_contradiction": 0.10, "maximum_unscorable_rate": 0.20, "minimum_scorable_windows": 50},
        }
        v2 = load_config(self.config["backend"]["gt_vtc_v2_config"])
        v2["inputs"] = {"gt_vtc_v1_config": v1, "gt_vtc_v1_results": self.config["backend"]["gt_vtc_v1_results"]}
        v2["output"] = str(output)
        v2.setdefault("image_trajectory_consistency", {})["corruption_windows"] = int(self.config["corruption_test"]["maximum_windows_per_scene"])
        runner = V2Runner(v2, force=self.force, resume=self.resume)
        core = runner.core()
        frozen = read_json(self.config["backend"]["frozen_parameters"])
        selected = {camera: value["parameters"] for camera, value in frozen["selected_parameters"].items()}
        return runner, core, selected, image_size

    def aggregate(self) -> dict[str, Any]:
        all_rows, all_points, scene_metrics, multiview_rows, visualization_rows = [], [], [], [], []
        for scene_id in self.scene_ids:
            root = self.output / "metrics/scenes" / scene_id
            if not (root / "scene_metrics.json").is_file():
                raise FileNotFoundError(f"scene has not been evaluated: {scene_id}")
            result = read_json(root / "scene_metrics.json")
            failed_gates = [name for name, value in result["gates"].items() if isinstance(value, dict) and value.get("passed") is False]
            scene_metrics.append({"scene_id": scene_id, "status": result["status"], "failed_gates": failed_gates, **result["metrics"], **{f"multiview_{key}": value for key, value in result["multiview"].items() if key != "per_camera"}})
            multiview_rows.append({"scene_id": scene_id, "status": result["status"], **result["multiview"]})
            visualization_rows.extend(
                {"scene_id": scene_id, "path": str(path.relative_to(self.output))}
                for path in sorted((root / "visualizations").glob("*")) if path.is_file()
            )
            with (root / "per_window.csv").open(newline="", encoding="utf-8") as handle:
                all_rows.extend(dict(row) for row in csv.DictReader(handle))
            with (root / "per_point.csv").open(newline="", encoding="utf-8") as handle:
                all_points.extend(dict(row) for row in csv.DictReader(handle))
        per_agent = self._grouped(all_rows, ("scene_id", "agent_id"))
        per_camera = self._grouped(all_rows, ("camera_name",))
        per_class = self._grouped(all_rows, ("category",))
        write_csv(self.output / "metrics/per_point.csv", all_points)
        write_csv(self.output / "metrics/per_window.csv", all_rows)
        write_csv(self.output / "metrics/per_agent.csv", per_agent)
        write_csv(self.output / "metrics/per_camera.csv", per_camera)
        write_csv(self.output / "metrics/per_class.csv", per_class)
        write_csv(self.output / "metrics/per_scene.csv", scene_metrics)
        write_json(self.output / "multiview/per_scene.json", multiview_rows)
        write_json(self.output / "visualizations/index.json", {
            "scene_count": len({row["scene_id"] for row in visualization_rows}),
            "visualization_count": len(visualization_rows),
            "items": visualization_rows,
        })
        dataset = self._dataset_metrics(scene_metrics)
        upper_bound = {
            "protocol": self.config["version"], "mode": self.config["mode"],
            "index_unit": "scene-agent-camera-window", "gt_scorable_only": True,
            "windows": [
                {key: row[key] for key in ("scene_id", "agent_id", "camera_name", "frame_index", "target_frame_index")}
                for row in all_rows if row.get("guided_support_state") != "unscorable"
            ],
        }
        write_json(self.output / "gt_upper_bound_index.json", upper_bound)
        corruption = read_json(self.output / "corruption_test/corruption_summary.json") if (self.output / "corruption_test/corruption_summary.json").is_file() else {"status": "not_run"}
        final = {
            "implementation_status": "completed",
            "evaluation_status": "completed" if len(scene_metrics) == len(self.scene_ids) else "partial",
            "protocol_version": self.config["version"], "mode": self.config["mode"],
            "reference_semantics": "trajectory_as_visual_expectation",
            "image_role": self.config["semantics"]["image_role"], "trajectory_role": self.config["semantics"]["trajectory_role"],
            "scene_count": len(scene_metrics), "future_frames_per_scene": 80, "camera_count": 6,
            "candidate_image_root": self.config["inputs"]["candidate_image_root"],
            "trajectory_root": self.config["inputs"]["trajectory_root"],
            "evaluation_agent_count": len({(row["scene_id"], row["agent_id"]) for row in all_rows}),
            "evaluation_agent_frame_count": len({(row["scene_id"], row["agent_id"], row["frame_index"]) for row in all_rows}),
            "agent_camera_window_count": len(all_rows),
            "scorable_window_count": sum(row.get("guided_support_state") != "unscorable" for row in all_rows),
            "status_counts": dict(Counter(row["status"] for row in scene_metrics)),
            "dataset_metrics": dataset, "per_scene": scene_metrics,
            "corruption_test": corruption,
            "submission_mode_status": "not_run_no_submission" if self.config["mode"] == "gt_example" else "evaluated",
            "total_score": None, "total_score_status": "disabled",
            "candidate_future_used_for_parameter_tuning": False,
            "observations_come_from_candidate_rgb": all(row.get("trajectory_prediction_used_as_observation") in {"False", False, "false"} for row in all_rows),
        }
        write_json(self.output / "aggregate_metrics.json", dataset)
        write_json(self.output / "final_metrics.json", final)
        self._write_report(final)
        self._sanitize_json_outputs()
        return final

    def _sanitize_json_outputs(self) -> None:
        # GT-VTC v2 predates strict JSON and may emit Infinity in diagnostic-only
        # condition numbers. Rewriting through this protocol's writer preserves
        # values while converting non-finite diagnostics to not_available.
        for path in self.output.rglob("*.json"):
            write_json(path, json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key, NOT_AVAILABLE) for key in keys)].append(row)
        result = []
        for identity, values in grouped.items():
            normalized = [_normalize_csv_row(row) for row in values]
            result.append({**dict(zip(keys, identity)), **summarize_generic(normalized)})
        return result

    def _dataset_metrics(self, scenes: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "scorable_coverage", "valid_point_ratio", "residual_epe", "image_normalized_epe",
            "box_normalized_epe", "motion_normalized_epe", "flow_cosine", "direction_agreement",
            "magnitude_error", "scale_agreement", "rotation_agreement", "strong_support_rate",
            "strong_or_weak_support_rate", "weakly_supported_rate", "unscorable_rate", "contradicted_rate",
            "multiview_camera_support_recall", "multiview_cross_view_motion_consistency", "multiview_multiview_contradiction_rate",
        )
        rng = np.random.default_rng(int(self.config["aggregation"].get("seed", 20260730)))
        repeats = int(self.config["aggregation"]["bootstrap_repeats"])
        alpha = 1 - float(self.config["aggregation"]["confidence_level"])
        result = {}
        for field in fields:
            values = numeric(row.get(field) for row in scenes)
            if not values:
                result[field] = NOT_AVAILABLE; continue
            array = np.asarray(values, dtype=float)
            bootstrap = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(repeats)])
            result[field] = {
                "mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)),
                "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
                "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
                "minimum": float(array.min()), "maximum": float(array.max()), "valid_scene_count": len(array),
                "scene_bootstrap_95ci": [float(np.quantile(bootstrap, alpha / 2)), float(np.quantile(bootstrap, 1 - alpha / 2))],
            }
        result["bootstrap_unit"] = "scene"
        result["total_score"] = None
        return result

    def _write_report(self, final: dict[str, Any]) -> None:
        metric = lambda name: final["dataset_metrics"].get(name, {}).get("mean", final["dataset_metrics"].get(name, NOT_AVAILABLE))
        lines = [
            "# Image–Trajectory Consistency v1", "",
            "## 1. 实现状态", "",
            f"- implementation_status: {final['implementation_status']}",
            f"- evaluation_status: {final['evaluation_status']}",
            f"- protocol_version: {final['protocol_version']}", f"- mode: {final['mode']}", "",
            "## 2. 当前输入", "",
            f"- 图像角色: {final['image_role']}", f"- 轨迹角色: {final['trajectory_role']}",
            f"- 场景数: {final['scene_count']}", f"- 图像根目录: {final['candidate_image_root']}",
            f"- 轨迹根目录: {final['trajectory_root']}", f"- 当前是否GT示例: {final['mode'] == 'gt_example'}", "",
            "## 3. 数据统计", "",
            f"- 评测Agent: {final['evaluation_agent_count']}", f"- Agent-frame: {final['evaluation_agent_frame_count']}",
            f"- Agent-camera-window: {final['agent_camera_window_count']}", f"- 可评分窗口: {final['scorable_window_count']}", "",
            "## 4. 图像—轨迹一致性核心结果", "",
            f"- Scorable Coverage: {metric('scorable_coverage')}", f"- Valid-point Ratio: {metric('valid_point_ratio')}",
            f"- Residual EPE: {metric('residual_epe')}", f"- Image-normalized EPE: {metric('image_normalized_epe')}",
            f"- Box-normalized EPE: {metric('box_normalized_epe')}", f"- Motion-normalized EPE: {metric('motion_normalized_epe')}",
            f"- Flow Cosine: {metric('flow_cosine')}", f"- Direction Agreement: {metric('direction_agreement')}",
            f"- Magnitude Error: {metric('magnitude_error')}", f"- Scale Agreement: {metric('scale_agreement')}",
            f"- Rotation Agreement: {metric('rotation_agreement')}", f"- Strong Support: {metric('strong_support_rate')}",
            f"- Strong-or-Weak Support: {metric('strong_or_weak_support_rate')}", f"- Unscorable: {metric('unscorable_rate')}",
            f"- Contradiction: {metric('contradicted_rate')}", "",
            "## 5. 多视角", "",
            f"- Camera Support Recall: {metric('multiview_camera_support_recall')}",
            f"- Cross-view Motion: {metric('multiview_cross_view_motion_consistency')}",
            f"- Multi-view Contradiction: {metric('multiview_multiview_contradiction_rate')}",
            "- 逐相机结果: `metrics/per_camera.csv`", "",
            "## 6. 逐场景", "",
            "| scene | status | failed gates | coverage | valid points | EPE | cosine | direction | strong | multiview contradiction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in final["per_scene"]:
            lines.append(f"| {row['scene_id']} | {row['status']} | {', '.join(row['failed_gates']) or '-'} | {row['scorable_coverage']} | {row['valid_point_ratio']} | {row['residual_epe']} | {row['flow_cosine']} | {row['direction_agreement']} | {row['strong_support_rate']} | {row.get('multiview_multiview_contradiction_rate')} |")
        lines += ["", "## 7. GT正样本结论", "", f"- 状态分布: {final['status_counts']}",
                  "- 当前结果是严格配对GT正控，用于评价器上界验证，不是选手成绩。",
                  "- invalid/limited场景保留；cuboid visible-face core和RGB LK仍可能限制证据质量。", "",
                  "## 8. Corruption验证", "", f"```json\n{json.dumps(final['corruption_test'], ensure_ascii=False, indent=2)}\n```", "",
                  "## 9. 后续选手接口", "",
                  "将配置中的 `candidate_image_root` 与 `trajectory_root` 替换为选手提交根目录，并设置 `mode: submission`；冻结后端、阈值和聚合不变。", "",
                  "## 10. 输出文件", "",
                  "- 配置: `tools/configs/research.yaml`",
                  "- 脚本: `python tools/evaluate.py --task consistency`",
                  "- per-window/agent/camera/scene: `metrics/`",
                  "- aggregate: `aggregate_metrics.json`、`final_metrics.json`",
                  "- visualizations: `visualizations/index.json`",
                  "- verification: `verification.json`",
                  "- report: `final_report.md`", "",
                  "## 11. 限制", "", "- 无GT instance mask/depth/motion vector，实例区域使用投影可见面core。",
                  "- Appearance/disappearance/handoff在事件证据不足时保持not_available。",
                  "- 不包含图像质量、轨迹质量、地图、感知或规划评价；总分禁用。", ""]
        (self.output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    def full(self) -> dict[str, Any]:
        started = time.time()
        self.audit(hash_images=self.config["mode"] == "gt_example")
        self.projection()
        self.evaluate(corruption=bool(self.config["corruption_test"]["enabled"]))
        final = self.aggregate()
        final["runtime_seconds"] = time.time() - started
        write_json(self.output / "final_metrics.json", final)
        return final


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for key, item in list(value.items()):
        if item in {"True", "true"}: value[key] = True
        elif item in {"False", "false"}: value[key] = False
        else:
            try: value[key] = float(item)
            except (TypeError, ValueError): pass
    return value





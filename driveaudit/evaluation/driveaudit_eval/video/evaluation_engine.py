from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from driveaudit_eval.video.manifest import _resolve_path, _resolve_optional_path_string, resolve_config_paths, _json_hash, _file_signature, _path_list_signature, build_dataset_index, enrich_manifest_with_locks, enforce_manifest_lock
from driveaudit_eval.video.reporting import _markdown, _write_tables
from driveaudit_eval.video.evaluation_graph import EvaluationGraph
from driveaudit_eval.video.distribution import FVD_FEATURE_DIM, FVD_PROTOCOL, FVD_OUTPUT_KIND, FVD_REFERENCE_WEIGHTS_SHA256
from driveaudit_eval.video.execution_binding import ExecutionBindingLayer
from driveaudit_eval.video.io import SceneSample, discover_samples, samples_from_sources, validate_sample
from driveaudit_eval.video.metrics.epipolar_metric import EpipolarMetric
from driveaudit_eval.video.metrics.fvd_metric import FVDMetric
from driveaudit_eval.video.metrics.loftr_multiview_metric import LoFTRMultiviewMetric
from driveaudit_eval.video.runtime_state import RuntimeState, set_reproducibility
from driveaudit_eval.video.sampling import build_evaluation_manifest, build_sampling_plan
from driveaudit_eval.video.utils import assert_no_forbidden_keys, ensure_dir, write_csv, write_json
from driveaudit_eval.video.validators import audit_markdown, build_metric_audit, validate_summary


BENCHMARK_VERSION = "2.0"
from driveaudit_eval.video.manifest import BENCHMARK_PROTOCOL


def _metric_disabled(name: str) -> Dict[str, Any]:
    return {
        "enabled": False,
        "skipped": True,
        "reason": "disabled",
        "backend": "disabled",
        "metric_confidence": "disabled",
        "metric_validity": "disabled",
        "comparison_validity": "invalid",
        "metric_name": name,
    }




















def _dataset_statistics(samples: Sequence[SceneSample], plan) -> Dict[str, Any]:
    return {
        "num_scenes": len(samples),
        "scene_ids": [sample.scene_id for sample in samples],
        "cameras": plan.cameras,
        "history_frame_counts": {
            f"{sample.scene_id}/{camera}": len(sample.history_images.get(camera, []))
            for sample in samples
            for camera in plan.cameras
        },
        "future_frame_counts": {
            f"{sample.scene_id}/{camera}": len(sample.future_images.get(camera, []))
            for sample in samples
            for camera in plan.cameras
        },
        "real_reference_frame_counts": {
            f"{sample.scene_id}/{camera}": len(sample.real_reference_images.get(camera, []))
            for sample in samples
            for camera in plan.cameras
        },
        "sampled_real_images": len(plan.real_images),
        "sampled_fake_images": len(plan.fake_images),
        "sampled_real_clips": len(plan.real_clips),
        "sampled_fake_clips": len(plan.fake_clips),
    }


def _first_clip_positions(plan, scene_id: str) -> List[int]:
    per_camera = plan.scene_clip_starts.get(scene_id, {})
    starts = [starts[0] for starts in per_camera.values() if starts]
    if starts:
        start = min(starts)
        return list(range(start, start + plan.clip_length))
    return plan.scene_frame_positions.get(scene_id, [])[: plan.clip_length]


def build_consistency_samples(samples: Sequence[SceneSample], plan, frame_count: Optional[int] = None) -> List[SceneSample]:
    ret: List[SceneSample] = []
    for sample in samples:
        positions = _first_clip_positions(plan, sample.scene_id)
        if frame_count is not None and frame_count > 0:
            available = [len(sample.future_images.get(camera, [])) for camera in plan.cameras]
            positions = list(range(min([frame_count, *available]))) if available else []
        if not positions:
            continue
        future_images = {}
        for camera in plan.cameras:
            paths = sample.future_images.get(camera, [])
            future_images[camera] = [paths[pos] for pos in positions if pos < len(paths)]
        ret.append(replace(sample, cameras=list(plan.cameras), future_images=future_images))
    return ret






class EvaluationEngine:
    def __init__(self, cfg: Dict[str, Any], config_dir: Path = Path.cwd()) -> None:
        self.cfg = deepcopy(cfg)
        if self.cfg.get('output', {}).get('retention') == 'metrics':
            self.cfg['output'].update(save_cache=False, save_visualizations=False, save_curves=False)
        self.config_dir = Path(config_dir).resolve()
        resolve_config_paths(self.cfg, self.config_dir)
        paths_cfg = self.cfg.get("paths", {})
        output_dir = _resolve_path(paths_cfg.get("output_dir", "outputs/eval_results"), self.config_dir)
        if output_dir is None:
            raise ValueError("paths.output_dir is required")
        ensure_dir(output_dir)
        self.output_dir = output_dir
        self.cfg.setdefault("paths", {})["output_dir"] = str(output_dir)
        self.state = RuntimeState(self.cfg, output_dir)
        self.graph = EvaluationGraph()

    def run(self) -> Dict[str, Any]:
        self.state.set_deterministic()
        self.load_dataset_manifest()
        self.build_execution_graph()
        self.state.graph = self.graph.describe()
        self.graph.execute(self.state)
        return self.state.results["summary"]

    def load_dataset_manifest(self) -> Dict[str, Any]:
        paths_cfg = self.cfg.get("paths", {})
        dataset_root = _resolve_path(paths_cfg.get("dataset_root"), self.config_dir)
        generated_root = _resolve_path(paths_cfg.get("generated_root"), self.config_dir)
        real_reference_root = _resolve_path(paths_cfg.get("real_reference_root"), self.config_dir)
        history_root = _resolve_path(paths_cfg.get("history_root"), self.config_dir)
        metadata_root = _resolve_path(paths_cfg.get("metadata_root"), self.config_dir)
        evaluation_manifest_path = _resolve_path(paths_cfg.get("evaluation_manifest"), self.config_dir)
        self.cfg["paths"].update(
            {
                "dataset_root": str(dataset_root) if dataset_root else None,
                "generated_root": str(generated_root) if generated_root else None,
                "real_reference_root": str(real_reference_root) if real_reference_root else None,
                "history_root": str(history_root) if history_root else None,
                "metadata_root": str(metadata_root) if metadata_root else None,
                "evaluation_manifest": str(evaluation_manifest_path) if evaluation_manifest_path else None,
            }
        )
        if generated_root is None:
            raise ValueError("paths.generated_root is required")
        needs_reference = (
            self.cfg.get("metrics", {}).get("fid", {}).get("enabled", True)
            or self.cfg.get("metrics", {}).get("fvd", {}).get("enabled", True)
        )
        if real_reference_root is None and needs_reference:
            raise ValueError("paths.real_reference_root is required for dataset-level FID/FVD.")
        samples = samples_from_sources(self.cfg['scene_sources'], self.cfg.get('dataset', {}).get('cameras', [])) if 'scene_sources' in self.cfg else discover_samples(
            dataset_root=dataset_root,
            generated_root=generated_root,
            cameras=self.cfg.get("dataset", {}).get("cameras", []),
            metadata_root=metadata_root,
            real_reference_root=real_reference_root,
            history_root=history_root,
            max_scenes=self.cfg.get("dataset", {}).get("max_scenes"),
        )
        samples = sorted(samples, key=lambda item: item.scene_id)
        if 'scene_sources' in self.cfg and self.cfg.get('dataset', {}).get('max_scenes') is not None:
            samples = samples[:self.cfg['dataset']['max_scenes']]
        if not samples:
            raise FileNotFoundError(f"No generated samples found under {generated_root}")
        for sample in samples:
            for message in validate_sample(sample):
                self.state.warnings.add("io", "sample_validation", message, scene_id=sample.scene_id)
        dataset_index = build_dataset_index(samples, self.cfg.get("dataset", {}).get("cameras", []))
        self.state.samples = samples
        self.state.samples_by_id = {sample.scene_id: sample for sample in samples}
        self.state.dataset_index = dataset_index
        self.state.dataset_index_order = list(dataset_index["scene_order"])
        self.state.dataset_version_hash = _json_hash(dataset_index)
        self.state.scene_order_hash = dataset_index["scene_order_hash"]
        if evaluation_manifest_path is not None and evaluation_manifest_path.exists():
            manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
            enforce_manifest_lock(manifest, dataset_index)
        else:
            manifest = build_evaluation_manifest(samples, self.cfg)
            manifest = enrich_manifest_with_locks(manifest, dataset_index)
            if evaluation_manifest_path is not None:
                write_json(evaluation_manifest_path, manifest)
        self.state.manifest = manifest
        write_json(self.output_dir / "evaluation_manifest.json", manifest)
        return manifest

    def build_execution_graph(self) -> EvaluationGraph:
        self.graph = EvaluationGraph()
        self.graph.add_node("sampling", self.run_sampling, kind="sampling", outputs=("sampling_plan",))
        self.graph.add_node(
            "binding_layer.initialize",
            self.run_binding_layer,
            depends_on=("sampling",),
            kind="binding_layer",
            outputs=("execution_binding",),
        )
        if self.cfg.get("metrics", {}).get("fid", {}).get("enabled", True):
            self.graph.add_node(
                "preprocessing.fid.inception_299_imagenet",
                self.run_fid_preprocessing,
                depends_on=("binding_layer.initialize",),
                kind="preprocessing",
                outputs=("fid.preprocessing_plan",),
            )
            self.graph.add_node(
                "features.fid.binding_layer_inception_forward",
                self.run_fid_feature_extractor,
                depends_on=("preprocessing.fid.inception_299_imagenet",),
                kind="feature_extraction",
                outputs=("fid.real_features", "fid.fake_features"),
            )
            self.graph.add_node(
                "metrics.fid.frechet_distance",
                self.run_fid_metric,
                depends_on=("features.fid.binding_layer_inception_forward",),
                kind="metric",
                outputs=("fid",),
            )
        else:
            self.state.results["fid"] = _metric_disabled("fid")
        if self.cfg.get("metrics", {}).get("fvd", {}).get("enabled", True):
            self.graph.add_node(
                "preprocessing.fvd.i3d_clips",
                self.run_fvd_preprocessing,
                depends_on=("binding_layer.initialize",),
                kind="preprocessing",
                outputs=("fvd.preprocessing_plan",),
            )
            self.graph.add_node(
                "features.fvd.binding_layer_i3d_forward",
                self.run_fvd_feature_extractor,
                depends_on=("preprocessing.fvd.i3d_clips",),
                kind="feature_extraction",
                outputs=("fvd.real_features", "fvd.fake_features"),
            )
            self.graph.add_node(
                "metrics.fvd.frechet_distance",
                self.run_fvd_metric,
                depends_on=("features.fvd.binding_layer_i3d_forward",),
                kind="metric",
                outputs=("fvd",),
            )
        else:
            self.state.results["fvd"] = _metric_disabled("fvd")
        if self.cfg.get("metrics", {}).get("temporal_warp", {}).get("enabled", True):
            self.graph.add_node(
                "metrics.temporal_consistency",
                self.run_temporal_consistency,
                depends_on=("sampling",),
                kind="metric",
                outputs=("temporal_consistency",),
            )
        else:
            self.state.results["temporal_consistency"] = _metric_disabled("temporal_consistency")
        if self.cfg.get("metrics", {}).get("history_future", {}).get("enabled", True):
            self.graph.add_node(
                "metrics.history_future_continuity",
                self.run_history_future_continuity,
                depends_on=("sampling",),
                kind="metric",
                outputs=("history_future_continuity",),
            )
        else:
            self.state.results["history_future_continuity"] = _metric_disabled("history_future_continuity")
        if self.cfg.get("metrics", {}).get("loftr", {}).get("enabled", True):
            self.graph.add_node(
                "metrics.multi_view_consistency",
                self.run_multi_view_consistency,
                depends_on=("sampling",),
                kind="metric",
                outputs=("multi_view_consistency", "loftr_matches"),
            )
        else:
            self.state.results["multi_view_consistency"] = _metric_disabled("multi_view_consistency")
        if self.cfg.get("metrics", {}).get("epipolar", {}).get("enabled", True):
            deps = ["sampling"]
            if self.cfg.get("metrics", {}).get("loftr", {}).get("enabled", True):
                deps.append("metrics.multi_view_consistency")
            self.graph.add_node(
                "metrics.geometry_consistency",
                self.run_geometry_consistency,
                depends_on=tuple(deps),
                kind="metric",
                outputs=("geometry_consistency",),
            )
        else:
            self.state.results["geometry_consistency"] = _metric_disabled("geometry_consistency")
        if any(item.get('enabled', False) for item in self.state.cfg.get('visual_quality', {}).values() if isinstance(item, dict)):
            self.graph.add_node('metrics.visual_quality', self.run_visual_quality,
                                depends_on=('sampling', 'binding_layer.initialize'), kind='metric', outputs=('visual_quality',))
        metric_nodes = [name for name, node in self.graph.nodes.items() if node.kind == "metric"]
        self.graph.add_node(
            "aggregation",
            self.aggregate_results,
            depends_on=tuple(metric_nodes),
            kind="aggregation",
            outputs=("results.json", "report.md"),
        )
        return self.graph

    def run_visual_quality(self, state: RuntimeState) -> None:
        from driveaudit_eval.video.visual_quality import evaluate
        state.results['visual_quality'] = evaluate(state, self._binding(state))

    def run_sampling(self, state: RuntimeState) -> None:
        plan = build_sampling_plan(state.samples, state.cfg, manifest=state.manifest)
        state.plan = plan
        state.consistency_samples = build_consistency_samples(
            state.samples, plan, state.cfg.get("sampling", {}).get("consistency_frames")
        )
        state.dataset_statistics = _dataset_statistics(state.samples, plan)
        write_json(state.output_dir / "sampling_manifest.json", plan.summary())

    def run_binding_layer(self, state: RuntimeState) -> None:
        state.shared["execution_binding"] = ExecutionBindingLayer(state, state.manifest)
        if state.execution_trace:
            state.execution_trace[-1]["details"] = {
                "binding_layer": "ExecutionBindingLayer",
                "manifest_version": state.manifest.get("version"),
                "dataset_version_hash": state.manifest.get("dataset_version_hash"),
                "executor_access_policy": "engine_calls_binding_only",
            }

    def _binding(self, state: RuntimeState) -> ExecutionBindingLayer:
        binding = state.shared.get("execution_binding")
        if not isinstance(binding, ExecutionBindingLayer):
            raise RuntimeError("ExecutionBindingLayer is not initialized by the DAG.")
        return binding

    def run_feature_extractors(self, state: RuntimeState) -> None:
        raise RuntimeError("Feature extraction is graph-only; use EvaluationEngine.run().")

    def run_fid_preprocessing(self, state: RuntimeState) -> None:
        plan = state.plan
        if plan is None:
            raise RuntimeError("sampling plan is not available")
        binding = self._binding(state)
        state.preprocessing["fid"] = {
            "real": binding.prepare_fid_images(plan.real_images, "real"),
            "fake": binding.prepare_fid_images(plan.fake_images, "fake"),
        }
        state.execution_trace[-1]["details"] = {
            "binding_layer": "ExecutionBindingLayer.prepare_fid_images",
            "executor": "InceptionFIDExecutor",
            "preprocessing": "torch_bilinear_299_align_corners_false_rgb_minus1_plus1",
            "real_items": len(plan.real_images),
            "fake_items": len(plan.fake_images),
        }

    def run_fvd_preprocessing(self, state: RuntimeState) -> None:
        plan = state.plan
        if plan is None:
            raise RuntimeError("sampling plan is not available")
        binding = self._binding(state)
        state.preprocessing["fvd"] = {
            "real": binding.prepare_fvd_clips(plan.real_clips, "real"),
            "fake": binding.prepare_fvd_clips(plan.fake_clips, "fake"),
        }
        state.execution_trace[-1]["details"] = {
            "binding_layer": "ExecutionBindingLayer.prepare_fvd_clips",
            "executor": "I3DFVDExecutor",
            "preprocessing": "native_rgb_BCTHW_float_0_255_reference_rescale_and_resize",
            "real_items": len(plan.real_clips),
            "fake_items": len(plan.fake_clips),
            "clip_length": int(plan.clip_length),
            "stride": int(plan.stride),
        }

    def run_fid_feature_extractor(self, state: RuntimeState) -> None:
        plan = state.plan
        if plan is None:
            raise RuntimeError("sampling plan is not available")
        if len(plan.real_images) < 2 or len(plan.fake_images) < 2:
            raise ValueError("Dataset-level FID needs at least two sampled real and fake images.")
        binding = self._binding(state)
        real_features, real_audit = binding.run_fid_features(plan.real_images, "real")
        fake_features, fake_audit = binding.run_fid_features(plan.fake_images, "fake")
        binding.validate_fid_trace()
        real_provenance = real_audit["provenance"]
        fake_provenance = fake_audit["provenance"]
        for key in ("backend", "protocol", "checkpoint_sha256"):
            if real_provenance[key] != fake_provenance[key]:
                raise RuntimeError(f"Real/fake FID feature provenance mismatch: {key}")
        state.features["fid"] = {
            "real": real_features,
            "fake": fake_features,
            "backend": real_provenance["backend"],
            "protocol": real_provenance["protocol"],
            "weights_sha256": real_provenance["checkpoint_sha256"],
            "backbone": real_audit["provenance"]["backbone"],
            "preprocessing": "torch_bilinear_299_align_corners_false_rgb_minus1_plus1",
            "cache_version": state.cache_versions["fid"],
            "real_forward_audit": real_audit,
            "fake_forward_audit": fake_audit,
        }
        state.feature_cache_events.append({"metric": "fid", "domain": "real", **real_audit})
        state.feature_cache_events.append({"metric": "fid", "domain": "fake", **fake_audit})
        state.execution_trace[-1]["details"] = {
            "binding_layer": "ExecutionBindingLayer.run_fid_features",
            "executor": "InceptionFIDExecutor.forward",
            "binding_layer.fid_forward_called": bool(state.binding_trace.get("binding_layer.fid_forward_called")),
            "real_forward_executed": bool(real_audit["forward_executed"]),
            "fake_forward_executed": bool(fake_audit["forward_executed"]),
            "real_feature_shape": real_audit["output_shape"],
            "fake_feature_shape": fake_audit["output_shape"],
            "cache_read_enabled": bool(state.cfg.get("metrics", {}).get("fid", {}).get("allow_cache_read", False)),
        }

    def run_fvd_feature_extractor(self, state: RuntimeState) -> None:
        plan = state.plan
        if plan is None:
            raise RuntimeError("sampling plan is not available")
        if len(plan.real_clips) < 2 or len(plan.fake_clips) < 2:
            raise ValueError("Dataset-level FVD needs at least two sampled real and fake clips.")
        binding = self._binding(state)
        real_features, real_audit = binding.run_fvd_features(plan.real_clips, "real")
        fake_features, fake_audit = binding.run_fvd_features(plan.fake_clips, "fake")
        binding.validate_fvd_trace()
        if (real_features.ndim != 2 or fake_features.ndim != 2
                or real_features.shape[1] != FVD_FEATURE_DIM or fake_features.shape[1] != FVD_FEATURE_DIM):
            raise RuntimeError(
                f"Invalid I3D feature shape: real={real_features.shape}, fake={fake_features.shape}. "
                "Standard FVD requires 400-dimensional Kinetics pre-softmax logits."
            )
        for audit in (real_audit, fake_audit):
            provenance = audit["provenance"]
            for key, expected in (("protocol", FVD_PROTOCOL), ("output_kind", FVD_OUTPUT_KIND),
                                  ("weights_sha256", FVD_REFERENCE_WEIGHTS_SHA256)):
                if provenance.get(key) != expected:
                    raise RuntimeError(f"Standard FVD feature provenance mismatch: {key}")
        state.features["fvd"] = {
            "real": real_features,
            "fake": fake_features,
            "backend": "i3d_torchscript",
            "backbone": "Kinetics-400 I3D pre-softmax logits",
            "protocol": FVD_PROTOCOL,
            "feature_kind": FVD_OUTPUT_KIND,
            "weights_sha256": real_audit["provenance"]["weights_sha256"],
            "weights_path": real_audit["provenance"]["weights_path"],
            "cache_version": state.cache_versions["fvd"],
            "real_forward_audit": real_audit,
            "fake_forward_audit": fake_audit,
        }
        state.feature_cache_events.append({"metric": "fvd", "domain": "real", **real_audit})
        state.feature_cache_events.append({"metric": "fvd", "domain": "fake", **fake_audit})
        state.execution_trace[-1]["details"] = {
            "binding_layer": "ExecutionBindingLayer.run_fvd_features",
            "executor": "I3DFVDExecutor.forward",
            "binding_layer.fvd_forward_called": bool(state.binding_trace.get("binding_layer.fvd_forward_called")),
            "real_forward_executed": bool(real_audit["forward_executed"]),
            "fake_forward_executed": bool(fake_audit["forward_executed"]),
            "real_feature_shape": real_audit["output_shape"],
            "fake_feature_shape": fake_audit["output_shape"],
            "cache_read_enabled": bool(state.cfg.get("metrics", {}).get("fvd", {}).get("allow_cache_read", False)),
        }

    def run_metrics(self, state: RuntimeState) -> None:
        raise RuntimeError("Metric computation is graph-only; use EvaluationEngine.run().")

    def run_fid_metric(self, state: RuntimeState) -> None:
        from driveaudit_eval.research.video.metrics.fid_metric import FIDMetric
        plan = state.plan
        features = state.features["fid"]
        real_features = np.asarray(features["real"], dtype=np.float64)
        fake_features = np.asarray(features["fake"], dtype=np.float64)
        value = FIDMetric.compute(real_features, fake_features)
        min_images = int(state.cfg.get("metrics", {}).get("fid", {}).get("min_images_for_normal_confidence", 5000))
        count = min(len(real_features), len(fake_features))
        state.results["fid"] = {
            "enabled": True,
            "skipped": False,
            "reason": None,
            "backend": features["backend"],
            "fid_backbone": features["backbone"],
            "fid": value,
            "fid_dataset": value,
            "fid_feature_backend": features["backend"],
            "fid_preprocessing": features["preprocessing"],
            "fid_feature_dim": int(real_features.shape[1]),
            "protocol": features["protocol"],
            "feature_kind": "tensorflow_inception_pool3",
            "weights_sha256": features["weights_sha256"],
            "fid_num_real_images": int(len(real_features)),
            "fid_num_fake_images": int(len(fake_features)),
            "fid_num_gen_images": int(len(fake_features)),
            "metric_confidence": "normal" if count >= min_images else "low_sample",
            "metric_validity": "valid" if count >= min_images else "valid_low_sample",
            "comparison_validity": "distribution_level",
            "sampling": "dataset_level_global",
            "scene_level_aggregation": False,
            "cache_version": features["cache_version"],
            "sampled_cameras": list(plan.cameras) if plan else [],
            "real_forward_executed": bool(features["real_forward_audit"]["forward_executed"]),
            "fake_forward_executed": bool(features["fake_forward_audit"]["forward_executed"]),
        }
        state.execution_trace[-1]["details"] = {
            "metric": "FIDMetric.compute",
            "input": "inception_embeddings_only",
            "real_embedding_shape": list(real_features.shape),
            "fake_embedding_shape": list(fake_features.shape),
            "value": value,
        }

    def run_fvd_metric(self, state: RuntimeState) -> None:
        plan = state.plan
        features = state.features["fvd"]
        real_features = np.asarray(features["real"], dtype=np.float64)
        fake_features = np.asarray(features["fake"], dtype=np.float64)
        value = FVDMetric.compute(real_features, fake_features)
        min_clips = int(state.cfg.get("metrics", {}).get("fvd", {}).get("min_clips_for_normal_confidence", 1000))
        count = min(len(real_features), len(fake_features))
        state.results["fvd"] = {
            "enabled": True,
            "skipped": False,
            "reason": None,
            "backend": features["backend"],
            "fvd": value,
            "fvd_full": value,
            "fvd_dataset": value,
            "fvd_standard": True,
            "protocol": features["protocol"],
            "feature_kind": features["feature_kind"],
            "weights_sha256": features["weights_sha256"],
            "i3d_model_path": features["weights_path"],
            "i3d_backbone": features["backbone"],
            "i3d_feature_dim": int(real_features.shape[1]),
            "i3d_feature_shape_real": list(real_features.shape),
            "i3d_feature_shape_gen": list(fake_features.shape),
            "clip_len": int(plan.clip_length) if plan else None,
            "clip_stride": int(plan.stride) if plan else None,
            "fixed_fps_normalization": state.cfg.get("sampling", {}).get("time_policy") != "native_order",
            "time_policy": state.cfg.get("sampling", {}).get("time_policy", "nominal_fps_normalization"),
            "source_fps": float(plan.source_fps) if plan else None,
            "target_fps": float(plan.target_fps) if plan else None,
            "num_real_clips": int(len(real_features)),
            "num_gen_clips": int(len(fake_features)),
            "fvd_num_real_clips": int(len(real_features)),
            "fvd_num_gen_clips": int(len(fake_features)),
            "metric_confidence": "normal" if count >= min_clips else "low_sample",
            "metric_validity": "valid" if count >= min_clips else "valid_low_sample",
            "comparison_validity": "distribution_level",
            "sampling": "dataset_level_global",
            "scene_level_aggregation": False,
            "cache_version": features["cache_version"],
            "real_forward_executed": bool(features["real_forward_audit"]["forward_executed"]),
            "fake_forward_executed": bool(features["fake_forward_audit"]["forward_executed"]),
        }
        state.execution_trace[-1]["details"] = {
            "metric": "FVDMetric.compute",
            "input": "i3d_embeddings_only",
            "real_embedding_shape": list(real_features.shape),
            "fake_embedding_shape": list(fake_features.shape),
            "value": value,
        }

    def _run_stateful_consistency_metric(self, metric_cls, state: RuntimeState, summary_name: str) -> None:
        metric = metric_cls(state.cfg, state)
        if not metric.enabled:
            metric.disable()
            state.results[summary_name] = metric.summary
            return
        metric.prepare()
        for sample in state.consistency_samples:
            metric.update(sample)
        if not metric.rows and not metric.skipped:
            metric.skip("no_valid_observations", message="No valid observations were available for this metric.")
        state.results[summary_name] = metric.compute()
        metric.save(state.output_dir)

    def run_temporal_consistency(self, state: RuntimeState) -> None:
        from driveaudit_eval.research.video.metrics.temporal_warp_metric import TemporalWarpMetric
        self._run_stateful_consistency_metric(TemporalWarpMetric, state, "temporal_consistency")

    def run_history_future_continuity(self, state: RuntimeState) -> None:
        from driveaudit_eval.research.video.metrics.history_future_metric import HistoryFutureContinuityMetric
        self._run_stateful_consistency_metric(
            HistoryFutureContinuityMetric, state, "history_future_continuity"
        )

    def run_multi_view_consistency(self, state: RuntimeState) -> None:
        max_frames = state.cfg.get("sampling", {}).get("max_multiview_frames")
        loftr_cfg = state.cfg.setdefault("metrics", {}).setdefault("loftr", {})
        if loftr_cfg.get("max_frames") is None and max_frames is not None:
            loftr_cfg["max_frames"] = int(max_frames)
        self._run_stateful_consistency_metric(LoFTRMultiviewMetric, state, "multi_view_consistency")

    def run_geometry_consistency(self, state: RuntimeState) -> None:
        self._run_stateful_consistency_metric(EpipolarMetric, state, "geometry_consistency")

    def validate_execution_bindings(self, state: RuntimeState) -> None:
        trace_names = [record.get("name") for record in state.execution_trace]
        if state.cfg.get("metrics", {}).get("fid", {}).get("enabled", True):
            required_nodes = [
                "binding_layer.initialize",
                "preprocessing.fid.inception_299_imagenet",
                "features.fid.binding_layer_inception_forward",
                "metrics.fid.frechet_distance",
            ]
            missing_nodes = [name for name in required_nodes if name not in trace_names]
            if missing_nodes:
                raise RuntimeError(f"FID execution trace missing graph nodes: {missing_nodes}")
            required_events = [
                "binding_layer.fid_forward_called",
                "binding_layer.fid_real_forward_called",
                "binding_layer.fid_fake_forward_called",
            ]
            missing_events = [key for key in required_events if not state.binding_trace.get(key)]
            if missing_events:
                raise RuntimeError(f"FID execution trace missing binding events: {missing_events}")
            allow_cache = bool(state.cfg.get("metrics", {}).get("fid", {}).get("allow_cache_read", False))
            for domain in ("real", "fake"):
                executed = bool(state.binding_trace.get(f"inception_{domain}_forward_executed"))
                verified_cache = bool(state.binding_trace.get(f"inception_{domain}_cache_verified"))
                if not executed and not (allow_cache and verified_cache):
                    raise RuntimeError(f"FID missing Inception forward execution for {domain}.")
        if state.cfg.get("metrics", {}).get("fvd", {}).get("enabled", True):
            required_nodes = [
                "binding_layer.initialize",
                "preprocessing.fvd.i3d_clips",
                "features.fvd.binding_layer_i3d_forward",
                "metrics.fvd.frechet_distance",
            ]
            missing_nodes = [name for name in required_nodes if name not in trace_names]
            if missing_nodes:
                raise RuntimeError(f"FVD execution trace missing graph nodes: {missing_nodes}")
            required_events = [
                "binding_layer.fvd_forward_called",
                "binding_layer.fvd_real_forward_called",
                "binding_layer.fvd_fake_forward_called",
            ]
            missing_events = [key for key in required_events if not state.binding_trace.get(key)]
            if missing_events:
                raise RuntimeError(f"FVD execution trace missing binding events: {missing_events}")
            allow_cache = bool(state.cfg.get("metrics", {}).get("fvd", {}).get("allow_cache_read", False))
            for domain in ("real", "fake"):
                executed = bool(state.binding_trace.get(f"i3d_{domain}_forward_executed"))
                verified_cache = bool(state.binding_trace.get(f"i3d_{domain}_cache_verified"))
                if not executed and not (allow_cache and verified_cache):
                    raise RuntimeError(f"FVD missing I3D forward execution for {domain}.")

    def aggregate_results(self, state: RuntimeState) -> None:
        self.validate_execution_bindings(state)
        graph_description = state.graph or self.graph.describe()
        summary: Dict[str, Any] = {
            "benchmark": {
                "name": "ground_truth_self_evaluation" if state.cfg.get("input_source") == "ground_truth_oracle" else "paper_grade_dataset_level_video_generation",
                "version": BENCHMARK_VERSION,
                "protocol": BENCHMARK_PROTOCOL,
                "engine_driven": True,
                "deterministic_execution_graph": True,
                "unified_runtime_state": True,
                "reproducible": True,
                "seed": state.seed,
                "strict_i3d_fvd": True,
                "strict_inception_fid": True,
                "no_fallback_fvd": True,
                "no_per_scene_fid_fvd": True,
                "execution_graph_hash": _json_hash(graph_description),
            },
            "manifest": {
                "dataset_version_hash": state.manifest.get("dataset_version_hash"),
                "scene_order_hash": state.manifest.get("scene_order_hash"),
                "benchmark_protocol": state.manifest.get("benchmark_protocol"),
                "version": state.manifest.get("version"),
            },
            "sampling": state.plan.summary() if state.plan else {},
            "fid": state.results.get("fid", _metric_disabled("fid")),
            "fvd": state.results.get("fvd", _metric_disabled("fvd")),
            "temporal_consistency": state.results.get(
                "temporal_consistency", _metric_disabled("temporal_consistency")
            ),
            "history_future_continuity": state.results.get(
                "history_future_continuity", _metric_disabled("history_future_continuity")
            ),
            "multi_view_consistency": state.results.get(
                "multi_view_consistency", _metric_disabled("multi_view_consistency")
            ),
            "geometry_consistency": state.results.get(
                "geometry_consistency", _metric_disabled("geometry_consistency")
            ),
            "execution_graph": graph_description,
            "execution_trace": state.execution_trace,
            "runtime_state": {
                "seed": state.seed,
                "device": state.device,
                "cache_dir": str(state.cache_dir),
                "cache_versions": state.cache_versions,
                "feature_cache_policy": {
                    "disk_cache_enabled": bool(state.cfg.get('output', {}).get('save_cache', True)),
                    "internal_cache_only": True,
                    "cache_generated_after_real_forward": True,
                    "default_cache_read_enabled": False,
                    "cross_metric_reuse_allowed": False,
                    "feature_generation_path": "engine_dag_to_execution_binding_layer_to_executor_forward",
                },
                "feature_cache_events": state.feature_cache_events,
                "binding_trace": state.binding_trace,
                "binding_events": state.binding_events,
                "preprocessing": state.preprocessing,
                "dataset_index_order": state.dataset_index_order,
                "dataset_version_hash": state.dataset_version_hash,
                "scene_order_hash": state.scene_order_hash,
            },
            "warnings": state.warnings.items,
            "config": state.cfg,
            "dataset_statistics": state.dataset_statistics,
        }
        if 'visual_quality' in state.results:
            summary['visual_quality'] = state.results['visual_quality']
        assert_no_forbidden_keys(summary)
        validation_errors = validate_summary(summary)
        if validation_errors:
            state.warnings.add(
                "validators",
                "eval_output_consistency",
                "; ".join(validation_errors),
                severity="warning",
            )
            summary["warnings"] = state.warnings.items
        write_json(state.output_dir / "results.json", summary)
        write_json(state.output_dir / "summary_by_metric.json", summary)
        write_json(state.output_dir / "warnings.json", state.warnings.items)
        write_json(state.output_dir / "execution_graph.json", graph_description)
        write_json(state.output_dir / "execution_trace.json", state.execution_trace)
        write_json(state.output_dir / "runtime_state.json", summary["runtime_state"])
        audit = build_metric_audit(summary)
        write_json(state.output_dir / "metric_audit_report.json", audit)
        (state.output_dir / "metric_audit_report.md").write_text(audit_markdown(audit), encoding="utf-8")
        report = _markdown(summary)
        (state.output_dir / "report.md").write_text(report, encoding="utf-8")
        (state.output_dir / "summary.md").write_text(report, encoding="utf-8")
        _write_tables(state.output_dir, state, summary)
        state.results["summary"] = summary

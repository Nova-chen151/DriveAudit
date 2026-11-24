from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "cache" / "models" / "video"
DEFAULT_DINO_REPO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "dataset_root": None,
        "generated_root": None,
        "real_reference_root": None,
        "history_root": None,
        "metadata_root": None,
        "evaluation_manifest": None,
        "output_dir": "outputs/eval_results",
    },
    "dataset": {
        "history_frames": 10,
        "future_frames": 80,
        "cameras": CAMERAS,
        "image_extension": "jpg",
    },
    "sampling": {
        "seed": 42,
        "camera_set": "surround",
        "cameras": None,
        "frames_per_scene": 50,
        "num_frames": 16,
        "clip_length": 16,
        "stride": 2,
        "source_fps": 12,
        "target_fps": 12,
        "max_clips_per_scene_camera": None,
        "max_multiview_frames": 16,
    },
    "time_windows": {
        "short": [0, 15],
        "middle": [16, 47],
        "long": [48, 79],
        "full": [0, 79],
    },
    "preprocess": {
        "resize_height": 224,
        "resize_width": 400,
        "rgb_order": "RGB",
        "normalize": "0_1",
    },
    "geometry": {
        "extrinsic_convention": "auto",
        "convention_candidates": ["sensor_to_ego", "world_to_camera"],
        "evaluate_intrinsic_scaling_modes": ["direct_resize", "center_crop_1600x896"],
        "evaluate_extrinsic_conventions": ["sensor_to_ego", "world_to_camera", "ego_to_camera"],
        "scale_intrinsics_for_resize": True,
    },
    "real_reference": {
        "mode": "unpaired_distribution",
        "align_by_stem": False,
        "allow_paired_scene_sanity": True,
    },
    "weights": {
        "root": str(DEFAULT_WEIGHTS_ROOT),
        "dino_repo": str(DEFAULT_DINO_REPO),
        "dino_vits14": str(DEFAULT_WEIGHTS_ROOT / "dinov2" / "dinov2_vits14_pretrain.pth"),
        "loftr_outdoor": str(DEFAULT_WEIGHTS_ROOT / "loftr" / "loftr_outdoor.ckpt"),
        "yolo": str(DEFAULT_WEIGHTS_ROOT / "yolo" / "yolo11s.pt"),
        "raft_dir": str(DEFAULT_WEIGHTS_ROOT / "raft" / "models"),
        "fvd_i3d_torchscript": str(DEFAULT_WEIGHTS_ROOT / "fvd" / "i3d_torchscript.pt"),
        "fid_inception_v3": str(DEFAULT_WEIGHTS_ROOT / "fid" / "pt_inception-2015-12-05-6726825d.pth"),
    },
    "metrics": {
        "fid": {
            "enabled": True,
            "backend": "pytorch_fid_inception_v3",
            "weights_path": None,
            "feature": 2048,
            "batch_size": 32,
            "resize": [299, 299],
            "allow_cache_read": False,
            "min_images_for_normal_confidence": 5000,
            "compute_sanity": False,
        },
        "fvd": {
            "enabled": True,
            "backend": "i3d_torchscript",
            "expected_weights_sha256": "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2",
            "weights_path": None,
            "clip_len": 16,
            "clip_stride": 2,
            "drop_last_incomplete_clip": True,
            "min_clips_for_normal_confidence": 1000,
            "compute_sanity": False,
            "fvd_standard_required": True,
            "batch_size": 2,
            "resize": [224, 224],
            "allow_cache_read": False,
        },
        "boundary": {
            "enabled": False,
            "dino": {
                "enabled": True,
                "weights_path": None,
                "model_name": "dinov2_vits14",
                "repo_path": str(DEFAULT_DINO_REPO),
            },
            "clip": {
                "enabled": True,
                "backend": "open_clip",
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b79k",
                "weights_path": None,
            },
            "use_basic_fallback": False,
        },
        "temporal_warp": {
            "enabled": True,
            "flow_backend": "opencv",
            "warp_error_unit": "mean_abs_rgb_0_1",
            "motion_collapse_warning_threshold": 0.6,
            "bad_frame_threshold": 0.12,
            "max_visualizations": 6,
        },
        "history_future": {
            "enabled": True,
            "flow_backend": "opencv",
            "warp_error_unit": "mean_abs_rgb_0_1",
        },
        "flicker": {
            "enabled": False,
            "flicker_unit": "mean_abs_gray_0_1",
            "static_flow_threshold": 1.0,
            "bad_frame_threshold": 0.08,
            "max_visualizations": 6,
        },
        "loftr": {
            "enabled": True,
            "backend": "kornia",
            "weights": "outdoor",
            "weights_path": None,
            "allow_orb_fallback": False,
            "fallback_mark_low_confidence": True,
            "download_weights": False,
            "low_match_threshold": 30,
            "max_frames": None,
            "max_visualizations": 12,
            "camera_pairs": [
                ["CAM_FRONT_LEFT", "CAM_FRONT"],
                ["CAM_FRONT", "CAM_FRONT_RIGHT"],
                ["CAM_BACK_LEFT", "CAM_BACK"],
                ["CAM_BACK", "CAM_BACK_RIGHT"],
            ],
        },
        "epipolar": {
            "enabled": True,
            "thresholds_px": [1, 3, 5],
            "bad_frame_threshold_px": 5.0,
        },
        "tracking": {
            "enabled": False,
            "backend": "ultralytics",
            "yolo_weights_path": None,
            "tracker_config": "bytetrack.yaml",
            "classes": [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "pedestrian",
            ],
            "class_aliases": {"person": "pedestrian"},
            "min_track_len": 5,
        },
    },
    "output": {
        "save_visualizations": True,
        "save_curves": True,
        "save_cache": True,
    },
    "runtime": {
        "device": "cuda",
        "num_workers": 0,
        "skip_heavy": False,
        "strict_heavy": True,
        "allow_fallback": False,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

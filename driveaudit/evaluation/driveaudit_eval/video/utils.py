from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


FORBIDDEN_OUTPUT_KEYS = {
    "overall_score",
    "final_score",
    "weighted_score",
    "combined_score",
    "rank_score",
    "total_score",
    "normalized_score_sum",
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def stable_hash(parts: Iterable[Any]) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: Path, size: Optional[Sequence[int]] = None) -> Image.Image:
    # Preserve stored orientation; calibration refers to the stored pixel grid.
    with Image.open(path) as source:
        image = source.convert("RGB")
    if size is not None:
        width, height = int(size[0]), int(size[1])
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
    return image


def image_to_np(path: Path, size: Optional[Sequence[int]] = None) -> np.ndarray:
    return np.asarray(read_image(path, size=size), dtype=np.uint8)


def read_image_bgr(path: Path) -> np.ndarray:
    """Use the same decoded pixels for existing OpenCV BGR consumers."""
    return image_to_np(path)[..., ::-1].copy()


def decoder_provenance() -> Dict[str, Any]:
    import cv2
    import PIL
    from PIL import features
    return {
        "decoder": "Pillow.Image.open.convert_RGB",
        "pillow_version": PIL.__version__, "jpeg_library_version": features.version_codec("jpg"),
        "libjpeg_turbo_version": features.version_feature("libjpeg_turbo"),
        "opencv_version": cv2.__version__, "color_order": "RGB",
        "orientation": "stored_pixels; EXIF transpose not applied",
        "resize": "none_at_decode", "crop": "none_at_decode", "dtype": "uint8",
        "model_input_transforms": {
            "fvd": "native RGB BCTHW float32 0..255; reference I3D rescale=True, resize=True (torch bilinear 224, align_corners=False)",
            "clip_iqa": "native RGB CHW float32 /255; pyiqa RN50 CLIP mean/std normalization; no resize/crop",
            "cmmd": "RGB center square; Pillow bicubic 336x336; CHW /255; CLIP mean/std; L2 features",
            "dino_t": "RGB torchvision Resize256 bicubic; CenterCrop224; ToTensor; ImageNet mean/std; L2 features",
            "epi_3px": "configured preprocess width/height; Pillow bilinear; OpenCV RGB2GRAY; BCHW float32 /255",
            "tvc": "RGB to contiguous BGR; configured tracking width/height via OpenCV INTER_AREA; BGR2GRAY",
            "tvc_future_entrants": "RGB to contiguous native BGR; OpenCV BGR2GRAY; existing conditional tracker",
        },
        "normalization": {
            "clip_mean": [.48145466, .4578275, .40821073],
            "clip_std": [.26862954, .26130258, .27577711],
            "imagenet_mean": [.485, .456, .406], "imagenet_std": [.229, .224, .225],
        },
        "historical_pixel_difference_resolved": False,
    }


def audit_inputs(scene_paths: Sequence[Path], output: Path) -> Dict[str, Any]:
    """Audit every selected history/future JPEG with two independent current decodes.

    Write the failing image's evidence before raising on any source/pixel change.
    The digest binds ordered image identities, bytes, decoded pixels and libraries.
    """
    scenes = [Path(scene).resolve() for scene in scene_paths]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("Decoded-input audit requires distinct selected scenes")
    profile = decoder_provenance()
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    pixel_digest = hashlib.sha256()
    output = Path(output)
    ensure_dir(output.parent)
    fields = ["scene_id", "scene_path", "split", "camera", "frame_index", "path",
              "jpeg_sha256", "jpeg_sha256_after", "decoded_pixel_sha256", "repeat_pixel_sha256",
              "pixel_shape", "pixel_dtype", "jpeg_unchanged", "repeat_match",
              "pillow_version", "opencv_version", "color_order", "orientation", "resize", "crop"]
    count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            for split in ("history", "future"):
                paths = sorted((scene / split / "rgb").glob("*/*.jpg"))
                if not paths:
                    raise ValueError(f"No {split} JPEG inputs: {scene}")
                for path in paths:
                    before = hashlib.sha256(path.read_bytes()).hexdigest()
                    pixels, repeated = image_to_np(path), image_to_np(path)
                    after = hashlib.sha256(path.read_bytes()).hexdigest()
                    pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                    repeat_hash = hashlib.sha256(repeated.tobytes()).hexdigest()
                    match = (pixels.shape == repeated.shape and pixels.dtype == repeated.dtype
                             and pixel_hash == repeat_hash)
                    row = dict(scene_id=scene.parent.parent.name + "__" + scene.name,
                        scene_path=str(scene), split=split, camera=path.parent.name,
                        frame_index=int(path.stem), path=str(path), jpeg_sha256=before,
                        jpeg_sha256_after=after, decoded_pixel_sha256=pixel_hash,
                        repeat_pixel_sha256=repeat_hash, pixel_shape=json.dumps(list(pixels.shape)),
                        pixel_dtype=str(pixels.dtype), jpeg_unchanged=before == after, repeat_match=match,
                        **{key: profile[key] for key in ("pillow_version", "opencv_version", "color_order",
                                                       "orientation", "resize", "crop")})
                    writer.writerow(row)
                    if before != after or not match:
                        stream.flush()
                        raise RuntimeError(f"Current input decode/source is not repeatable: {path}")
                    digest.update(json.dumps(row, sort_keys=True).encode())
                    pixel_digest.update(json.dumps({key: row[key] for key in (
                        "scene_id", "split", "camera", "frame_index", "pixel_shape", "pixel_dtype",
                        "decoded_pixel_sha256")}, sort_keys=True).encode())
                    count += 1
    return dict(status="PASS", scene_count=len(scenes), image_count=count,
                aggregate_input_sha256=digest.hexdigest(), decoded_pixels_sha256=pixel_digest.hexdigest(),
                evidence_csv=str(output), **profile)


def image_to_float(path: Path, size: Optional[Sequence[int]] = None) -> np.ndarray:
    return image_to_np(path, size=size).astype(np.float32) / 255.0


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ).astype(np.uint8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def stats(values: Iterable[Any], prefix: str = "") -> Dict[str, Optional[float]]:
    arr = finite(values)
    keys = ["mean", "median", "p10", "p50", "p90", "max"]
    if arr.size == 0:
        return {f"{prefix}{key}": None for key in keys}
    return {
        f"{prefix}mean": float(np.mean(arr)),
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}p10": float(np.percentile(arr, 10)),
        f"{prefix}p50": float(np.percentile(arr, 50)),
        f"{prefix}p90": float(np.percentile(arr, 90)),
        f"{prefix}max": float(np.max(arr)),
    }


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    arr = finite(values)
    return None if arr.size == 0 else float(np.mean(arr))


def percentile_or_none(values: Iterable[Any], p: float) -> Optional[float]:
    arr = finite(values)
    return None if arr.size == 0 else float(np.percentile(arr, p))


def time_window(frame_index: int, windows: Dict[str, Sequence[int]]) -> str:
    for name in ("short", "middle", "long"):
        if name in windows:
            start, end = windows[name]
            if start <= frame_index <= end:
                return name
    return "full"


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in fieldnames})


def assert_no_forbidden_keys(data: Any) -> None:
    if isinstance(data, dict):
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(data.keys())
        if forbidden:
            raise ValueError(f"forbidden score keys in output: {sorted(forbidden)}")
        for value in data.values():
            assert_no_forbidden_keys(value)
    elif isinstance(data, list):
        for value in data:
            assert_no_forbidden_keys(value)


def draw_side_by_side(images: Sequence[Path], labels: Sequence[str], out: Path) -> None:
    pil_images = [read_image(path) for path in images]
    width = max(image.width for image in pil_images)
    height = max(image.height for image in pil_images)
    label_h = 24
    canvas = Image.new("RGB", (width * len(pil_images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(pil_images):
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        x = idx * width
        canvas.paste(image, (x, label_h))
        draw.text((x + 5, 5), labels[idx], fill=(0, 0, 0))
    ensure_dir(out.parent)
    canvas.save(out)


class WarningCollector:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(
        self,
        metric: str,
        warning_type: str,
        message: str,
        scene_id: Optional[str] = None,
        severity: str = "warning",
    ) -> None:
        self.items.append(
            {
                "scene_id": scene_id,
                "metric": metric,
                "warning_type": warning_type,
                "message": message,
                "severity": severity,
            }
        )

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        self.items.extend(items)


class EvalState:
    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.cache_dir = output_dir / "eval_cache"
        self.warnings = WarningCollector()
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "scene_metrics": [],
            "camera_metrics": [],
            "time_window_metrics": [],
            "frame_metrics": [],
            "camera_pair_metrics": [],
            "track_metrics": [],
        }
        self.shared: Dict[str, Any] = {}

    def add_row(self, table: str, row: Dict[str, Any]) -> None:
        self.rows.setdefault(table, []).append(row)

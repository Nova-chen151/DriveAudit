from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np


def image_quality(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    pixels = mask > 0
    if not np.any(pixels):
        return {"image_quality_status": "invalid_crop", "blur_score": 0.0, "contrast_score": 0.0,
                "exposure_score": 0.0, "compression_score": 0.0, "repeated_frame_probability": 0.0,
                "frozen_region_probability": 0.0, "absolute_difference": None, "roi_ssim": None}
    gray_a, gray_b = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    a, b = gray_a[pixels].astype(np.float32), gray_b[pixels].astype(np.float32)
    blur = float(cv2.Laplacian(gray_a, cv2.CV_32F)[pixels].var())
    contrast = float(np.std(a)); mean_luma = float(np.mean(a))
    difference = float(np.mean(np.abs(a - b)))
    covariance = float(np.mean((a - a.mean()) * (b - b.mean())))
    c1, c2 = 6.5025, 58.5225
    ssim = float(((2*a.mean()*b.mean()+c1)*(2*covariance+c2)) /
                 ((a.mean()**2+b.mean()**2+c1)*(a.var()+b.var()+c2)))
    exact = hashlib.sha1(a.tobytes()).digest() == hashlib.sha1(b.tobytes()).digest()
    repeated = 1.0 if exact else float(np.exp(-difference / float(config["repeat_difference_scale"])))
    frozen = float(np.clip(.5 * repeated + .5 * np.exp(-difference / float(config["freeze_difference_scale"])), 0, 1))
    block = _blocking(gray_a, pixels); exposure = 1.0 - min(1.0, abs(mean_luma - 127.5) / 127.5)
    status = "usable"
    if blur < float(config["minimum_blur_score"]): status = "unscorable_blur"
    elif contrast < float(config["minimum_contrast"]): status = "unscorable_low_contrast"
    elif mean_luma < float(config["minimum_luma"]): status = "unscorable_underexposure"
    elif mean_luma > float(config["maximum_luma"]): status = "unscorable_overexposure"
    return {"image_quality_status": status, "blur_score": blur, "contrast_score": contrast,
            "exposure_score": exposure, "compression_score": max(0.0, 1.0 - block),
            "repeated_frame_probability": repeated, "frozen_region_probability": frozen,
            "absolute_difference": difference, "roi_ssim": ssim, "frame_hash_equal": exact,
            "near_uniform_region": contrast < float(config["minimum_contrast"])}


def _blocking(gray: np.ndarray, pixels: np.ndarray) -> float:
    vertical = np.abs(np.diff(gray.astype(np.float32), axis=1)); horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
    v = float(np.mean(vertical[:, 7::8])) if vertical.shape[1] > 8 else 0.0
    h = float(np.mean(horizontal[7::8, :])) if horizontal.shape[0] > 8 else 0.0
    return min(1.0, (v + h) / 64.0)


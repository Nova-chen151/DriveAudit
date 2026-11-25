"""Optional paired RGB pixel diagnostics, with genuine SSIM on request."""
import math
import numpy as np


def pixel_metrics(image, reference, *, ssim=False):
    a, b = np.asarray(image), np.asarray(reference)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('paired images must have the same H x W x 3 shape')
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('pixel diagnostics require uint8 images in the original 0..255 range')
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    result = {'pixel_l1': float(np.mean(np.abs(diff))) / 255.0,
              'pixel_l2': math.sqrt(mse) / 255.0,
              'psnr': None if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse)),
              'psnr_identical': mse == 0}
    if ssim:
        from skimage.metrics import structural_similarity
        result['ssim'] = float(structural_similarity(a, b, data_range=255, channel_axis=-1))
    return result

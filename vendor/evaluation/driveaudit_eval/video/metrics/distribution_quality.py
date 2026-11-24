"""Unpaired feature-distribution metrics; no changes to legacy FID/FVD.

CMMD estimator follows google-research/cmmd/distance.py (biased Gaussian MMD).
KID uses unbiased degree-three polynomial MMD; negative estimates are valid.
Feature extraction/provenance is supplied by the video evaluator, not this module.
"""
import numpy as np


def _features(real, generated):
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1] or real.shape[1] == 0:
        raise ValueError('features must be N x D arrays with matching positive D')
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError('features contain non-finite values')
    return real, generated




def cmmd(real, generated, sigma=10.0, scale=1000.0, block_size=512, seed=42,
         clip_model='OpenAI ViT-L/14@336px'):
    real, generated = _features(real, generated)
    if sigma <= 0 or scale <= 0 or block_size < 1:
        raise ValueError('sigma, scale and block_size must be positive')
    result = dict(real_sample_count=len(real), generated_sample_count=len(generated),
                  feature_dim=real.shape[1], kernel_sigma=sigma, scale=scale,
                  sampling_seed=seed, clip_model=clip_model, estimator='biased_gaussian_mmd', cmmd=None)
    if min(len(real), len(generated)) < 2:
        return dict(result, status='insufficient_samples', reason='at_least_two_samples_per_distribution_required', coverage=0.0)
    def mean_kernel(x, y):
        total = 0.0
        for i in range(0, len(x), block_size):
            a = x[i:i+block_size]
            for j in range(0, len(y), block_size):
                b = y[j:j+block_size]
                sq = np.maximum((a*a).sum(1)[:, None]+(b*b).sum(1)[None, :]-2*a@b.T, 0)
                total += np.exp(-sq/(2*sigma*sigma)).sum()
        return total/(len(x)*len(y))
    value = scale*(mean_kernel(real, real)+mean_kernel(generated, generated)-2*mean_kernel(real, generated))
    return dict(result, cmmd=float(value), status='valid', reason=None, coverage=1.0)

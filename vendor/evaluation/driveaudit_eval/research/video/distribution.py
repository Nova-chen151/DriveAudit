import numpy as np
from driveaudit_eval.video.metrics.distribution_quality import _features

def kid(real, generated, subset_size=100, num_subsets=100, seed=42):
    real, generated = _features(real, generated)
    if subset_size < 2 or num_subsets < 1:
        raise ValueError('subset_size >= 2 and num_subsets >= 1 are required')
    result = dict(real_sample_count=len(real), generated_sample_count=len(generated),
                  feature_dim=real.shape[1], subset_size=subset_size, num_subsets=num_subsets,
                  seed=seed, kid_mean=None, kid_std=None)
    if min(len(real), len(generated)) < subset_size:
        return dict(result, status='insufficient_samples', reason='fewer_samples_than_configured_subset', coverage=0.0)
    rng = np.random.default_rng(seed)
    estimates = []
    d, m = real.shape[1], subset_size
    for _ in range(num_subsets):
        x, y = real[rng.choice(len(real), m, replace=False)], generated[rng.choice(len(generated), m, replace=False)]
        xx, yy, xy = (x @ x.T / d + 1)**3, (y @ y.T / d + 1)**3, (x @ y.T / d + 1)**3
        estimates.append((xx.sum()-np.trace(xx)+yy.sum()-np.trace(yy))/(m*(m-1))-2*xy.mean())
    return dict(result, kid_mean=float(np.mean(estimates)), kid_std=float(np.std(estimates)),
                status='valid', reason=None, coverage=1.0)


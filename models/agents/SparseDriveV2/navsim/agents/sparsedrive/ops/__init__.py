import os

# Optional ABI-matched build of these two vendored CUDA operators.
if os.environ.get("DRIVEAUDIT_SPARSE_OPS"):
    __path__.append(os.environ["DRIVEAUDIT_SPARSE_OPS"])

from .deformable_aggregation import (
    deformable_aggregation_func,
    feature_maps_format,
    deformable_format,
)

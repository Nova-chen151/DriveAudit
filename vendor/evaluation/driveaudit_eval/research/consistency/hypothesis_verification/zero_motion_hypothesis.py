from __future__ import annotations

from typing import Any
import numpy as np


def build(observed:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    vectors=np.asarray(observed["observed_flow_vectors"],float);magnitude=float(np.median(np.linalg.norm(vectors,axis=1))) if len(vectors) else 0.0
    repeated=float(observed.get("repeated_frame_probability",0));frozen=float(observed.get("frozen_region_probability",0))
    residual=float(np.median(np.linalg.norm(vectors,axis=1))) if len(vectors) else 0.0
    visual=max(repeated,frozen,float(np.exp(-magnitude/max(float(config["motion_scale_px"]),1e-6))))
    return {"zero_expected_flow_per_point":np.zeros_like(vectors).tolist(),"zero_motion_residual":residual,
            "zero_visual_prior":visual,"repeated_frame_probability":repeated,"frozen_region_probability":frozen,
            "observed_motion_magnitude":magnitude}


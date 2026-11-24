from __future__ import annotations

from pathlib import Path
from typing import Any


def write(metrics:dict[str,Any],camera:dict[str,Any],output:str|Path)->None:
    lines=["# GT-VTC v2 root cause report","","| cause | evidence | conclusion |","|---|---|---|"]
    for name in ("CAM_BACK_LEFT","CAM_BACK_RIGHT"):
        value=camera.get(name,{});lines.append(f"| {name} large displacement / low texture | ordinary/guided EPE={value.get('ordinary_lk_epe')}/{value.get('geometry_lk_epe')}, valid gain={value.get('valid_point_gain')}, expected displacement={value.get('expected_pixel_displacement')}, blur={value.get('image_blur')}, JPEG block={value.get('jpeg_block_score')}, surface transition={value.get('surface_transition_rate')} | {value.get('primary_failure_reason','insufficient evidence')} |")
    lines += ["| native timestamp omission | v1 was already camera-exposure interpolated; v2 common-time ablation retained | not the principal v1 failure |",
              "| GT prediction leakage | oracle-leakage corruptions are scored from RGB-tracked points | checked separately; GT points never become observations |",
              "| missing exact instance surface | no GT mask/depth; visible cuboid faces remain an approximation | confirmed limitation |"]
    Path(output,"root_cause_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

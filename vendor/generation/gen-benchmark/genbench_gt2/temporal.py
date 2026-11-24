"""Fixed temporal plans and deterministic MagicDrive window selection."""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageStat

from .constants import CAMERAS


_RESAMPLING = getattr(Image, "Resampling", Image)


def _tuple(value, name):
    if not isinstance(value, tuple):
        raise ValueError("{} must be a tuple".format(name))
    return value


@dataclass(frozen=True)
class HorizonPlan:
    condition_frames: Tuple[int, ...]
    generated_windows: Tuple[Tuple[int, ...], ...]

    def __post_init__(self):
        conditions = _tuple(self.condition_frames, "condition_frames")
        windows = _tuple(self.generated_windows, "generated_windows")
        if conditions != tuple(range(10)) + (9,):
            raise ValueError("Horizon condition frames must be 0..9 followed by 9")
        expected = tuple(tuple(range(10 + 10 * i, 20 + 10 * i)) for i in range(8))
        if windows != expected:
            raise ValueError("Horizon windows must contain eight ten-frame windows")
        if tuple(frame for window in windows for frame in window) != tuple(range(10, 90)):
            raise ValueError("Horizon windows must cover generated frames 10..89 exactly")


@dataclass(frozen=True)
class MagicDrivePlan:
    condition_windows: Tuple[Tuple[int, ...], ...]

    def __post_init__(self):
        windows = _tuple(self.condition_windows, "condition_windows")
        expected = (tuple(range(10, 71)), tuple(range(29, 90)))
        if windows != expected:
            raise ValueError("MagicDrive requires fixed 61-frame windows 10..70 and 29..89")


@dataclass(frozen=True)
class OpenDWMPlan:
    reference_frames: Tuple[int, ...]
    sequence_frames: Tuple[int, ...]
    iteration_starts: Tuple[int, ...]
    generated_frames: Tuple[int, ...]

    def __post_init__(self):
        values = (
            (self.reference_frames, (7, 8, 9), "reference_frames"),
            (self.sequence_frames, tuple(range(7, 90)), "sequence_frames"),
            (self.iteration_starts, (0, 16, 32, 48, 64), "iteration_starts"),
            (self.generated_frames, tuple(range(10, 90)), "generated_frames"),
        )
        for actual, expected, name in values:
            if _tuple(actual, name) != expected:
                raise ValueError("OpenDWM {} do not match the fixed five-window plan".format(name))


@dataclass(frozen=True)
class PanaceaWindow:
    condition_frame: int
    control_frames: Tuple[int, ...]
    output_frames: Tuple[int, ...]

    def __post_init__(self):
        controls = _tuple(self.control_frames, "control_frames")
        outputs = _tuple(self.output_frames, "output_frames")
        if not outputs or len(outputs) > 7:
            raise ValueError("Panacea output window must contain one to seven frames")
        if len(controls) != 7 or controls[: len(outputs)] != outputs:
            raise ValueError("Panacea controls must begin with the unpadded output frames")
        if controls[len(outputs) :] != (89,) * (7 - len(outputs)):
            raise ValueError("Panacea controls may only be padded with frame 89")
        if outputs != tuple(range(outputs[0], outputs[0] + len(outputs))):
            raise ValueError("Panacea output frames must be contiguous")
        if any(frame < 10 or frame > 89 for frame in outputs):
            raise ValueError("Panacea output frames must be in 10..89")
        if any(frame < 10 or frame > 89 for frame in controls):
            raise ValueError("Panacea control frames must be in 10..89")


@dataclass(frozen=True)
class PanaceaPlan:
    windows: Tuple[PanaceaWindow, ...]

    def __post_init__(self):
        windows = _tuple(self.windows, "windows")
        if len(windows) != 12 or not all(isinstance(window, PanaceaWindow) for window in windows):
            raise ValueError("Panacea requires twelve windows")
        if windows[0].condition_frame != 9:
            raise ValueError("Panacea must begin from history frame 9")
        for previous, current in zip(windows, windows[1:]):
            if current.condition_frame != previous.output_frames[-1]:
                raise ValueError("Panacea conditions must chain from the prior output")
        if tuple(frame for window in windows for frame in window.output_frames) != tuple(range(10, 90)):
            raise ValueError("Panacea windows must cover generated frames 10..89 exactly")


@dataclass(frozen=True)
class SpliceDecision:
    frame_index: int
    score: float
    metric: str = "mean_lab_l1_50x28"
    candidates: int = 41

    def __post_init__(self):
        if isinstance(self.frame_index, bool) or self.frame_index not in range(30, 71):
            raise ValueError("MagicDrive splice frame must be in 30..70")
        if not isinstance(self.score, (int, float)) or not math.isfinite(float(self.score)):
            raise ValueError("MagicDrive splice score must be finite")
        if self.metric != "mean_lab_l1_50x28":
            raise ValueError("unsupported MagicDrive splice metric")
        if self.candidates != 41:
            raise ValueError("MagicDrive splice must evaluate 41 candidates")


def horizon_plan() -> HorizonPlan:
    windows = tuple(tuple(range(10 + 10 * i, 20 + 10 * i)) for i in range(8))
    return HorizonPlan(tuple(range(10)) + (9,), windows)


def magicdrive_plan() -> MagicDrivePlan:
    return MagicDrivePlan((tuple(range(10, 71)), tuple(range(29, 90))))


def opendwm_plan() -> OpenDWMPlan:
    return OpenDWMPlan((7, 8, 9), tuple(range(7, 90)), (0, 16, 32, 48, 64), tuple(range(10, 90)))


def panacea_plan() -> PanaceaPlan:
    windows = []
    condition = 9
    for window_index in range(12):
        real = tuple(range(10 + 7 * window_index, min(17 + 7 * window_index, 90)))
        controls = real + (89,) * (7 - len(real))
        windows.append(PanaceaWindow(condition, controls, real))
        condition = real[-1]
    return PanaceaPlan(tuple(windows))


def _validate_windows(first: Mapping[str, Sequence[Image.Image]], second: Mapping[str, Sequence[Image.Image]]) -> None:
    if not isinstance(first, Mapping) or not isinstance(second, Mapping) or not first:
        raise ValueError("MagicDrive windows must be non-empty mappings")
    canonical_cameras = set(CAMERAS)
    if set(first) != canonical_cameras or set(second) != canonical_cameras:
        raise ValueError("MagicDrive windows must contain exactly the canonical six cameras")
    if set(first) != set(second):
        raise ValueError("MagicDrive windows must contain the same cameras")
    for camera in CAMERAS:
        first_frames, second_frames = first[camera], second[camera]
        if not isinstance(first_frames, Sequence) or not isinstance(second_frames, Sequence):
            raise ValueError("MagicDrive camera windows must be sequences")
        if len(first_frames) < 61 or len(second_frames) < 61:
            raise ValueError("MagicDrive windows must contain at least 61 frames")
        if not all(isinstance(image, Image.Image) for image in first_frames[:61]) or not all(isinstance(image, Image.Image) for image in second_frames[:61]):
            raise ValueError("MagicDrive windows must contain Pillow images")


def _mean_lab_l1(left: Image.Image, right: Image.Image) -> float:
    left_small = _lab(left).resize((50, 28), _RESAMPLING.BOX)
    right_small = _lab(right).resize((50, 28), _RESAMPLING.BOX)
    channel_means = ImageStat.Stat(ImageChops.difference(left_small, right_small)).mean
    return float(sum(channel_means) / len(channel_means))


def _lab(image: Image.Image) -> Image.Image:
    """Convert to LAB, with a Pillow-8-compatible fallback.

    Pillow versions shipped in the MagicDrive environment do not expose the
    RGB-to-LAB conversion despite supporting the mode on newer releases.
    Encoding CIELAB into an RGB container keeps the resize and difference
    operations dependency-light and deterministic there.
    """
    try:
        return image.convert("LAB")
    except ValueError:
        import numpy as np
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
        xyz = np.stack(((red * 0.4124 + green * 0.3576 + blue * 0.1805) / 0.95047,
                        red * 0.2126 + green * 0.7152 + blue * 0.0722,
                        (red * 0.0193 + green * 0.1192 + blue * 0.9505) / 1.08883), axis=-1)
        pivot = np.where(xyz > 0.008856, xyz ** (1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
        x, y, z = pivot[..., 0], pivot[..., 1], pivot[..., 2]
        encoded = np.stack((np.clip(116.0 * y - 16.0, 0, 100) * 255.0 / 100.0,
                            np.clip(500.0 * (x - y), -128, 127) + 128.0,
                            np.clip(200.0 * (y - z), -128, 127) + 128.0), axis=-1)
        return Image.fromarray(np.rint(encoded).astype(np.uint8), "RGB")



def choose_magicdrive_splice(first: Mapping[str, Sequence[Image.Image]], second: Mapping[str, Sequence[Image.Image]]) -> SpliceDecision:
    """Choose the lowest-LAB-difference seam across the two native windows.

    The first and second lists are indexed by their native window starts (10 and
    29 respectively).  Candidate tuples are ordered by ``(score, frame)`` so
    an equal score always selects the earliest boundary.
    """
    _validate_windows(first, second)
    scored = []
    for frame in range(30, 71):
        scores = []
        for camera in CAMERAS:
            # The native MagicDrive seam contract indexes the first window at
            # f-11 (the adjacent first frame f-1) and the second at f-29 for
            # output boundary f.
            scores.append(_mean_lab_l1(first[camera][frame - 11], second[camera][frame - 29]))
        scored.append((sum(scores) / len(scores), frame))
    score, frame = min(scored)
    return SpliceDecision(frame, score, metric="mean_lab_l1_50x28", candidates=41)


__all__ = [
    "HorizonPlan", "MagicDrivePlan", "OpenDWMPlan", "PanaceaWindow",
    "PanaceaPlan", "SpliceDecision", "horizon_plan", "magicdrive_plan",
    "opendwm_plan", "panacea_plan", "choose_magicdrive_splice",
]

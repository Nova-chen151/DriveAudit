"""Participant traffic prediction and six-camera image generation."""

from pathlib import Path

from driveaudit.resources import resource_root

GENERATION = Path(__file__).resolve().parent
MODEL_DIRS = {
    name.lower().replace("-", ""): resource_root() / "models/generation/image" / name
    for name in ("BaseDreamer", "DreamForge", "HorizonDrive", "MagicDrive", "OpenDWM", "Panacea", "MagicDrive-V2")
}
MODELS = tuple(MODEL_DIRS)

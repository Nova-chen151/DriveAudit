"""Participant traffic prediction and six-camera image generation."""

from pathlib import Path

from driveaudit.resources import resource_root

VENDOR = resource_root() / "vendor" / "generation"
MODELS = ("basedreamer", "dreamforge", "horizondrive", "magicdrive", "opendwm", "panacea", "magicdrivev2")

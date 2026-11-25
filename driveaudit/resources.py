"""Locate native assets in a source checkout or an installed wheel."""
from pathlib import Path


def resource_root():
    source = Path(__file__).resolve().parents[1]
    if (source / "models/generation/trajectory/CAT-K/run_inference.py").is_file():
        return source
    return Path(__file__).resolve().parent / "_assets"

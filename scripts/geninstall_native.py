#!/usr/bin/env python3
"""Install native DriveAudit engines from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaudit.generation.native import main


if __name__ == "__main__":
    main()

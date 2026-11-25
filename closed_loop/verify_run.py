"""Command-line entry point for driveaudit.closed_loop.verify_run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaudit.closed_loop.verify_run import main


if __name__ == "__main__":
    main()

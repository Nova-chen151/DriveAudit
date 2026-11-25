"""Command-line entry point for driveaudit.closed_loop.configure."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaudit.closed_loop.configure import main


if __name__ == "__main__":
    main()

"""Compatibility entry point for the model-owned DriveAudit adapter."""
import sys
from driveaudit.agents.registry import load_adapter, run_adapter

if __name__ == "__main__":
    run_adapter("uniad")
else:
    sys.modules[__name__] = load_adapter("uniad")

"""Launch one of the six native ego inference services in its matching environment."""
import argparse
import os
import runpy
import sys

from .runtime import model_root
from .registry import POLICIES, resources


SERVICES = {key: value[0] for key, value in POLICIES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("policy", choices=tuple(SERVICES))
    parser.add_argument("--preflight", action="store_true", help="Check model config and required released assets")
    if sys.argv[1:] in ([], ["--help"], ["-h"]):
        parser.print_help()
        return
    options, remaining = parser.parse_known_args()
    if options.preflight:
        config, assets = resources(options.policy)
        for path in (config, *assets.values()):
            if not path.is_file() or not path.stat().st_size:
                parser.error(f"Missing {path}")
        print(f"{options.policy}: found config and {len(assets)} required assets")
        return
    path = model_root(SERVICES[options.policy])
    sys.path.insert(0, str(path))
    if os.environ.get("NUPLAN_DEVKIT_ROOT"):
        sys.path.insert(1, os.environ["NUPLAN_DEVKIT_ROOT"])
    os.environ["DRIVEAUDIT_SENSOR_PROTOCOL"] = "camera_only"
    os.environ["DRIVEAUDIT_DETERMINISTIC_AGENT"] = "1"
    if options.policy == "uniad":
        os.environ.setdefault("UNIAD_SR_OOM_FALLBACK", "0")
    if options.policy in ("uniad", "vad") and "--no-visualization" not in remaining:
        remaining.append("--no-visualization")
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module("driveaudit.agents.native." + options.policy, run_name="__main__")


if __name__ == "__main__":
    main()

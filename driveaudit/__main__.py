import argparse
from pathlib import Path


def main():
    from .pipeline import Pipeline, STEPS
    parser = argparse.ArgumentParser(prog="driveaudit", description="Run a participant submission and paired organizer evaluation.")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Execute an inclusive range of pipeline stages")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--start", choices=STEPS, default=STEPS[0])
    run.add_argument("--stop", choices=STEPS, default=STEPS[-1])
    run.add_argument("--plan", action="store_true", help="Print resolved commands without loading any models")
    panel = sub.add_parser("panel", help="Run the six-policy matched replay panel")
    parts = panel.add_subparsers(dest="panel_action", required=True)
    infer = parts.add_parser("infer", help="Replay one side of one native policy")
    score = parts.add_parser("score", help="Join and score all twelve branches")
    from .agents.registry import POLICIES
    infer.add_argument("--policy", required=True, choices=tuple(POLICIES))
    infer.add_argument("--source", required=True, choices=("generated", "real"))
    for command in (infer, score):
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.action == "panel":
        from .panel import run as run_panel
        run_panel(args.config, args.panel_action, getattr(args, "policy", None), getattr(args, "source", None), args.plan)
        return
    Pipeline(args.config).run(args.start, args.stop, args.plan)


if __name__ == "__main__":
    main()

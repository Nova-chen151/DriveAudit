import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired camera policy predictions against native traffic and logged routes.")
    for name in ("generated", "real"):
        parser.add_argument("--"+name, type=Path, required=True, help=f"{name} policy JSONL")
    parser.add_argument("--scenes", type=Path, required=True, help="JSON mapping case IDs to paired scene directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", type=Path, help="Matched per-scene raw quality and support records")
    parser.add_argument("--trs", type=Path, help="Native paired population TRS result")
    args = parser.parse_args()
    from .score import run
    run(args)
    print(args.output/"summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the one-scene, seven-renderer integration table from scored replay outputs."""
import argparse
from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaudit.generation import MODELS


DISPLAY = dict(basedreamer="BaseDreamer", dreamforge="DreamForge", horizondrive="HorizonDrive",
               magicdrive="MagicDrive", magicdrivev2="MagicDrive-V2", opendwm="OpenDWM", panacea="Panacea")


def summarize(root, case):
    records = []
    reference = None
    for renderer in MODELS:
        summary = json.loads((root / renderer / "paired/summary.json").read_text())
        if summary["protocol"] != "paired_logged_replay_one_policy" or summary["policy_identity"]["policy"] != "drivor":
            raise ValueError(f"Unexpected replay protocol for {renderer}")
        scenes = summary["scenes"]
        if len(scenes) != 1 or scenes[0]["case_id"] != case:
            raise ValueError(f"Expected only {case} for {renderer}")
        row = scenes[0]
        if reference is None:
            reference = summary["hashes"]["real"]
        if reference != summary["hashes"]["real"]:
            raise ValueError("Expected the same DrivoR reference on every renderer")
        gen = row["generated"]
        records.append(dict(renderer=DISPLAY[renderer], case_id=case, **gen["quality_raw"],
                            population_trs=summary["TRS_population"], rpdms_real=row["real"]["rpdms"],
                            rpdms_generated=gen["rpdms"], route_real=row["real"]["route_completion"],
                            route_generated=gen["route_completion"], paired_windows=row["paired_windows"],
                            legal_windows=row["legal_windows"], TE=row["single_policy_TE"],
                            single_policy_QTES=summary["single_policy_QTES_diagnostic"],
                            status=summary["single_policy_QTES_status"]))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    rows = summarize(args.root, args.case)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

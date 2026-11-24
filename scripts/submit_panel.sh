#!/usr/bin/env bash
# Run on staged scenes after the renderer and the organizer quality stage.
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:-configs/pipeline.json}
driver=${DRIVEAUDIT_DRIVER_PYTHON:-$HOME/miniconda3/bin/python3}
log_dir=${DRIVEAUDIT_JOB_LOG_DIR:-$PWD/work/slurm}
mkdir -p "$log_dir"
common=(--parsable --chdir="$PWD" --output="$log_dir/%x-%A-%a.log" --kill-on-invalid-dep=yes)
deps=()
if [[ -n "${2:-}" ]]; then deps+=(--dependency="afterok:$2"); fi
inference=$(sbatch "${common[@]}" "${deps[@]}" --job-name=da-panel-ego --array=0-11%4 \
  -p L40 --gres=gpu:l40:1 -c 6 --mem=64G scripts/panel_job.sh "$driver" "$config" infer)
scoring=$(sbatch "${common[@]}" --job-name=da-panel-score --dependency="afterok:$inference" \
  -p intel -c 6 --mem=32G scripts/panel_job.sh "$driver" "$config" score)
printf '%s\n' "six-policy inference $inference" "six-policy score $scoring"

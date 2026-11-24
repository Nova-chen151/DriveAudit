#!/usr/bin/env bash
# Submit the two-scene participant and organizer pipeline on a Slurm cluster.
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:-configs/pipeline.json}
mode=${2:-fresh}
case "$mode" in
    fresh) first=traffic ;;
    staged) first=render ;;
    rendered) first=policy-generated ;;
    *) echo "Usage: submit_demo.sh CONFIG [fresh|staged|rendered]" >&2; exit 2 ;;
esac
driver=${DRIVEAUDIT_DRIVER_PYTHON:-$HOME/miniconda3/bin/python3}
log_dir=${DRIVEAUDIT_JOB_LOG_DIR:-$PWD/work/slurm}
mkdir -p "$log_dir"
renderer=$("$driver" -c 'import json,sys;print(json.load(open(sys.argv[1]))["renderer"])' "$config")
"$driver" -m driveaudit run --config "$config" --start "$first" --plan >/dev/null
submit() {
    local name=$1 first=$2 last=$3 dependency=$4
    shift 4
    local flags=(--parsable --chdir="$PWD" --job-name="da-$name" --output="$log_dir/%x-%j.log" --kill-on-invalid-dep=yes)
    if [[ -n "$dependency" ]]; then flags+=(--dependency="afterok:$dependency"); fi
    sbatch "${flags[@]}" "$@" scripts/run_stage.sh "$driver" "$config" "$first" "$last"
}
stage=""
render=""
if [[ "$mode" = fresh ]]; then
    traffic=$(submit traffic traffic traffic "" -p L40 --gres=gpu:l40:1 -c 4 --mem=16G)
    stage=$(submit stage stage stage "$traffic" -p intel -c 4 --mem=16G)
    printf '%s\n' "traffic $traffic" "stage $stage"
fi
if [[ "$mode" != rendered ]]; then
    cards=2; cores=12; memory=96G
    if [[ "$renderer" = magicdrivev2 ]]; then cards=4; cores=24; memory=256G; fi
    render=$(submit render render render "$stage" -p L40 --gres="gpu:l40:$cards" -c "$cores" --mem="$memory")
    printf '%s\n' "render $render"
fi
generated=$(submit policy-gen policy-generated policy-generated "$render" -p L40 --gres=gpu:l40:1 -c 6 --mem=64G)
real=$(submit policy-real policy-real policy-real "$stage" -p L40 --gres=gpu:l40:1 -c 6 --mem=64G)
pair=$(submit pair pair pair "$render" -p intel -c 4 --mem=16G)
quality=$(submit quality quality quality "$pair" -p L40 --gres=gpu:l40:1 -c 6 --mem=48G)
score=$(submit score score score "$generated:$real:$quality" -p intel -c 4 --mem=24G)
printf '%s\n' "policy-generated $generated" "policy-real $real" "pair $pair" "quality $quality" "score $score"

#!/usr/bin/env bash
set -euo pipefail
driver=$1
config=$2
if [[ "$3" = score ]]; then
    exec "$driver" -m driveaudit panel score --config "$config"
fi
policies=(drivor uniad vad sparsedrivev2 goalflow diffusiondrivev2)
sources=(generated real)
index=$SLURM_ARRAY_TASK_ID
exec "$driver" -m driveaudit panel infer --config "$config" \
  --policy "${policies[index/2]}" --source "${sources[index%2]}"

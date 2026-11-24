#!/usr/bin/env bash
set -euo pipefail
exec "$1" -m driveaudit run --config "$2" --start "$3" --stop "$4"

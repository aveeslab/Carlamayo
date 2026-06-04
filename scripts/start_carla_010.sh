#!/usr/bin/env bash
set -euo pipefail

carla_root="${CARLA_010_ROOT:-$HOME/Carla-0.10.0}"
carla_exe="$carla_root/CarlaUnreal.sh"

for arg in "$@"; do
  case "${arg,,}" in
    *quality-level*)
      echo "Do not pass quality-level arguments for this CARLA 0.10 setup." >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$carla_exe" ]]; then
  echo "CARLA 0.10 launcher not found or not executable: $carla_exe" >&2
  exit 1
fi

exec "$carla_exe" -RenderOffScreen "$@"

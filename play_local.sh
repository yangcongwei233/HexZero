#!/usr/bin/env bash
set -euo pipefail
hex_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${HEX_PYTHON:-}" ]]; then
  hex_python="$HEX_PYTHON"
elif [[ -x "$HOME/Tools/miniforge3/envs/lab232/bin/python" ]]; then
  hex_python="$HOME/Tools/miniforge3/envs/lab232/bin/python"
else
  hex_python="python3"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$hex_python" "$hex_root/hex_play_server.py" "$@"

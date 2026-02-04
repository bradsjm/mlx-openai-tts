#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_UVICORN="$ROOT_DIR/.venv/bin/uvicorn"

export PYTHONWARNINGS

if [[ -x "$VENV_UVICORN" ]]; then
  exec "$VENV_UVICORN" mlx_openai_tts.server:app --host "$HOST" --port "$PORT" --workers 1
elif command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn mlx_openai_tts.server:app --host "$HOST" --port "$PORT" --workers 1
else
  echo "uvicorn not found. Create the venv and install deps:" >&2
  echo "  cd \"$ROOT_DIR\" && uv venv --python 3.12 && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

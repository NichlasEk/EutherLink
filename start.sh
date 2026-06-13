#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VOXCPM_ROOT="${VOXCPM_ROOT:-/home/nichlas/ai/voxcpm2/VoxCPM}"
PYTHON="${PYTHON:-$VOXCPM_ROOT/.venv/bin/python}"

export PYTHONPATH="$VOXCPM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export EUTHERLINK_HOST="${EUTHERLINK_HOST:-0.0.0.0}"
export EUTHERLINK_PORT="${EUTHERLINK_PORT:-8765}"
export EUTHERLINK_DATA_DIR="${EUTHERLINK_DATA_DIR:-/home/nichlas/EutherLink/data}"
export EUTHERLINK_MODEL_PATH="${EUTHERLINK_MODEL_PATH:-/home/nichlas/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/e8b928065859f2869644c1e2881cbd21f888c659}"

exec "$PYTHON" eutherlink.py "$@"

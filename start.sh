#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export EUTHERLINK_CONFIG="${EUTHERLINK_CONFIG:-$PWD/eutherlink.toml}"

toml_value() {
  python - "$EUTHERLINK_CONFIG" "$1" "$2" <<'PY'
import sys
import tomllib

path, section, key = sys.argv[1:4]
with open(path, "rb") as handle:
    config = tomllib.load(handle)
value = config.get(section, {}).get(key, "")
if value is not None:
    print(value)
PY
}

PYTHON="${PYTHON:-$(toml_value server python)}"
PYTHON="${PYTHON:-python}"
CONFIG_PYTHONPATH="$(toml_value server pythonpath)"
if [[ -n "$CONFIG_PYTHONPATH" ]]; then
  export PYTHONPATH="$CONFIG_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$PYTHON" eutherlink.py "$@"

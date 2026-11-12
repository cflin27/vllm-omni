#!/usr/bin/env bash
set -euo pipefail
: "${PUBLIC_PYTHON:?Set PUBLIC_PYTHON to the configured isolated runtime Python}"
: "${OMNI_SOURCE_ROOT:?Set OMNI_SOURCE_ROOT to the selected public checkout}"
export PYTHONPATH="${OMNI_INSTRUMENT_ROOT:+${OMNI_INSTRUMENT_ROOT}:}${OMNI_SOURCE_ROOT}:${PYTHONPATH:-}"
exec "$PUBLIC_PYTHON" "$@"

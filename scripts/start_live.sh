#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${LIVE_TRADING:-false}" != "true" ]]; then
  echo "LIVE_TRADING=true is required before live mode can start."
  exit 1
fi

export APEX_MODE=LIVE
PYTHONPATH=. python -m src.main

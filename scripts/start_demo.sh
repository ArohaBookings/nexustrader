#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export APEX_MODE=DEMO
export LIVE_TRADING=false
PYTHONPATH=. python -m src.main

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PRESET="${1:-realistic}"
PYTHONPATH=. python -m src.main --backtest --preset "$PRESET"

#!/usr/bin/env bash
# Full walk-forward study across every implemented strategy.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/s151 backtest --output "${1:-results}"

#!/usr/bin/env bash
# Bring up QuestDB, install the package and load the study universe.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e ".[dev]"

if ! curl -sf -m 3 "http://localhost:9000/exec?query=select%201" >/dev/null; then
  echo "starting QuestDB..."
  docker compose up -d questdb
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://localhost:9000/exec?query=select%201" >/dev/null && break
    sleep 2
  done
fi

echo "loading bars into stooq.daily..."
./.venv/bin/s151 load
./.venv/bin/s151 status

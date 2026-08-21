#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[OpenHumSim] macOS bootstrap"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first with the official Astral installer:"
  echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'
  exit 2
fi

uv python install 3.12
uv sync --extra local

echo
echo "[OpenHumSim] preflight"
uv run openhumsim doctor

echo
echo "[OpenHumSim] 30-minute smoke simulation"
uv run openhumsim demo --scenario baseline --minutes 30 --seed 42 >/dev/null

echo
echo "READY. In VSCode select interpreter: .venv/bin/python"
echo "Then run: uv run openhumsim demo --scenario oral_glucose_75g --minutes 180"

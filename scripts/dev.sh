#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/backend"
PYTHONPATH="$ROOT/backend" "$PYTHON_BIN" -m uvicorn app.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT" &

cd "$ROOT/frontend"
npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &

wait

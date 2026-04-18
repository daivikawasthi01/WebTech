#!/usr/bin/env bash
# start.sh — Start FastAPI backend and Vite frontend in parallel.
# Usage:
#   ./start.sh          — starts both servers
#   ./start.sh --api    — backend only
#   ./start.sh --ui     — frontend only

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stability Fix: Resolves segmentation faults between PyTorch and XGBoost
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

start_api() {
  echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
  cd "$ROOT"
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload \
    --log-level info
}

start_ui() {
  echo "▶ Starting Vite frontend on http://localhost:5173 ..."
  cd "$ROOT/frontend"
  if [ ! -d node_modules ]; then
    echo "  Installing npm dependencies..."
    npm install
  fi
  npm run dev
}

case "${1:-}" in
  --api) start_api ;;
  --ui)  start_ui  ;;
  *)
    # Run both in parallel; Ctrl-C kills both
    trap 'kill 0' SIGINT SIGTERM
    start_api &
    sleep 1   # give uvicorn a moment to bind the port
    start_ui  &
    wait
    ;;
esac

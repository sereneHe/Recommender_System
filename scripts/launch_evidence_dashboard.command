#!/usr/bin/env bash
# One-click launcher for the Evidence Tree dashboard (step-5 replacement).
# Starts the local server (which syncs the server evidence on open) and opens it.
cd "$(dirname "$0")/.."
PORT="${PORT:-8770}"
PY="${PYTHON_BIN:-python3}"

if ! curl -s "http://127.0.0.1:${PORT}/api/state" >/dev/null 2>&1; then
  "$PY" scripts/serve_evidence_dashboard.py --port "$PORT" >/tmp/evidence-dashboard.log 2>&1 &
  sleep 2
fi
open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || echo "open http://127.0.0.1:${PORT}/"

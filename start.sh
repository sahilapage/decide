#!/usr/bin/env bash
# One command to serve Decide on a clean CPU-only Linux box:
#   ./start.sh            (listens on 0.0.0.0:8000)
#   PORT=9000 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
# One worker, one model in memory. ONNX Runtime uses the 4 cores for each request.
exec .venv/bin/uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --workers 1 --loop uvloop --http httptools --no-access-log --timeout-keep-alive 120

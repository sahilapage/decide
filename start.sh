#!/usr/bin/env bash
# One command to serve Decide on a clean CPU-only Linux box:
#   ./start.sh            (listens on 0.0.0.0:8000)
#   PORT=9000 ./start.sh
# Installs the five serving packages into .venv: from PyPI when reachable,
# otherwise from prebuilt Linux x86_64 wheels (CPython 3.10-3.12) in ./wheels,
# fetched from the repo's GitHub release if not already there. OFFLINE=1
# skips PyPI.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
if .venv/bin/python -c "import fastapi, uvicorn, numpy, onnxruntime, tokenizers, uvloop, httptools" 2>/dev/null; then
  echo "start.sh: serving packages already installed"
elif [ "${OFFLINE:-0}" != "1" ] && .venv/bin/pip install -q --disable-pip-version-check --retries 3 --timeout 30 -r requirements.txt; then
  echo "start.sh: installed from PyPI"
else
  echo "start.sh: PyPI not reachable (or OFFLINE=1); using the prebuilt wheels"
  if [ ! -d wheels ]; then
    mkdir -p wheels
    curl -fsSL --retry 3 "${WHEELS_URL:-https://github.com/sahilapage/decide/releases/download/wheels-v1/decide-wheels-linux-x86_64.tar.gz}" \
      | tar xz -C wheels
  fi
  .venv/bin/pip install -q --disable-pip-version-check --no-index --find-links wheels -r requirements.txt
fi
# One worker, one model in memory. ONNX Runtime uses the 4 cores for each request.
exec .venv/bin/uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --workers 1 --loop uvloop --http httptools --no-access-log --timeout-keep-alive 120

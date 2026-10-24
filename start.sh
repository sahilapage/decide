#!/usr/bin/env bash
# Install the packages and start the Decide server on 0.0.0.0:8000.
cd "$(dirname "$0")"
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

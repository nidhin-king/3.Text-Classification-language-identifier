#!/usr/bin/env bash
# Starts the FastAPI backend (background) and the Streamlit UI (foreground).
# Used as the default CMD of the Docker image.
#
# Comments in this script use their own line (no inline comments).

set -euo pipefail

echo "Starting FastAPI backend on :8000 ..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap to stop the backend when the container stops.
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

echo "Starting Streamlit UI on :8501 ..."
exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0

#!/usr/bin/env bash
# Simple launcher for macOS/Linux.
set -e

# Change to the directory where this script lives
cd "$(dirname "$0")"

VENV_DIR="$(pwd)/.venv"

# Prefer the virtual environment's Python if it exists
if [ -x "$VENV_DIR/bin/python" ]; then
  "$VENV_DIR/bin/python" main.py
else
  echo "[INFO] No .venv found yet. We'll use system Python."
  python3 main.py
fi

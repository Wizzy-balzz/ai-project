#!/usr/bin/env bash
# ==============================================================================
# Single-command launcher for Graph-Based Attack Path Predictor AI
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "================================================================="
echo "  Starting Attack Path Predictor AI..."
echo "================================================================="

# Check if numpy is installed; if not, install all dependencies
if ! python3 -c "import numpy, torch, fastapi, uvicorn" >/dev/null 2>&1; then
    echo "[*] Initializing dependencies..."
    
    # Try CPU torch first for speed and low bandwidth
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages 2>/dev/null || \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true

    # Install remaining requirements
    python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || \
    python3 -m pip install -r requirements.txt 2>/dev/null || true
fi

# Run combined frontend & backend
python3 run.py

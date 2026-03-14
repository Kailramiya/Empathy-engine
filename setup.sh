#!/bin/bash
# ──────────────────────────────────────────────────────────────
# The Empathy Engine — Quick Setup Script
# ──────────────────────────────────────────────────────────────

set -e

echo ""
echo "============================================================"
echo "   The Empathy Engine — Setup"
echo "============================================================"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Check for espeak-ng
echo "[3/4] Checking TTS engines..."
if command -v espeak-ng &> /dev/null; then
    echo "  ✓ espeak-ng found (offline fallback available)"
else
    echo "  ⚠ espeak-ng not found. Install with: sudo apt-get install espeak-ng"
    echo "    Edge-TTS (online) will be used as primary engine."
fi

# Create output directory
mkdir -p output

echo "[4/4] Setup complete!"
echo ""
echo "============================================================"
echo "  Usage:"
echo "    source venv/bin/activate"
echo ""
echo "    # Web UI (recommended)"
echo "    python run_web.py"
echo "    # Then open http://localhost:8000"
echo ""
echo "    # CLI"
echo "    python cli.py \"I'm so happy today!\""
echo "============================================================"
echo ""

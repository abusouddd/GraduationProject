#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Smart Glasses AI — Raspberry Pi 4 One-Click Setup
# ═══════════════════════════════════════════════════════════════
# Run this script ON the Raspberry Pi:
#   bash install_pi.sh
# ═══════════════════════════════════════════════════════════════

set -e

echo "=========================================="
echo "  Smart Glasses AI — Pi 4 Setup"
echo "=========================================="

# ── 1. System packages ──
echo ""
echo "[1/6] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-libcamera \
    python3-picamera2 \
    espeak \
    espeak-ng \
    libcap-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils

# ── 2. Python virtual environment ──
echo ""
echo "[2/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── 3. Make system Python packages visible in venv ──
echo ""
echo "[3/6] Linking system packages into venv..."
PYVER=$(python3 -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
echo "/usr/lib/python${PYVER}/dist-packages" > venv/lib/python${PYVER}/site-packages/system-packages.pth

# ── 4. Upgrade pip ──
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip

# ── 5. Install Python dependencies ──
echo ""
echo "[5/6] Installing Python packages (this takes a while)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pi.txt

# ── 6. Verify ──
echo ""
echo "[6/6] Verifying installation..."

echo ""
python3 -c "
import sys
print('Python:', sys.version)
try:
    import cv2; print('OpenCV:', cv2.__version__)
except: print('OpenCV: NOT INSTALLED')
try:
    import ultralytics; print('Ultralytics: OK')
except: print('Ultralytics: NOT INSTALLED')
try:
    import pyttsx3; print('pyttsx3: OK')
except: print('pyttsx3: NOT INSTALLED')
try:
    import torch; print('PyTorch:', torch.__version__)
except: print('PyTorch: NOT INSTALLED')
try:
    from picamera2 import Picamera2; print('picamera2: OK')
except: print('picamera2: NOT INSTALLED')
try:
    import libcamera; print('libcamera: OK')
except: print('libcamera: NOT INSTALLED')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Verify camera:"
echo "    libcamera-hello --list-cameras"
echo ""
echo "  Run the app (with display):"
echo "    source venv/bin/activate"
echo "    python3 src/detect.py --device pi --language en"
echo ""
echo "  Run headless (via SSH, no display window):"
echo "    source venv/bin/activate"
echo "    python3 src/detect.py --device pi --no-display --language en"
echo ""
echo "  Run in Arabic:"
echo "    source venv/bin/activate"
echo "    python3 src/detect.py --device pi --no-display --language ar"
echo ""

#!/bin/bash
# Create local Python venv with all dependencies for test scripts.
# Installs openvino + openvino-genai from the OV/GenAI builds,
# plus librosa and numpy for whisper tests.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/venv"

# --- Read saved paths ---
for pathfile in .ovpath .genaipath; do
    if [[ ! -f "$HERE/$pathfile" ]]; then
        echo "ERROR: $pathfile not found. Run ./build.sh first."
        exit 1
    fi
done
OV_SRC="$(cat "$HERE/.ovpath")"
GENAI_SRC="$(cat "$HERE/.genaipath")"

# --- Source setupvars for OV python ---
SETUPVARS="$OV_SRC/build-ninja/install/setupvars.sh"
if [[ ! -f "$SETUPVARS" ]]; then
    echo "ERROR: setupvars.sh not found. Build OpenVINO first."
    exit 1
fi
set +u
source "$SETUPVARS"
set -u

# --- Create venv ---
if [[ -d "$VENV_DIR" ]]; then
    echo "Venv already exists at $VENV_DIR"
    echo "To recreate, delete it and re-run: rm -rf venv && ./setup_venv.sh"
    exit 0
fi

echo "Creating venv..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip -q

# Install OV wheel from build if available, otherwise from PyPI
OV_WHEEL=$(find "$OV_SRC/build-ninja" -name "openvino-*.whl" 2>/dev/null | head -1)
if [[ -n "$OV_WHEEL" ]]; then
    pip install "$OV_WHEEL" -q
else
    pip install openvino -q
fi

# Install GenAI - try build wheel, then editable install, then PyPI
GENAI_WHEEL=$(find "$GENAI_SRC/build-ninja" -name "openvino_genai-*.whl" 2>/dev/null | head -1)
if [[ -n "$GENAI_WHEEL" ]]; then
    pip install "$GENAI_WHEEL" -q
elif [[ -f "$GENAI_SRC/pyproject.toml" ]]; then
    pip install -e "$GENAI_SRC" -q 2>/dev/null || pip install openvino-genai -q
else
    pip install openvino-genai -q
fi

# Install remaining deps from requirements.txt
pip install -r "$HERE/requirements.txt" -q

echo ""
echo "Venv created: $VENV_DIR"
python -c "import openvino; print(f'  openvino {openvino.__version__}')" 2>/dev/null || true
python -c "import openvino_genai; print(f'  openvino-genai installed')" 2>/dev/null || true
python -c "import librosa; print(f'  librosa {librosa.__version__}')" 2>/dev/null || true

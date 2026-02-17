#!/bin/bash
# Build npuw_model_generator_demo from OpenVINO source.
# On first run, prompts for the OpenVINO source path and saves it to .ovpath.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVPATH_FILE="$HERE/.ovpath"
BUILD_DIR="$HERE/build"
SRC_DIR="$HERE/src"

# --- Resolve OpenVINO source path ---
if [[ -f "$OVPATH_FILE" ]]; then
    OV_SRC="$(cat "$OVPATH_FILE")"
    if [[ ! -d "$OV_SRC/src/plugins/intel_npu" ]]; then
        echo "ERROR: Saved path '$OV_SRC' no longer valid (missing intel_npu plugin dir)"
        echo "Delete .ovpath and re-run, or edit it manually."
        exit 1
    fi
else
    echo "OpenVINO source path not configured."
    read -rp "Enter path to OpenVINO source tree: " OV_SRC
    OV_SRC="${OV_SRC/#\~/$HOME}"
    OV_SRC="$(realpath "$OV_SRC")"
    if [[ ! -d "$OV_SRC/src/plugins/intel_npu" ]]; then
        echo "ERROR: '$OV_SRC' does not look like an OpenVINO source tree"
        exit 1
    fi
    echo "$OV_SRC" > "$OVPATH_FILE"
    echo "Saved to .ovpath"
fi

# --- Source setupvars.sh (sets up OpenVINO_DIR for find_package) ---
SETUPVARS="$OV_SRC/build-ninja/install/setupvars.sh"
if [[ ! -f "$SETUPVARS" ]]; then
    echo "ERROR: setupvars.sh not found at $SETUPVARS"
    echo "Build OpenVINO first, or set OpenVINO_DIR manually."
    exit 1
fi
set +u
source "$SETUPVARS"
set -u

# --- Build ---
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DOPENVINO_SOURCE_DIR="$OV_SRC" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "Built: $BUILD_DIR/npuw_model_generator_demo"

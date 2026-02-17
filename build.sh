#!/bin/bash
# Build npuw_model_generator_demo from OpenVINO source.
# On first run, prompts for OpenVINO and GenAI source paths.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$HERE/build"
SRC_DIR="$HERE/src"

# --- Helper: resolve and save a path ---
resolve_path() {
    local label="$1" file="$2" check_dir="$3"

    if [[ -f "$file" ]]; then
        local saved
        saved="$(cat "$file")"
        if [[ ! -d "$saved/$check_dir" ]]; then
            echo "ERROR: Saved $label path '$saved' no longer valid (missing $check_dir)"
            echo "Delete $(basename "$file") and re-run."
            exit 1
        fi
        echo "$saved"
        return
    fi

    echo "$label source path not configured." >&2
    read -rp "Enter path to $label source tree: " input
    input="${input/#\~/$HOME}"
    input="$(realpath "$input")"
    if [[ ! -d "$input/$check_dir" ]]; then
        echo "ERROR: '$input' does not look like a $label source tree" >&2
        exit 1
    fi
    echo "$input" > "$file"
    echo "Saved to $(basename "$file")" >&2
    echo "$input"
}

# --- Resolve paths ---
OV_SRC=$(resolve_path "OpenVINO" "$HERE/.ovpath" "src/plugins/intel_npu")
GENAI_SRC=$(resolve_path "OpenVINO GenAI" "$HERE/.genaipath" "tools/llm_bench")

# --- Source setupvars.sh ---
SETUPVARS="$OV_SRC/build-ninja/install/setupvars.sh"
if [[ ! -f "$SETUPVARS" ]]; then
    echo "ERROR: setupvars.sh not found at $SETUPVARS"
    echo "Build OpenVINO first."
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
echo "OV source: $OV_SRC"
echo "GenAI source: $GENAI_SRC"

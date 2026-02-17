#!/bin/bash
# Download tokenizer models from Intel Artifactory for NPUW model builder testing.
# Always uses the latest release. Re-downloads if a newer release is available.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$HERE/models"
BASE_URL="https://af01p-ir.devtools.intel.com/artifactory/ir-public-models-ir-local/ov-genai-models/releases"
VERSION_FILE="$MODELS_DIR/.release_version"

# --- Find latest release ---
echo "Checking latest release..."
LATEST=$(curl -sk "$BASE_URL/" | grep -oP 'href="\K[^"]+(?=/")' | sort -V | tail -1)
if [[ -z "$LATEST" ]]; then
    echo "ERROR: Could not determine latest release from $BASE_URL"
    exit 1
fi
echo "Latest release: $LATEST"

# Check if we already have this version
if [[ -f "$VERSION_FILE" ]]; then
    CURRENT=$(cat "$VERSION_FILE")
    if [[ "$CURRENT" == "$LATEST" ]]; then
        echo "Models already up to date ($LATEST)"
        echo "To force re-download, delete $MODELS_DIR and re-run."
        exit 0
    fi
    echo "Upgrading from $CURRENT -> $LATEST"
    rm -rf "$MODELS_DIR"
fi

mkdir -p "$MODELS_DIR"

# --- Smallest model for each type ---
declare -A MODELS=(
    ["LLM/MiniCPM4-0.5B_int4_sym_group-1_dyn_stateful.tgz"]="llm"
    ["VLM/Qwen2.5-VL-3B-Instruct_int4_sym_group128_dyn_stateful.tgz"]="vlm"
    ["whisper/whisper-tiny_fp16_dyn_stateful.tgz"]="whisper"
    ["RAG/Qwen3-Embedding-0.6B_int4_sym_group-1_dyn_stateful.tgz"]="embedding-decoder"
    ["RAG/Facebook_Contriever_int4_sym_group-1.tgz"]="embedding-encoder"
)

FAIL=0

for path in "${!MODELS[@]}"; do
    label="${MODELS[$path]}"
    file="${path##*/}"
    url="$BASE_URL/$LATEST/$path"
    dest="$MODELS_DIR/$label"

    printf "  %-20s %s ... " "$label" "$file"

    tmp="$MODELS_DIR/$file"
    if curl -sk -o "$tmp" "$url" && [[ -s "$tmp" ]]; then
        mkdir -p "$dest"
        tar -xzf "$tmp" -C "$dest" --strip-components=1
        rm -f "$tmp"
        echo "OK ($(du -sh "$dest" | cut -f1))"
    else
        echo "FAILED"
        rm -f "$tmp"
        FAIL=1
    fi
done

# --- Generate test audio (1s silence, 16kHz mono WAV for whisper tests) ---
printf "  %-20s " "test_audio.wav"
python3 -c "
import struct, wave
with wave.open('$MODELS_DIR/test_audio.wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
" 2>/dev/null && echo "OK" || echo "FAILED"

# --- Save version ---
echo "$LATEST" > "$VERSION_FILE"

echo ""
echo "Models downloaded to: $MODELS_DIR/ (release: $LATEST)"
ls -d "$MODELS_DIR"/*/
[[ $FAIL -eq 0 ]] || exit 1

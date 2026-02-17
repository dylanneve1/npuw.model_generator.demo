#!/bin/bash
# Quick whisper NPU test for generated models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"

WHISPER_GENAI="$WORKSPACE/applications.ai.vpu-accelerators.npuw/samples/whisper/whisper_genai.py"
TEST_AUDIO="$WORKSPACE/test_audio.wav"
CONFIGS_DIR="$SCRIPT_DIR/configs"

python "$WHISPER_GENAI" -c "$CONFIGS_DIR/whisper.json" /tmp/npuw_model_builder_test/whisper_default "$TEST_AUDIO" NPU
python "$WHISPER_GENAI" -c "$CONFIGS_DIR/whisper.json" /tmp/npuw_model_builder_test/whisper_fp16 "$TEST_AUDIO" NPU

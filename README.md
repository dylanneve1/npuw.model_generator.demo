# npuw.model_generator.demo

Standalone build and test harness for the OpenVINO NPUW synthetic model generator. Generates LLM, VLM, Whisper, and Embedding models for NPUW partitioning and inference testing.

## Prerequisites

- OpenVINO source tree (with NPU plugin) — built with `ENABLE_TESTS=ON`
- OpenVINO GenAI source tree (for `llm_bench`)

## Setup

```bash
# 1. Build the generator (prompts for OV + GenAI source paths on first run)
./build.sh

# 2. Create Python venv with openvino, openvino-genai, librosa, numpy
./setup_venv.sh

# 3. Download tokenizer models from Intel Artifactory (smallest per type)
./download_models.sh
```

Paths are saved to `.ovpath` and `.genaipath` for subsequent runs. Models are downloaded to `models/`, venv is created in `venv/`.

## Usage

### Generate and test all model configurations

```bash
./test_model_builder.sh
```

Options:
- `--quick` — CPU only (skip NPU backends)
- `--dump` — generate NPUW subgraph dumps after testing
- `--generate-only` — generate models without testing
- `--filter PATTERN` — only run configs matching pattern

### Test whisper models on NPU

```bash
./test_whisper_models_npu.sh
```

## Structure

```
build.sh                    # Build generator (manages .ovpath/.genaipath)
setup_venv.sh               # Create Python venv with all dependencies
download_models.sh          # Download tokenizer models from Artifactory
src/
  demo_model_generator.cpp  # Generator CLI source
  CMakeLists.txt            # Standalone cmake (links model_builder from OV source)
tools/
  whisper_genai.py          # Whisper inference test script
configs/                    # NPUW/HFA/Pyramid/Whisper/Embedding JSON configs
test_model_builder.sh       # Comprehensive test suite
test_whisper_models_npu.sh  # Whisper NPU quick test
```

## Downloaded models

| Type | Model | Size |
|------|-------|------|
| LLM | MiniCPM4-0.5B (int4) | ~300MB |
| VLM | Qwen2.5-VL-3B-Instruct (int4) | ~2GB |
| Whisper | whisper-tiny (fp16) | ~150MB |
| Embedding (decoder) | Qwen3-Embedding-0.6B (int4) | ~400MB |
| Embedding (encoder) | Facebook Contriever (int4) | ~100MB |

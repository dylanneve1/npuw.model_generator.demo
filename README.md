# npuw.model_generator.demo

Standalone build and test harness for the OpenVINO NPUW synthetic model generator. Generates LLM, VLM, Whisper, and Embedding models for NPUW partitioning and inference testing.

## Setup

```bash
./build.sh
```

On first run, you'll be prompted for your OpenVINO source tree path. This is saved to `.ovpath` for subsequent builds. OpenVINO must already be built (`setupvars.sh` is sourced automatically).

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
build.sh                    # Build script (manages .ovpath, runs cmake)
src/
  demo_model_generator.cpp  # Generator CLI source
  CMakeLists.txt            # Standalone cmake (links model_builder from OV source)
configs/                    # NPUW/HFA/Pyramid/Whisper/Embedding JSON configs
test_model_builder.sh       # Comprehensive test suite
test_whisper_models_npu.sh  # Whisper NPU quick test
```

## Requirements

- OpenVINO source tree (with NPU plugin) — built with `ENABLE_TESTS=ON`
- OpenVINO GenAI (`llm_bench`) and NPUW whisper sample in the parent workspace
- Tokenizer directories (Llama, Qwen, Whisper, Contriever) in the parent workspace
- Python venv with OpenVINO + GenAI packages (`test-env/` in parent workspace)

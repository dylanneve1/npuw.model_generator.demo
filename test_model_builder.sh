#!/bin/bash
# Test script for NPUW model builder — generates LLM, Whisper, and Embedding models
# with all option combinations and validates them on CPU + NPU.
#
# Usage: ./test_model_builder.sh [--quick] [--dump] [--generate-only] [--filter PATTERN]
#   --quick          Only test CPU (skip NPU configs)
#   --dump           After testing, generate NPUW subgraph dumps for all configs
#   --generate-only  Generate models only, no testing or dumping
#   --filter         Only run configs matching PATTERN (grep-style)
#
# Known limitations (tested but reported as XFAIL, not counted as failures):
#   - Non-standard layer counts on HFA/Pyramid: partitioning may assert

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read saved paths (created by build.sh)
for pathfile in .ovpath .genaipath; do
    if [[ ! -f "$SCRIPT_DIR/$pathfile" ]]; then
        echo "ERROR: $pathfile not found. Run ./build.sh first."
        exit 1
    fi
done
OV_SRC="$(cat "$SCRIPT_DIR/.ovpath")"
GENAI_SRC="$(cat "$SCRIPT_DIR/.genaipath")"

GENERATOR="$SCRIPT_DIR/build/npuw_model_generator_demo"
LLM_BENCH="$GENAI_SRC/tools/llm_bench/benchmark.py"
WHISPER_GENAI="$SCRIPT_DIR/tools/whisper_genai.py"
TOKENIZER_DIR="$SCRIPT_DIR/models/llm"
QWEN_TOKENIZER_DIR="$SCRIPT_DIR/models/vlm"
QWEN3_EMB_TOKENIZER_DIR="$SCRIPT_DIR/models/embedding-decoder"
CONTRIEVER_TOKENIZER_DIR="$SCRIPT_DIR/models/embedding-encoder"
WHISPER_TOKENIZER_DIR="$SCRIPT_DIR/models/whisper"
TEST_AUDIO="$SCRIPT_DIR/models/test_audio.wav"
CONFIGS_DIR="$SCRIPT_DIR/configs"
OUTPUT_DIR="/tmp/npuw_model_builder_test"
VENV="$SCRIPT_DIR/venv/bin/activate"
SETUPVARS="$OV_SRC/build-ninja/install/setupvars.sh"
EMB_TEST_SCRIPT="$OUTPUT_DIR/_test_embedding.py"
EMB_DUMP_SCRIPT="$OUTPUT_DIR/_dump_embedding.py"
BARE_DIR="$OUTPUT_DIR/bare"
DUMP_DIR="$OUTPUT_DIR/dumps"

QUICK=0
FILTER=""
DUMP=0
GENERATE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --dump) DUMP=1; shift ;;
        --generate-only) GENERATE_ONLY=1; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--quick] [--dump] [--generate-only] [--filter PATTERN]"
            echo "  --quick          Only test CPU (skip NPU configs)"
            echo "  --dump           After testing, generate NPUW subgraph dumps"
            echo "  --generate-only  Generate models only, no testing or dumping"
            echo "  --filter         Only run configs matching PATTERN"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Preflight checks ---
for f in "$GENERATOR" "$LLM_BENCH" "$VENV" "$SETUPVARS"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        echo "Run: ./build.sh && ./setup_venv.sh && ./download_models.sh"
        exit 1
    fi
done
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "ERROR: Tokenizer directory not found: $TOKENIZER_DIR"
    exit 1
fi
if [[ ! -d "$QWEN_TOKENIZER_DIR" ]]; then
    echo "ERROR: Qwen tokenizer directory not found: $QWEN_TOKENIZER_DIR"
    exit 1
fi
if [[ ! -d "$QWEN3_EMB_TOKENIZER_DIR" ]]; then
    echo "ERROR: Qwen3-Embedding tokenizer directory not found: $QWEN3_EMB_TOKENIZER_DIR"
    exit 1
fi
if [[ ! -d "$CONTRIEVER_TOKENIZER_DIR" ]]; then
    echo "ERROR: Contriever tokenizer directory not found: $CONTRIEVER_TOKENIZER_DIR"
    exit 1
fi
if [[ ! -f "$CONFIGS_DIR/npuw.json" ]]; then
    echo "ERROR: Config files not found in $CONFIGS_DIR"
    exit 1
fi

# --- Test configurations ---
# Format: "NAME|EXTRA_GENERATOR_ARGS"
# Each config tests a specific builder option or combination.
CONFIGS=(
    # Individual options — weight types
    "fp32_default|"
    "fp16_weights|--weight-type fp16"
    "int8_weights|--weight-type int8"
    "int4_weights|--weight-type int4"
    "int4_group128|--weight-type int4 --group-size 128"
    "int4_group32|--weight-type int4 --group-size 32"

    # Norm types
    "layer_norm|--norm-type layer"

    # FFN types
    "gelu_ffn|--ffn-type gelu"

    # RoPE types
    "interleaved_rope|--rope-type interleaved"

    # GQA (grouped query attention)
    "gqa_4kv|--num-kv-heads 4"
    "gqa_2kv|--num-kv-heads 2"
    "gqa_1kv|--num-kv-heads 1"

    # VLM 3D position_ids (qwen2_5_vl model type — Qwen m-rope)
    "vlm_3d_default|--inputs-embeds --position-ids 3d --vocab-size 151936"
    "vlm_3d_gqa|--inputs-embeds --position-ids 3d --num-kv-heads 4 --vocab-size 151936"
    "vlm_3d_int8|--inputs-embeds --position-ids 3d --weight-type int8 --vocab-size 151936"

    # VLM 2D position_ids (llava model type — standard VLM)
    "vlm_2d_default|--inputs-embeds --position-ids 2d --vocab-size 151936"
    "vlm_2d_gqa|--inputs-embeds --position-ids 2d --num-kv-heads 4 --vocab-size 151936"
    "vlm_2d_int8|--inputs-embeds --position-ids 2d --weight-type int8 --vocab-size 151936"

    # Combined configs — stress different code paths together
    "int8_layer_gelu_interleaved_gqa|--weight-type int8 --norm-type layer --ffn-type gelu --rope-type interleaved --num-kv-heads 2"
    "int4_group128_gqa_interleaved|--weight-type int4 --group-size 128 --num-kv-heads 2 --rope-type interleaved"
    "fp16_gqa1_interleaved_3layer|--weight-type fp16 --num-kv-heads 1 --rope-type interleaved --num-layers 3"
    "vlm_3d_fp16_gelu_gqa_interleaved|--inputs-embeds --position-ids 3d --weight-type fp16 --ffn-type gelu --num-kv-heads 2 --rope-type interleaved --vocab-size 151936"
    "vlm_2d_fp16_gelu_gqa_interleaved|--inputs-embeds --position-ids 2d --weight-type fp16 --ffn-type gelu --num-kv-heads 2 --rope-type interleaved --vocab-size 151936"

    # Whisper configs
    "whisper_default|--type whisper"
    "whisper_fp16|--type whisper --weight-type fp16"

    # Embedding configs — decoder (Qwen3-style: no KV cache, no lm_head, QK-norm)
    "emb_decoder_default|--type embedding --arch decoder --vocab-size 151669"
    "emb_decoder_interleaved|--type embedding --arch decoder --rope-type interleaved --vocab-size 151669"
    "emb_decoder_gqa|--type embedding --arch decoder --num-kv-heads 4 --vocab-size 151669"
    "emb_decoder_int8|--type embedding --arch decoder --weight-type int8 --vocab-size 151669"
    "emb_decoder_fp16_gelu_gqa|--type embedding --arch decoder --weight-type fp16 --ffn-type gelu --num-kv-heads 2 --vocab-size 151669"

    # Embedding configs — encoder (BERT/Contriever-style: post-norm, learned positions, token_type_ids)
    "emb_encoder_default|--type embedding --arch encoder --vocab-size 30522"
    "emb_encoder_int8|--type embedding --arch encoder --weight-type int8 --vocab-size 30522"
    "emb_encoder_fp16|--type embedding --arch encoder --weight-type fp16 --vocab-size 30522"
)

# --- Backend configurations ---
if [[ $QUICK -eq 1 ]]; then
    BACKENDS=("CPU|")
else
    BACKENDS=(
        "CPU|"
        "NPU_npuw|-d NPU -lc $CONFIGS_DIR/npuw.json"
        "NPU_hfa|-d NPU -lc $CONFIGS_DIR/hfa.json"
        "NPU_pyramid|-d NPU -lc $CONFIGS_DIR/pyramid.json"
    )
fi

# --- Known limitation checks ---
# Returns 0 if this config/backend combo is a known limitation (XFAIL)
is_known_limitation() {
    local config_name="$1"
    local config_args="$2"
    local backend_name="$3"

    # HFA/Pyramid partitioning requires >=12 layers (frequency-based RoPE puts layer 0
    # in the preamble, and NPUW's keep_blocks=10 threshold needs >=10 repeating layers).
    if [[ "$backend_name" == "NPU_hfa" || "$backend_name" == "NPU_pyramid" ]]; then
        local num_layers=12  # default
        if [[ "$config_args" == *"--num-layers"* ]]; then
            num_layers=$(echo "$config_args" | grep -oP '(?<=--num-layers )\d+')
        fi
        if [[ -n "$num_layers" ]] && [[ "$num_layers" -lt 12 ]]; then
            echo "HFA/Pyramid partitioning requires >=12 layers (keep_blocks=10 threshold)"
            return 0
        fi
    fi

    # VLM 3D (qwen2_5_vl) + HFA: GenAI VLMPipeline falls back to Optimum Intel which
    # doesn't handle MAX_PROMPT_LEN config property, causing NPU plugin rejection.
    if [[ "$backend_name" == "NPU_hfa" ]] && [[ "$config_args" == *"--position-ids 3d"* ]]; then
        echo "qwen2_5_vl VLMPipeline + HFA attention hint causes Optimum fallback"
        return 0
    fi

    # Whisper + NPU requires NPU hardware (L0 driver). Without it, whisper_genai segfaults
    # because WhisperPipeline doesn't handle missing NPU gracefully.
    if [[ "$backend_name" == "NPU_npuw" ]] && [[ "$config_args" == *"--type whisper"* ]]; then
        echo "Whisper NPU requires L0 driver (no NPU hardware available)"
        return 0
    fi

    return 1
}

# --- Tracking ---
TOTAL=0
PASSED=0
FAILED=0
XFAILED=0
XPASSED=0
SKIPPED=0
declare -a FAILURES=()
declare -a XFAILS=()
declare -a XPASSES=()
declare -A GENERATED=()
DUMP_OK=0
DUMP_FAIL=0
DUMP_SKIP=0
DUMP_WARN=0
declare -a DUMP_WARNINGS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$BARE_DIR"

# Create embedding test helper (avoids bash/python quoting nightmare)
cat > "$EMB_TEST_SCRIPT" << 'PYEOF'
import sys, numpy as np
from openvino import Core

model_dir, device = sys.argv[1], sys.argv[2]
BATCH, SEQ = 1, 4

core = Core()
config = {}
if device == "NPU":
    config = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "CPU"}

m = core.read_model(f"{model_dir}/openvino_model.xml")

# NPUW requires static shapes — reshape before compile
if device != "CPU":
    shapes = {}
    for inp in m.inputs:
        shapes[inp.any_name] = [BATCH, SEQ]
    m.reshape(shapes)

cm = core.compile_model(m, device, config)
inputs = {"input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
          "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64)}
if any(i.any_name == "token_type_ids" for i in m.inputs):
    inputs["token_type_ids"] = np.array([[0, 0, 0, 0]], dtype=np.int64)
r = cm(inputs)
print(f"Inference OK: output shape {list(r[cm.output(0)].shape)}")
PYEOF

# Create embedding dump helper (loads NPUW config from JSON, compiles on NPU to trigger dump)
cat > "$EMB_DUMP_SCRIPT" << 'PYEOF'
import sys, json, numpy as np
from openvino import Core

model_dir, config_file = sys.argv[1], sys.argv[2]
BATCH, SEQ = 1, 4

with open(config_file) as f:
    config = json.load(f)
config["NPU_USE_NPUW"] = "YES"

core = Core()
m = core.read_model(f"{model_dir}/openvino_model.xml")

# NPUW requires static shapes
shapes = {}
for inp in m.inputs:
    shapes[inp.any_name] = [BATCH, SEQ]
m.reshape(shapes)

cm = core.compile_model(m, "NPU", config)
inputs = {"input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
          "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64)}
if any(i.any_name == "token_type_ids" for i in m.inputs):
    inputs["token_type_ids"] = np.array([[0, 0, 0, 0]], dtype=np.int64)
r = cm(inputs)
print(f"Dump OK: output shape {list(r[cm.output(0)].shape)}")
PYEOF

# Creates a temporary JSON config file for NPUW subgraph dumping.
# Usage: make_dump_config <backend> <dump_dir>
# Writes config path to stdout.
make_dump_config() {
    local backend="$1"
    local dump_dir="$2"
    local tmp_config
    tmp_config=$(mktemp "$OUTPUT_DIR/_dump_config_XXXXXX.json")

    case "$backend" in
        npuw)
            cat > "$tmp_config" << EOF
{
   "NPUW_DEVICES": "CPU",
   "NPUW_DUMP_SUBS": "MIN",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir"
}
EOF
            ;;
        hfa)
            cat > "$tmp_config" << EOF
{
   "NPUW_DEVICES": "CPU",
   "MAX_PROMPT_LEN": 8192,
   "NPUW_DUMP_SUBS": "MIN",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir",
   "NPUW_ATTN": "HFA",
   "NPUW_LLM_PREFILL_ATTENTION_HINT": "HFA"
}
EOF
            ;;
        pyramid)
            cat > "$tmp_config" << EOF
{
   "NPUW_DEVICES": "CPU",
   "MAX_PROMPT_LEN": 8192,
   "NPUW_DUMP_SUBS": "MIN",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir",
   "NPUW_ATTN": "PYRAMID",
   "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID"
}
EOF
            ;;
        whisper)
            cat > "$tmp_config" << EOF
{
   "NPU_USE_NPUW": "YES",
   "NPUW_ONLINE_PIPELINE": "NONE",
   "NPUW_DEVICES": "CPU",
   "NPUW_DUMP_SUBS": "YES",
   "NPUW_DUMP_IO": "YES",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir"
}
EOF
            ;;
        emb_decoder)
            cat > "$tmp_config" << EOF
{
   "NPU_USE_NPUW": "YES",
   "NPUW_ONLINE_PIPELINE": "NONE",
   "NPUW_DEVICES": "CPU",
   "NPUW_DUMP_SUBS": "YES",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir"
}
EOF
            ;;
        emb_encoder)
            cat > "$tmp_config" << EOF
{
   "NPUW_DEVICES": "CPU",
   "NPUW_DUMP_SUBS": "YES",
   "NPUW_DUMP_SUBS_DIR": "$dump_dir"
}
EOF
            ;;
    esac
    echo "$tmp_config"
}

echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD} NPUW Model Builder Test Suite${NC}"
echo -e "${BOLD}========================================${NC}"
echo "Generator:  $GENERATOR"
echo "Tokenizer:  $TOKENIZER_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Bare:       $BARE_DIR"
echo "Configs:    ${#CONFIGS[@]}"
echo "Backends:   ${#BACKENDS[@]}"
if [[ $GENERATE_ONLY -eq 1 ]]; then
    echo "Mode:       generate-only (no testing)"
fi
if [[ $DUMP -eq 1 ]]; then
    echo "Dump:       $DUMP_DIR"
fi
if [[ -n "$FILTER" ]]; then
    echo "Filter:     $FILTER"
fi
echo ""

for config_entry in "${CONFIGS[@]}"; do
    CONFIG_NAME="${config_entry%%|*}"
    CONFIG_ARGS="${config_entry#*|}"

    # Apply filter
    if [[ -n "$FILTER" ]] && ! echo "$CONFIG_NAME" | grep -q "$FILTER"; then
        continue
    fi

    MODEL_DIR="$OUTPUT_DIR/$CONFIG_NAME"

    echo -e "${CYAN}--- [$CONFIG_NAME] Generating model ---${NC}"

    # Determine model type and tokenizer
    IS_WHISPER=0
    IS_EMB_DECODER=0
    IS_EMB_ENCODER=0
    if [[ "$CONFIG_ARGS" == *"--type whisper"* ]]; then
        IS_WHISPER=1
        TOK_DIR="$WHISPER_TOKENIZER_DIR"
    elif [[ "$CONFIG_ARGS" == *"--type embedding"* ]] && [[ "$CONFIG_ARGS" == *"--arch encoder"* ]]; then
        IS_EMB_ENCODER=1
        TOK_DIR="$CONTRIEVER_TOKENIZER_DIR"
    elif [[ "$CONFIG_ARGS" == *"--type embedding"* ]]; then
        IS_EMB_DECODER=1
        TOK_DIR="$QWEN3_EMB_TOKENIZER_DIR"
    elif [[ "$CONFIG_ARGS" == *"--inputs-embeds"* ]]; then
        TOK_DIR="$QWEN_TOKENIZER_DIR"
    else
        TOK_DIR="$TOKENIZER_DIR"
    fi

    # Generate model — whisper/embedding configs already include --type, LLM configs need it prepended
    TOK_FLAG=""
    [[ -n "$TOK_DIR" ]] && TOK_FLAG="-t $TOK_DIR"
    if [[ $IS_WHISPER -eq 1 || $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
        GEN_CMD="$GENERATOR $TOK_FLAG -o $OUTPUT_DIR -n $CONFIG_NAME $CONFIG_ARGS"
    else
        GEN_CMD="$GENERATOR --type llm $TOK_FLAG -o $OUTPUT_DIR -n $CONFIG_NAME $CONFIG_ARGS"
    fi
    GEN_LOG="$OUTPUT_DIR/${CONFIG_NAME}_gen.log"

    if $GEN_CMD > "$GEN_LOG" 2>&1; then
        echo -e "  ${GREEN}Generated OK${NC}"
        GENERATED[$CONFIG_NAME]=1

        # Generate bare model (no tokenizer/config files — just serialized OV IR)
        if [[ $IS_WHISPER -eq 1 || $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
            BARE_CMD="$GENERATOR -o $BARE_DIR -n $CONFIG_NAME $CONFIG_ARGS"
        else
            BARE_CMD="$GENERATOR --type llm -o $BARE_DIR -n $CONFIG_NAME $CONFIG_ARGS"
        fi
        $BARE_CMD > /dev/null 2>&1

        # In generate-only mode, skip testing
        if [[ $GENERATE_ONLY -eq 1 ]]; then
            continue
        fi
    else
        echo -e "  ${RED}GENERATION FAILED${NC}"
        cat "$GEN_LOG"
        # Count expected backends as failures
        if [[ $IS_WHISPER -eq 1 || $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
            if [[ $QUICK -eq 1 ]]; then
                TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1))
                FAILURES+=("${CONFIG_NAME}/CPU (generation failed)")
            else
                TOTAL=$((TOTAL + 2)); FAILED=$((FAILED + 2))
                FAILURES+=("${CONFIG_NAME}/CPU (generation failed)")
                FAILURES+=("${CONFIG_NAME}/NPU_npuw (generation failed)")
            fi
        else
            for backend_entry in "${BACKENDS[@]}"; do
                BACKEND_NAME="${backend_entry%%|*}"
                TOTAL=$((TOTAL + 1))
                FAILED=$((FAILED + 1))
                FAILURES+=("${CONFIG_NAME}/${BACKEND_NAME} (generation failed)")
            done
        fi
        continue
    fi

    # Whisper/Embedding use only CPU and NPU_npuw backends (no HFA/Pyramid)
    if [[ $IS_WHISPER -eq 1 ]]; then
        if [[ $QUICK -eq 1 ]]; then
            ACTIVE_BACKENDS=("CPU|")
        else
            ACTIVE_BACKENDS=("CPU|" "NPU_npuw|-d NPU -lc $CONFIGS_DIR/synth_whisper.json")
        fi
    elif [[ $IS_EMB_DECODER -eq 1 ]]; then
        if [[ $QUICK -eq 1 ]]; then
            ACTIVE_BACKENDS=("CPU|")
        else
            ACTIVE_BACKENDS=("CPU|" "NPU_npuw|-d NPU -lc $CONFIGS_DIR/synth_embed.json")
        fi
    elif [[ $IS_EMB_ENCODER -eq 1 ]]; then
        if [[ $QUICK -eq 1 ]]; then
            ACTIVE_BACKENDS=("CPU|")
        else
            ACTIVE_BACKENDS=("CPU|" "NPU_npuw|-d NPU")
        fi
    else
        ACTIVE_BACKENDS=("${BACKENDS[@]}")
    fi

    # Test on each backend
    for backend_entry in "${ACTIVE_BACKENDS[@]}"; do
        BACKEND_NAME="${backend_entry%%|*}"
        BACKEND_ARGS="${backend_entry#*|}"
        TOTAL=$((TOTAL + 1))

        TEST_NAME="${CONFIG_NAME}/${BACKEND_NAME}"
        TEST_LOG="$OUTPUT_DIR/${CONFIG_NAME}_${BACKEND_NAME}.log"

        # Check for known limitations
        XFAIL_REASON=""
        if XFAIL_REASON=$(is_known_limitation "$CONFIG_NAME" "$CONFIG_ARGS" "$BACKEND_NAME"); then
            IS_XFAIL=1
        else
            IS_XFAIL=0
        fi

        printf "  [%-12s] " "$BACKEND_NAME"

        # Build test command based on model type
        if [[ $IS_WHISPER -eq 1 ]]; then
            # Whisper: use whisper_genai.py
            if [[ "$BACKEND_NAME" == "CPU" ]]; then
                BENCH_CMD="bash -c 'source $VENV && source $SETUPVARS && python $WHISPER_GENAI $MODEL_DIR $TEST_AUDIO CPU'"
            else
                BENCH_CMD="bash -c 'source $VENV && source $SETUPVARS && python $WHISPER_GENAI $MODEL_DIR $TEST_AUDIO NPU -c $CONFIGS_DIR/synth_whisper.json'"
            fi
        elif [[ $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
            # Embedding models: use bare-OV test script (no torch dependency)
            EMB_DEVICE="${BACKEND_NAME%%_*}"  # "NPU_npuw" -> "NPU", "CPU" -> "CPU"
            BENCH_CMD="bash -c 'source $VENV && source $SETUPVARS && python3 $EMB_TEST_SCRIPT $MODEL_DIR $EMB_DEVICE'"
        else
            # LLM/VLM: use llm_bench
            if [[ "$BACKEND_NAME" == "CPU" ]]; then
                DEVICE_ARGS="-d CPU -ic 10"
            else
                DEVICE_ARGS="$BACKEND_ARGS"
            fi
            BENCH_CMD="bash -c 'source $VENV && source $SETUPVARS && python $LLM_BENCH --genai -m $MODEL_DIR $DEVICE_ARGS'"
        fi

        TEST_PASSED=0
        if eval "$BENCH_CMD" > "$TEST_LOG" 2>&1; then
            if [[ $IS_WHISPER -eq 1 ]]; then
                # Whisper: check for transcription output (timestamps lines)
                # Don't use error-absence check — NPUW warnings contain "Exception"/"Error"
                if grep -q "timestamps:" "$TEST_LOG"; then
                    TEST_PASSED=1
                fi
            elif [[ $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
                if grep -q "Inference OK:" "$TEST_LOG"; then
                    TEST_PASSED=1
                fi
            else
                if grep -q "Generation Time:" "$TEST_LOG"; then
                    TEST_PASSED=1
                fi
            fi
        fi

        if [[ $TEST_PASSED -eq 1 ]]; then
            if [[ $IS_WHISPER -eq 1 ]]; then
                INFO="whisper ok"
            elif [[ $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
                INFO=$(grep "Inference OK:" "$TEST_LOG" | head -1 | sed 's/Inference OK: //')
            else
                LATENCY=$(grep "First token latency:" "$TEST_LOG" | head -1 | sed 's/.*First token latency: \([0-9.]*\).*/\1/')
                INFO="first token: ${LATENCY}ms"
            fi
            if [[ $IS_XFAIL -eq 1 ]]; then
                echo -e "${MAGENTA}XPASS${NC} (${INFO}) [expected fail: $XFAIL_REASON]"
                XPASSED=$((XPASSED + 1))
                XPASSES+=("$TEST_NAME ($XFAIL_REASON)")
            else
                echo -e "${GREEN}PASS${NC} (${INFO})"
                PASSED=$((PASSED + 1))
            fi
        else
            if [[ $IS_XFAIL -eq 1 ]]; then
                echo -e "${YELLOW}XFAIL${NC} ($XFAIL_REASON)"
                XFAILED=$((XFAILED + 1))
                XFAILS+=("$TEST_NAME ($XFAIL_REASON)")
            else
                echo -e "${RED}FAIL${NC}"
                FAILED=$((FAILED + 1))
                FAILURES+=("$TEST_NAME")
                grep -E "ERROR|Exception|error|Traceback" "$TEST_LOG" | tail -3 | sed 's/^/    /'
            fi
        fi
    done
    echo ""
done

# Returns "min max" expected sg count for a model/backend dump combo.
# Prints nothing if no expectation is defined (no warning issued).
expected_sg_range() {
    local config_args="$1"
    local dump_backend="$2"

    # Whisper/embedding subgraph counts depend on NPUW_ONLINE_PIPELINE=NONE — no expectation
    if [[ "$config_args" == *"--type whisper"* ]] || [[ "$config_args" == *"--type embedding"* ]]; then
        return
    fi

    # LLM/VLM models: online partitioning produces FCEW + REP + FCEW = 3 subgraphs
    # for all backends (npuw, hfa, pyramid).
    echo "3 3"
}

# --- Generate-only: print summary and exit ---
if [[ $GENERATE_ONLY -eq 1 ]]; then
    GEN_COUNT=${#GENERATED[@]}
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD} Generation Summary${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo -e "Generated: ${GREEN}$GEN_COUNT${NC} models"
    echo -e "Output:    $OUTPUT_DIR/"
    echo -e "Bare:      $BARE_DIR/"
    if [[ $FAILED -gt 0 ]]; then
        echo -e "Failed:    ${RED}$FAILED${NC}"
        for f in "${FAILURES[@]}"; do
            echo -e "  ${RED}x${NC} $f"
        done
        exit 1
    fi
    exit 0
fi

# --- Dump phase ---
if [[ $DUMP -eq 1 ]]; then
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD} NPUW Subgraph Dump Phase${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""

    mkdir -p "$DUMP_DIR"

    for config_entry in "${CONFIGS[@]}"; do
        CONFIG_NAME="${config_entry%%|*}"
        CONFIG_ARGS="${config_entry#*|}"

        # Apply filter
        if [[ -n "$FILTER" ]] && ! echo "$CONFIG_NAME" | grep -q "$FILTER"; then
            continue
        fi

        # Skip configs that failed generation
        if [[ -z "${GENERATED[$CONFIG_NAME]+x}" ]]; then
            continue
        fi

        MODEL_DIR="$OUTPUT_DIR/$CONFIG_NAME"

        # Determine model type
        IS_WHISPER=0
        IS_EMB_DECODER=0
        IS_EMB_ENCODER=0
        IS_VLM=0
        if [[ "$CONFIG_ARGS" == *"--type whisper"* ]]; then
            IS_WHISPER=1
        elif [[ "$CONFIG_ARGS" == *"--type embedding"* ]] && [[ "$CONFIG_ARGS" == *"--arch encoder"* ]]; then
            IS_EMB_ENCODER=1
        elif [[ "$CONFIG_ARGS" == *"--type embedding"* ]]; then
            IS_EMB_DECODER=1
        elif [[ "$CONFIG_ARGS" == *"--inputs-embeds"* ]]; then
            IS_VLM=1
        fi

        # Determine applicable dump backends
        if [[ $IS_WHISPER -eq 1 ]]; then
            DUMP_BACKENDS=("whisper")
        elif [[ $IS_EMB_DECODER -eq 1 ]]; then
            DUMP_BACKENDS=("emb_decoder")
        elif [[ $IS_EMB_ENCODER -eq 1 ]]; then
            DUMP_BACKENDS=("emb_encoder")
        else
            DUMP_BACKENDS=("npuw" "hfa" "pyramid")
        fi

        echo -e "${CYAN}--- [$CONFIG_NAME] Dumping subgraphs ---${NC}"

        for dump_backend in "${DUMP_BACKENDS[@]}"; do
            DUMP_SUBDIR="$DUMP_DIR/$CONFIG_NAME/$dump_backend"

            # Check XFAIL: skip HFA/Pyramid for models with <12 layers
            if [[ "$dump_backend" == "hfa" || "$dump_backend" == "pyramid" ]]; then
                local_layers=12  # default
                if [[ "$CONFIG_ARGS" == *"--num-layers"* ]]; then
                    local_layers=$(echo "$CONFIG_ARGS" | grep -oP '(?<=--num-layers )\d+')
                fi
                if [[ -n "$local_layers" ]] && [[ "$local_layers" -lt 12 ]]; then
                    printf "  [%-9s] ${YELLOW}SKIP${NC} (<12 layers, partitioning threshold)\n" "$dump_backend"
                    DUMP_SKIP=$((DUMP_SKIP + 1))
                    continue
                fi
            fi

            # Skip VLM 3D + HFA dump (same Optimum fallback issue as test phase)
            if [[ "$dump_backend" == "hfa" ]] && [[ "$CONFIG_ARGS" == *"--position-ids 3d"* ]]; then
                printf "  [%-9s] ${YELLOW}SKIP${NC} (qwen2_5_vl + HFA Optimum fallback)\n" "$dump_backend"
                DUMP_SKIP=$((DUMP_SKIP + 1))
                continue
            fi

            mkdir -p "$DUMP_SUBDIR"
            DUMP_CONFIG=$(make_dump_config "$dump_backend" "$DUMP_SUBDIR")
            DUMP_LOG="$OUTPUT_DIR/${CONFIG_NAME}_dump_${dump_backend}.log"

            # Build dump command
            if [[ $IS_WHISPER -eq 1 ]]; then
                DUMP_CMD="bash -c 'source $VENV && source $SETUPVARS && python $WHISPER_GENAI $MODEL_DIR $TEST_AUDIO NPU -c $DUMP_CONFIG'"
            elif [[ $IS_EMB_DECODER -eq 1 || $IS_EMB_ENCODER -eq 1 ]]; then
                DUMP_CMD="bash -c 'source $VENV && source $SETUPVARS && python3 $EMB_DUMP_SCRIPT $MODEL_DIR $DUMP_CONFIG'"
            else
                DUMP_CMD="bash -c 'source $VENV && source $SETUPVARS && python $LLM_BENCH --genai -m $MODEL_DIR -d NPU -lc $DUMP_CONFIG'"
            fi

            printf "  [%-9s] " "$dump_backend"

            if eval "$DUMP_CMD" > "$DUMP_LOG" 2>&1; then
                # Count output files
                SG_COUNT=$(find "$DUMP_SUBDIR" -name "*.sg" 2>/dev/null | wc -l)
                XML_COUNT=$(find "$DUMP_SUBDIR" -name "*.xml" 2>/dev/null | wc -l)

                if [[ $SG_COUNT -gt 0 || $XML_COUNT -gt 0 ]]; then
                    echo -e "${GREEN}OK${NC} ($SG_COUNT sg, $XML_COUNT xml)"
                    DUMP_OK=$((DUMP_OK + 1))

                    # Check for anomalous subgraph count
                    EXPECTED_RANGE=$(expected_sg_range "$CONFIG_ARGS" "$dump_backend")
                    if [[ -n "$EXPECTED_RANGE" ]]; then
                        EXPECTED_MIN=${EXPECTED_RANGE%% *}
                        EXPECTED_MAX=${EXPECTED_RANGE##* }
                        if [[ $SG_COUNT -lt $EXPECTED_MIN || $SG_COUNT -gt $EXPECTED_MAX ]]; then
                            if [[ $EXPECTED_MIN -eq $EXPECTED_MAX ]]; then
                                EXPECT_STR="expected $EXPECTED_MIN"
                            else
                                EXPECT_STR="expected $EXPECTED_MIN-$EXPECTED_MAX"
                            fi
                            echo -e "           ${YELLOW}WARNING: anomalous subgraph count: $SG_COUNT sg ($EXPECT_STR)${NC}"
                            DUMP_WARN=$((DUMP_WARN + 1))
                            DUMP_WARNINGS+=("${CONFIG_NAME}/${dump_backend}: $SG_COUNT sg ($EXPECT_STR)")
                        fi
                    fi
                else
                    echo -e "${RED}FAIL${NC} (no output files)"
                    DUMP_FAIL=$((DUMP_FAIL + 1))
                fi
            else
                echo -e "${RED}FAIL${NC}"
                DUMP_FAIL=$((DUMP_FAIL + 1))
                grep -E "ERROR|Exception|error|Traceback" "$DUMP_LOG" | tail -3 | sed 's/^/    /'
            fi

            rm -f "$DUMP_CONFIG"
        done
        echo ""
    done
fi

# --- Summary ---
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD} Summary${NC}"
echo -e "${BOLD}========================================${NC}"
echo -e "Total:   $TOTAL"
echo -e "Passed:  ${GREEN}$PASSED${NC}"
if [[ $XPASSED -gt 0 ]]; then
    echo -e "XPass:   ${MAGENTA}$XPASSED${NC} (expected fail but passed — update known limitations!)"
fi
if [[ $XFAILED -gt 0 ]]; then
    echo -e "XFail:   ${YELLOW}$XFAILED${NC} (known limitations)"
fi
echo -e "Failed:  ${RED}$FAILED${NC}"

if [[ $XFAILED -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Known limitations (XFAIL):${NC}"
    for f in "${XFAILS[@]}"; do
        echo -e "  ${YELLOW}-${NC} $f"
    done
fi

if [[ $XPASSED -gt 0 ]]; then
    echo ""
    echo -e "${MAGENTA}Unexpected passes (XPASS):${NC}"
    for f in "${XPASSES[@]}"; do
        echo -e "  ${MAGENTA}!${NC} $f"
    done
fi

if [[ $DUMP -eq 1 ]]; then
    echo ""
    echo -e "Dumps:   ${GREEN}OK=$DUMP_OK${NC}  ${RED}FAIL=$DUMP_FAIL${NC}  ${YELLOW}SKIP=$DUMP_SKIP${NC}  ${YELLOW}WARN=$DUMP_WARN${NC}"
    echo -e "Output:  $DUMP_DIR/"
    if [[ $DUMP_WARN -gt 0 ]]; then
        echo ""
        echo -e "${YELLOW}Anomalous subgraph counts:${NC}"
        for w in "${DUMP_WARNINGS[@]}"; do
            echo -e "  ${YELLOW}!${NC} $w"
        done
    fi
fi

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo -e "${RED}Unexpected failures:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "  ${RED}x${NC} $f"
    done
    echo ""
    echo "Logs are in: $OUTPUT_DIR/"
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed!${NC} ($XFAILED known limitations)"
    if [[ $DUMP -eq 1 ]] && [[ $DUMP_FAIL -gt 0 ]]; then
        echo -e "${YELLOW}Warning: some dumps failed (see above)${NC}"
    fi
    if [[ $DUMP -eq 1 ]] && [[ $DUMP_WARN -gt 0 ]]; then
        echo -e "${YELLOW}Warning: anomalous subgraph counts detected (see above)${NC}"
    fi
    exit 0
fi

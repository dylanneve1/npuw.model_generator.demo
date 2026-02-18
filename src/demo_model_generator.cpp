// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Utility to generate synthetic models for NPUW testing.
// Outputs OpenVINO IR models compatible with llm_bench/LLMPipeline and
// WhisperPipeline.
//
// Usage: npuw_model_generator_demo --type <llm|whisper|embedding> [options]

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "model_builder.hpp"
#include "openvino/pass/serialize.hpp"

namespace fs = std::filesystem;

using namespace ov::test::npuw;

static void print_help(const char *prog) {
  std::cout << "Usage: " << prog
            << " --type <llm|whisper|embedding> [options]\n\n";
  std::cout << "Generate synthetic OpenVINO IR models for NPUW testing.\n\n";
  std::cout << "Required:\n";
  std::cout
      << "  --type <type>           Model type: llm, whisper, embedding\n\n";
  std::cout << "Output:\n";
  std::cout << "  -o, --output <dir>      Output directory (default: "
               "npuw_test_models)\n";
  std::cout << "  -n, --name <name>       Model subdirectory name (default: "
               "model)\n\n";
  std::cout << "Tokenizer:\n";
  std::cout << "  -t, --tokenizer <dir>   Copy tokenizer files from an "
               "existing model directory\n\n";
  std::cout << "LLM options:\n";
  std::cout
      << "  --num-layers <N>        Number of decoder layers (default: 12)\n";
  std::cout << "  --hidden-size <N>       Hidden dimension (default: 256)\n";
  std::cout
      << "  --num-heads <N>         Number of attention heads (default: 8)\n";
  std::cout
      << "  --num-kv-heads <N>      Number of KV heads, 0=MHA (default: 0)\n";
  std::cout << "  --head-dim <N>          Head dimension (default: 32)\n";
  std::cout
      << "  --intermediate-size <N> FFN intermediate size (default: 1024)\n";
  std::cout << "  --vocab-size <N>        Vocabulary size (default: 32000)\n";
  std::cout << "  --context-len <N>       Max context length for RoPE "
               "(default: 128)\n";
  std::cout
      << "  --ffn-type <type>       FFN type: swiglu, gelu (default: swiglu)\n";
  std::cout
      << "  --norm-type <type>      Normalization: rms, layer (default: rms)\n";
  std::cout << "  --rope-type <type>      RoPE type: half, interleaved, none "
               "(default: half)\n";
  std::cout << "  --weight-type <type>    Weight format: fp32, fp16, int8, "
               "int4 (default: fp32)\n";
  std::cout << "  --group-size <N>        Group size for int4 group "
               "quantization, 0=per-channel (default: 0)\n";
  std::cout << "  --inputs-embeds         Use inputs_embeds instead of "
               "input_ids (VLM-style)\n";
  std::cout << "  --position-ids <type>   Position IDs shape: 2d, 3d, none "
               "(default: 2d)\n";
  std::cout
      << "  --no-kv-cache           Disable KV cache (stateless model)\n\n";
  std::cout << "Whisper options (defaults match whisper-tiny):\n";
  std::cout << "  --d-model <N>           Model dimension (default: 384)\n";
  std::cout << "  --encoder-layers <N>    Encoder layers (default: 4)\n";
  std::cout << "  --decoder-layers <N>    Decoder layers (default: 4)\n";
  std::cout << "  --num-heads <N>         Attention heads (default: 6)\n";
  std::cout << "  --ffn-dim <N>           FFN intermediate dimension (default: "
               "1536)\n";
  std::cout << "  --vocab-size <N>        Vocabulary size (default: 51865)\n";
  std::cout << "  --weight-type <type>    Weight format: fp32, fp16 (default: "
               "fp32)\n\n";
  std::cout << "Embedding options:\n";
  std::cout << "  --arch <type>           Architecture: decoder (Qwen3-style), "
               "encoder (BERT-style)\n";
  std::cout << "                          (default: decoder)\n";
  std::cout << "  (decoder arch also accepts all LLM options above)\n";
  std::cout << "  (encoder arch also accepts --hidden-size, --num-heads, "
               "--head-dim,\n";
  std::cout << "   --intermediate-size, --vocab-size, --num-layers, "
               "--weight-type)\n\n";
  std::cout << "General:\n";
  std::cout << "  -h, --help              Show this help\n";
}

static void write_config_json(const fs::path &dir, const ModelConfig &config,
                              size_t context_len,
                              const std::string &model_type_id) {
  bool qwen_tokens =
      (model_type_id == "qwen2_5_vl" || model_type_id == "llava");
  {
    std::ofstream ofs(dir / "config.json");
    ofs << "{\n";
    ofs << "  \"model_type\": \"" << model_type_id << "\",\n";
    ofs << "  \"hidden_size\": " << config.hidden_size << ",\n";
    ofs << "  \"num_attention_heads\": " << config.num_heads << ",\n";
    ofs << "  \"num_key_value_heads\": " << config.get_kv_heads() << ",\n";
    ofs << "  \"num_hidden_layers\": " << config.num_layers << ",\n";
    ofs << "  \"intermediate_size\": " << config.intermediate_size << ",\n";
    ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
    ofs << "  \"max_position_embeddings\": " << context_len << ",\n";
    if (qwen_tokens) {
      ofs << "  \"bos_token_id\": 151643,\n";
      ofs << "  \"eos_token_id\": 151645,\n";
      ofs << "  \"pad_token_id\": 151643\n";
    } else {
      ofs << "  \"bos_token_id\": 1,\n";
      ofs << "  \"eos_token_id\": 2,\n";
      ofs << "  \"pad_token_id\": 0\n";
    }
    ofs << "}\n";
  }
  {
    std::ofstream ofs(dir / "generation_config.json");
    ofs << "{\n";
    if (qwen_tokens) {
      ofs << "  \"bos_token_id\": 151643,\n";
      ofs << "  \"eos_token_id\": [151645, 151643],\n";
      ofs << "  \"pad_token_id\": 151643,\n";
    } else {
      ofs << "  \"bos_token_id\": 1,\n";
      ofs << "  \"eos_token_id\": 2,\n";
      ofs << "  \"pad_token_id\": 0,\n";
    }
    ofs << "  \"max_length\": 4096\n";
    ofs << "}\n";
  }
}

static void write_embedding_config_json(const fs::path &dir,
                                        const ModelConfig &config,
                                        size_t context_len) {
  std::ofstream ofs(dir / "config.json");
  ofs << "{\n";
  ofs << "  \"model_type\": \"qwen3\",\n";
  ofs << "  \"hidden_size\": " << config.hidden_size << ",\n";
  ofs << "  \"num_attention_heads\": " << config.num_heads << ",\n";
  ofs << "  \"num_key_value_heads\": " << config.get_kv_heads() << ",\n";
  ofs << "  \"num_hidden_layers\": " << config.num_layers << ",\n";
  ofs << "  \"intermediate_size\": " << config.intermediate_size << ",\n";
  ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
  ofs << "  \"max_position_embeddings\": " << context_len << ",\n";
  ofs << "  \"bos_token_id\": 151643,\n";
  ofs << "  \"eos_token_id\": 151643,\n";
  ofs << "  \"pad_token_id\": 151643\n";
  ofs << "}\n";
}

static void write_encoder_config_json(const fs::path &dir,
                                      const ModelConfig &config) {
  std::ofstream ofs(dir / "config.json");
  ofs << "{\n";
  ofs << "  \"model_type\": \"bert\",\n";
  ofs << "  \"hidden_size\": " << config.hidden_size << ",\n";
  ofs << "  \"num_attention_heads\": " << config.num_heads << ",\n";
  ofs << "  \"num_hidden_layers\": " << config.num_layers << ",\n";
  ofs << "  \"intermediate_size\": " << config.intermediate_size << ",\n";
  ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
  ofs << "  \"max_position_embeddings\": " << config.max_position_embeddings
      << ",\n";
  ofs << "  \"pad_token_id\": 0\n";
  ofs << "}\n";
}

static void copy_tokenizer_files(const fs::path &src_dir,
                                 const fs::path &dst_dir) {
  const std::vector<std::string> files_to_copy = {
      "openvino_tokenizer.xml",
      "openvino_tokenizer.bin",
      "openvino_detokenizer.xml",
      "openvino_detokenizer.bin",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer.model",
      "vocab.json",
      "vocab.txt",
      "merges.txt",
      "added_tokens.json",
  };

  size_t copied = 0;
  for (const auto &fname : files_to_copy) {
    auto src = src_dir / fname;
    if (fs::exists(src)) {
      fs::copy_file(src, dst_dir / fname, fs::copy_options::overwrite_existing);
      ++copied;
    }
  }
  std::cout << "Copied " << copied << " tokenizer file(s) from " << src_dir
            << "\n";
}

static void symlink_vlm_assets(const fs::path &src_dir,
                               const fs::path &dst_dir) {
  const std::vector<std::string> vlm_files = {
      "openvino_vision_embeddings_model.xml",
      "openvino_vision_embeddings_model.bin",
      "openvino_vision_embeddings_merger_model.xml",
      "openvino_vision_embeddings_merger_model.bin",
      "preprocessor_config.json",
      "video_preprocessor_config.json",
  };

  size_t linked = 0;
  for (const auto &fname : vlm_files) {
    auto src = fs::absolute(src_dir / fname);
    auto dst = dst_dir / fname;
    if (fs::exists(src)) {
      if (fs::exists(dst))
        fs::remove(dst);
      fs::create_symlink(src, dst);
      ++linked;
    }
  }
  std::cout << "Symlinked " << linked << " VLM asset(s) from " << src_dir
            << "\n";
}

static void write_whisper_configs(const fs::path &dir,
                                  const ModelConfig &config) {
  {
    std::ofstream ofs(dir / "config.json");
    ofs << "{\n";
    ofs << "  \"model_type\": \"whisper\",\n";
    ofs << "  \"architectures\": [\"WhisperForConditionalGeneration\"],\n";
    ofs << "  \"is_encoder_decoder\": true,\n";
    ofs << "  \"d_model\": " << config.hidden_size << ",\n";
    ofs << "  \"encoder_layers\": " << config.get_encoder_layers() << ",\n";
    ofs << "  \"encoder_attention_heads\": " << config.num_heads << ",\n";
    ofs << "  \"encoder_ffn_dim\": " << config.intermediate_size << ",\n";
    ofs << "  \"decoder_layers\": " << config.get_decoder_layers() << ",\n";
    ofs << "  \"decoder_attention_heads\": " << config.num_heads << ",\n";
    ofs << "  \"decoder_ffn_dim\": " << config.intermediate_size << ",\n";
    ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
    ofs << "  \"num_mel_bins\": " << config.num_mel_bins << ",\n";
    ofs << "  \"max_source_positions\": " << config.max_source_positions
        << ",\n";
    ofs << "  \"max_target_positions\": " << config.max_target_positions
        << ",\n";
    ofs << "  \"activation_function\": \"gelu\",\n";
    ofs << "  \"bos_token_id\": 50257,\n";
    ofs << "  \"eos_token_id\": 50257,\n";
    ofs << "  \"decoder_start_token_id\": 50258,\n";
    ofs << "  \"pad_token_id\": 50257,\n";
    ofs << "  \"use_cache\": true\n";
    ofs << "}\n";
  }
  {
    std::ofstream ofs(dir / "generation_config.json");
    ofs << "{\n";
    ofs << "  \"bos_token_id\": 50257,\n";
    ofs << "  \"eos_token_id\": 50257,\n";
    ofs << "  \"pad_token_id\": 50257,\n";
    ofs << "  \"decoder_start_token_id\": 50258,\n";
    ofs << "  \"no_timestamps_token_id\": 50363,\n";
    ofs << "  \"begin_suppress_tokens\": [220, 50257],\n";
    ofs << "  \"is_multilingual\": true,\n";
    ofs << "  \"max_length\": " << config.max_target_positions << ",\n";
    ofs << "  \"return_timestamps\": false,\n";
    ofs << "  \"lang_to_id\": {\"<|en|>\": 50259},\n";
    ofs << "  \"task_to_id\": {\"transcribe\": 50359, \"translate\": 50358}\n";
    ofs << "}\n";
  }
  {
    std::ofstream ofs(dir / "preprocessor_config.json");
    ofs << "{\n";
    ofs << "  \"feature_extractor_type\": \"WhisperFeatureExtractor\",\n";
    ofs << "  \"feature_size\": " << config.num_mel_bins << ",\n";
    ofs << "  \"sampling_rate\": 16000,\n";
    ofs << "  \"hop_length\": 160,\n";
    ofs << "  \"n_fft\": 400,\n";
    ofs << "  \"chunk_length\": 30,\n";
    ofs << "  \"n_samples\": 480000,\n";
    ofs << "  \"nb_max_frames\": 3000,\n";
    ofs << "  \"padding_side\": \"right\",\n";
    ofs << "  \"padding_value\": 0.0,\n";
    ofs << "  \"return_attention_mask\": false\n";
    ofs << "}\n";
  }
}

static bool parse_size_t(const char *str, size_t &out) {
  try {
    out = std::stoull(str);
    return true;
  } catch (...) {
    return false;
  }
}

int main(int argc, char *argv[]) {
  fs::path output_dir = "npuw_test_models";
  std::string model_name = "model";
  std::string model_type;
  std::string tokenizer_dir;

  ModelConfig config;
  config.num_layers = 12;
  config.hidden_size = 256;
  config.num_heads = 8;
  config.head_dim = 32;
  config.num_kv_heads = 0;
  config.intermediate_size = 1024;
  config.vocab_size = 32000;
  config.use_kv_cache = true;
  config.precision = ov::element::f32;

  ModelConfig whisper_cfg;
  whisper_cfg.hidden_size = 384;
  whisper_cfg.num_heads = 6;
  whisper_cfg.head_dim = 64;
  whisper_cfg.encoder_layers = 4;
  whisper_cfg.decoder_layers = 4;
  whisper_cfg.intermediate_size = 1536;
  whisper_cfg.vocab_size = 51865;
  whisper_cfg.num_mel_bins = 80;
  whisper_cfg.max_source_positions = 1500;
  whisper_cfg.max_target_positions = 448;

  size_t context_len = 128;
  std::string ffn_type_str = "swiglu";
  std::string norm_type_str = "rms";
  std::string rope_type_str = "half";
  std::string weight_type_str = "fp32";
  std::string position_ids_str = "2d";
  std::string arch_str = "decoder";
  size_t group_size = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--type" && i + 1 < argc) {
      model_type = argv[++i];
    } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_dir = argv[++i];
    } else if ((arg == "--name" || arg == "-n") && i + 1 < argc) {
      model_name = argv[++i];
    } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
      tokenizer_dir = argv[++i];
    } else if (arg == "--num-layers" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.num_layers)) {
        std::cerr << "Error: invalid value for --num-layers\n";
        return 1;
      }
    } else if (arg == "--hidden-size" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.hidden_size)) {
        std::cerr << "Error: invalid value for --hidden-size\n";
        return 1;
      }
    } else if (arg == "--num-heads" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.num_heads)) {
        std::cerr << "Error: invalid value for --num-heads\n";
        return 1;
      }
    } else if (arg == "--num-kv-heads" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.num_kv_heads)) {
        std::cerr << "Error: invalid value for --num-kv-heads\n";
        return 1;
      }
    } else if (arg == "--head-dim" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.head_dim)) {
        std::cerr << "Error: invalid value for --head-dim\n";
        return 1;
      }
    } else if (arg == "--intermediate-size" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.intermediate_size)) {
        std::cerr << "Error: invalid value for --intermediate-size\n";
        return 1;
      }
    } else if (arg == "--vocab-size" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], config.vocab_size)) {
        std::cerr << "Error: invalid value for --vocab-size\n";
        return 1;
      }
    } else if (arg == "--context-len" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], context_len)) {
        std::cerr << "Error: invalid value for --context-len\n";
        return 1;
      }
    } else if (arg == "--ffn-type" && i + 1 < argc) {
      ffn_type_str = argv[++i];
      if (ffn_type_str != "swiglu" && ffn_type_str != "gelu") {
        std::cerr << "Error: unknown --ffn-type '" << ffn_type_str
                  << "' (expected: swiglu, gelu)\n";
        return 1;
      }
    } else if (arg == "--norm-type" && i + 1 < argc) {
      norm_type_str = argv[++i];
      if (norm_type_str != "rms" && norm_type_str != "layer") {
        std::cerr << "Error: unknown --norm-type '" << norm_type_str
                  << "' (expected: rms, layer)\n";
        return 1;
      }
    } else if (arg == "--rope-type" && i + 1 < argc) {
      rope_type_str = argv[++i];
      if (rope_type_str != "half" && rope_type_str != "interleaved" &&
          rope_type_str != "none") {
        std::cerr << "Error: unknown --rope-type '" << rope_type_str
                  << "' (expected: half, interleaved, none)\n";
        return 1;
      }
    } else if (arg == "--weight-type" && i + 1 < argc) {
      weight_type_str = argv[++i];
      if (weight_type_str != "fp32" && weight_type_str != "fp16" &&
          weight_type_str != "int8" && weight_type_str != "int4") {
        std::cerr << "Error: unknown --weight-type '" << weight_type_str
                  << "' (expected: fp32, fp16, int8, int4)\n";
        return 1;
      }
    } else if (arg == "--group-size" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], group_size)) {
        std::cerr << "Error: invalid value for --group-size\n";
        return 1;
      }
    } else if (arg == "--position-ids" && i + 1 < argc) {
      position_ids_str = argv[++i];
      if (position_ids_str != "2d" && position_ids_str != "3d" &&
          position_ids_str != "none") {
        std::cerr << "Error: unknown --position-ids '" << position_ids_str
                  << "' (expected: 2d, 3d, none)\n";
        return 1;
      }
    } else if (arg == "--arch" && i + 1 < argc) {
      arch_str = argv[++i];
      if (arch_str != "decoder" && arch_str != "encoder") {
        std::cerr << "Error: unknown --arch '" << arch_str
                  << "' (expected: decoder, encoder)\n";
        return 1;
      }
    } else if (arg == "--inputs-embeds") {
      config.use_inputs_embeds = true;
    } else if (arg == "--no-kv-cache") {
      config.use_kv_cache = false;
    } else if (arg == "--d-model" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], whisper_cfg.hidden_size)) {
        std::cerr << "Error: invalid value for --d-model\n";
        return 1;
      }
    } else if (arg == "--encoder-layers" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], whisper_cfg.encoder_layers)) {
        std::cerr << "Error: invalid value for --encoder-layers\n";
        return 1;
      }
    } else if (arg == "--decoder-layers" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], whisper_cfg.decoder_layers)) {
        std::cerr << "Error: invalid value for --decoder-layers\n";
        return 1;
      }
    } else if (arg == "--ffn-dim" && i + 1 < argc) {
      if (!parse_size_t(argv[++i], whisper_cfg.intermediate_size)) {
        std::cerr << "Error: invalid value for --ffn-dim\n";
        return 1;
      }
    } else {
      std::cerr << "Error: unknown option '" << arg << "'\n";
      std::cerr << "Run with --help for usage information.\n";
      return 1;
    }
  }

  if (model_type.empty()) {
    std::cerr << "Error: --type is required\n";
    std::cerr << "Run with --help for usage information.\n";
    return 1;
  }
  if (model_type != "llm" && model_type != "whisper" &&
      model_type != "embedding") {
    std::cerr << "Error: unsupported model type '" << model_type << "'\n";
    return 1;
  }

  if (!tokenizer_dir.empty() && !fs::is_directory(tokenizer_dir)) {
    std::cerr << "Error: tokenizer directory does not exist: " << tokenizer_dir
              << "\n";
    return 1;
  }

  auto dest = output_dir / model_name;
  fs::create_directories(dest);

  if (model_type == "whisper") {
    if (config.num_heads != 8)
      whisper_cfg.num_heads = config.num_heads;
    if (config.vocab_size != 32000)
      whisper_cfg.vocab_size = config.vocab_size;

    WeightFn wf;
    if (weight_type_str == "fp16")
      wf = FP16Weight{};
    else
      wf = FP32Weight{};
    whisper_cfg.weight = wf;

    whisper_cfg.head_dim = whisper_cfg.hidden_size / whisper_cfg.num_heads;

    std::cout << "Generating Whisper model:\n";
    std::cout << "  d_model:           " << whisper_cfg.hidden_size << "\n";
    std::cout << "  encoder_layers:    " << whisper_cfg.get_encoder_layers()
              << "\n";
    std::cout << "  decoder_layers:    " << whisper_cfg.get_decoder_layers()
              << "\n";
    std::cout << "  attention_heads:   " << whisper_cfg.num_heads << "\n";
    std::cout << "  ffn_dim:           " << whisper_cfg.intermediate_size
              << "\n";
    std::cout << "  vocab_size:        " << whisper_cfg.vocab_size << "\n";
    std::cout << "  weight:            " << weight_type_str << "\n";
    if (!tokenizer_dir.empty())
      std::cout << "  tokenizer_src:     " << tokenizer_dir << "\n";
    std::cout << "\n";

    ModelBuilder mb;
    FloatWeight bias_wf(whisper_cfg.precision);

    ModelConfig enc_cfg = whisper_cfg;
    enc_cfg.num_layers = whisper_cfg.get_encoder_layers();
    enc_cfg.head_dim = whisper_cfg.hidden_size / whisper_cfg.num_heads;
    enc_cfg.use_kv_cache = false;
    enc_cfg.lm_head_weight = {};
    enc_cfg.use_conv_features = true;
    enc_cfg.attn_bias = bias_wf;
    enc_cfg.norm = LayerNorm(whisper_cfg.hidden_size, whisper_cfg.precision);
    enc_cfg.ffn = GELU(whisper_cfg.hidden_size, whisper_cfg.intermediate_size,
                       whisper_cfg.precision, whisper_cfg.weight, bias_wf);

    auto encoder = mb.build_model(enc_cfg);
    ov::pass::Serialize enc_serializer(
        (dest / "openvino_encoder_model.xml").string(),
        (dest / "openvino_encoder_model.bin").string());
    enc_serializer.run_on_model(encoder);
    std::cout << "Encoder model saved.\n";

    ModelConfig dec_cfg = whisper_cfg;
    dec_cfg.num_layers = whisper_cfg.get_decoder_layers();
    dec_cfg.head_dim = whisper_cfg.hidden_size / whisper_cfg.num_heads;
    dec_cfg.use_kv_cache = true;
    dec_cfg.use_cross_attention = true;
    dec_cfg.attn_bias = bias_wf;
    dec_cfg.norm = LayerNorm(whisper_cfg.hidden_size, whisper_cfg.precision);
    dec_cfg.ffn = GELU(whisper_cfg.hidden_size, whisper_cfg.intermediate_size,
                       whisper_cfg.precision, whisper_cfg.weight, bias_wf);

    auto decoder = mb.build_model(dec_cfg);
    ov::pass::Serialize dec_serializer(
        (dest / "openvino_decoder_model.xml").string(),
        (dest / "openvino_decoder_model.bin").string());
    dec_serializer.run_on_model(decoder);
    std::cout << "Decoder model saved.\n";

    write_whisper_configs(dest, whisper_cfg);

    if (!tokenizer_dir.empty()) {
      copy_tokenizer_files(tokenizer_dir, dest);
      auto norm_src = fs::path(tokenizer_dir) / "normalizer.json";
      if (fs::exists(norm_src)) {
        fs::copy_file(norm_src, dest / "normalizer.json",
                      fs::copy_options::overwrite_existing);
      }
    }

    std::cout << "Model saved to: " << dest.string() << "/\n";
    std::cout << "Done.\n";
    return 0;
  }

  if (model_type == "embedding") {
    if (arch_str == "encoder") {
      ModelConfig enc_config;
      enc_config.hidden_size = config.hidden_size;
      enc_config.num_heads = config.num_heads;
      enc_config.head_dim = config.head_dim;
      enc_config.intermediate_size = config.intermediate_size;
      enc_config.vocab_size = config.vocab_size;
      enc_config.num_layers = config.num_layers;
      enc_config.precision = config.precision;
      enc_config.use_kv_cache = false;
      enc_config.lm_head_weight = {};

      WeightFn wf;
      if (weight_type_str == "fp32")
        wf = FP32Weight{};
      else if (weight_type_str == "fp16")
        wf = FP16Weight{};
      else if (weight_type_str == "int8")
        wf = INT8Weight{};
      else if (weight_type_str == "int4")
        wf = INT4Weight{};
      enc_config.weight = wf;
      enc_config.use_token_type_embedding = true;
      enc_config.pre_norm = false;
      FloatWeight bias_wf(config.precision);
      enc_config.attn_bias = bias_wf;
      enc_config.norm = LayerNorm(enc_config.hidden_size, config.precision);
      enc_config.ffn =
          GELU(enc_config.hidden_size, enc_config.intermediate_size,
               config.precision, wf, bias_wf);

      std::cout << "Generating BERT encoder embedding model:\n";
      std::cout << "  layers:            " << enc_config.num_layers << "\n";
      std::cout << "  hidden_size:       " << enc_config.hidden_size << "\n";
      std::cout << "  num_heads:         " << enc_config.num_heads << "\n";
      std::cout << "  head_dim:          " << enc_config.head_dim << "\n";
      std::cout << "  intermediate_size: " << enc_config.intermediate_size
                << "\n";
      std::cout << "  vocab_size:        " << enc_config.vocab_size << "\n";
      std::cout << "  weight:            " << weight_type_str << "\n";
      if (!tokenizer_dir.empty())
        std::cout << "  tokenizer_src:     " << tokenizer_dir << "\n";
      std::cout << "\n";

      ModelBuilder mb;
      auto model = mb.build_model(enc_config);

      ov::pass::Serialize serializer((dest / "openvino_model.xml").string(),
                                     (dest / "openvino_model.bin").string());
      serializer.run_on_model(model);

      write_encoder_config_json(dest, enc_config);
    } else {
      config.use_kv_cache = false;
      config.lm_head_weight = {};
      config.internal_position_ids = true;
      config.qk_norm = RMSNorm(config.head_dim, config.precision);

      WeightFn weight_fn;
      if (weight_type_str == "fp32")
        weight_fn = FP32Weight{};
      else if (weight_type_str == "fp16")
        weight_fn = FP16Weight{};
      else if (weight_type_str == "int8")
        weight_fn = INT8Weight{};
      else if (weight_type_str == "int4") {
        if (group_size > 0)
          weight_fn = INT4GroupWeight{group_size};
        else
          weight_fn = INT4Weight{};
      }
      config.weight = weight_fn;

      if (norm_type_str == "rms")
        config.norm = RMSNorm(config.hidden_size, config.precision);
      else
        config.norm = LayerNorm(config.hidden_size, config.precision);

      if (ffn_type_str == "swiglu")
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size,
                            config.precision, weight_fn);
      else
        config.ffn = GELU(config.hidden_size, config.intermediate_size,
                          config.precision, weight_fn);

      std::cout << "Generating decoder-only embedding model:\n";
      std::cout << "  layers:            " << config.num_layers << "\n";
      std::cout << "  hidden_size:       " << config.hidden_size << "\n";
      std::cout << "  num_heads:         " << config.num_heads << "\n";
      std::cout << "  num_kv_heads:      " << config.get_kv_heads()
                << (config.num_kv_heads == 0 ? " (MHA)" : " (GQA)") << "\n";
      std::cout << "  head_dim:          " << config.head_dim << "\n";
      std::cout << "  intermediate_size: " << config.intermediate_size << "\n";
      std::cout << "  vocab_size:        " << config.vocab_size << "\n";
      std::cout << "  ffn:               " << ffn_type_str << "\n";
      std::cout << "  norm:              " << norm_type_str << "\n";
      std::cout << "  weight:            " << weight_type_str << "\n";
      std::cout << "  qk_norm:           yes\n";
      std::cout << "  internal_pos_ids:  yes\n";
      if (!tokenizer_dir.empty())
        std::cout << "  tokenizer_src:     " << tokenizer_dir << "\n";
      std::cout << "\n";

      ModelBuilder mb;
      auto model = mb.build_model(config);

      ov::pass::Serialize serializer((dest / "openvino_model.xml").string(),
                                     (dest / "openvino_model.bin").string());
      serializer.run_on_model(model);

      write_embedding_config_json(dest, config, context_len);
    }

    if (!tokenizer_dir.empty()) {
      copy_tokenizer_files(tokenizer_dir, dest);
    }

    std::cout << "Model saved to: " << dest.string() << "/\n";
    std::cout << "Done.\n";
    return 0;
  }

  // LLM generation
  if (group_size > 0 && weight_type_str != "int4") {
    std::cerr
        << "Error: --group-size is only supported with --weight-type int4\n";
    return 1;
  }

  WeightFn weight_fn;
  if (weight_type_str == "fp32")
    weight_fn = FP32Weight{};
  else if (weight_type_str == "fp16")
    weight_fn = FP16Weight{};
  else if (weight_type_str == "int8")
    weight_fn = INT8Weight{};
  else if (weight_type_str == "int4") {
    if (group_size > 0)
      weight_fn = INT4GroupWeight{group_size};
    else
      weight_fn = INT4Weight{};
  }
  config.weight = weight_fn;

  if (norm_type_str == "rms")
    config.norm = RMSNorm(config.hidden_size, config.precision);
  else
    config.norm = LayerNorm(config.hidden_size, config.precision);

  // position_ids must be created before RoPE (baked into constructor)
  if (rope_type_str != "none" && position_ids_str != "none") {
    if (position_ids_str == "3d")
      config.position_ids = make_position_ids_3d();

    if (rope_type_str == "interleaved") {
      if (!config.position_ids.get_node())
        config.position_ids = make_position_ids_2d();
      config.rope = InterleavedRoPE(config.head_dim, config.precision,
                                    config.position_ids);
    }
  } else {
    // Identity pass-through prevents build_model from auto-creating RoPE
    config.rope = [](const ov::Output<ov::Node> &input, const std::string &) {
      return input;
    };
  }

  if (ffn_type_str == "swiglu")
    config.ffn = SwiGLU(config.hidden_size, config.intermediate_size,
                        config.precision, weight_fn);
  else
    config.ffn = GELU(config.hidden_size, config.intermediate_size,
                      config.precision, weight_fn);

  std::cout << "Generating " << (config.use_inputs_embeds ? "VLM" : "LLM")
            << " model:\n";
  std::cout << "  layers:            " << config.num_layers << "\n";
  std::cout << "  hidden_size:       " << config.hidden_size << "\n";
  std::cout << "  num_heads:         " << config.num_heads << "\n";
  std::cout << "  num_kv_heads:      " << config.get_kv_heads()
            << (config.num_kv_heads == 0 ? " (MHA)" : " (GQA)") << "\n";
  std::cout << "  head_dim:          " << config.head_dim << "\n";
  std::cout << "  intermediate_size: " << config.intermediate_size << "\n";
  std::cout << "  vocab_size:        " << config.vocab_size << "\n";
  std::cout << "  context_len:       " << context_len << "\n";
  std::cout << "  ffn:               " << ffn_type_str << "\n";
  std::cout << "  norm:              " << norm_type_str << "\n";
  std::cout << "  rope:              " << rope_type_str << "\n";
  std::cout << "  weight:            " << weight_type_str;
  if (group_size > 0)
    std::cout << " (group_size=" << group_size << ")";
  std::cout << "\n";
  std::cout << "  position_ids:      " << position_ids_str << "\n";
  std::cout << "  inputs_embeds:     "
            << (config.use_inputs_embeds ? "yes" : "no") << "\n";
  std::cout << "  kv_cache:          " << (config.use_kv_cache ? "yes" : "no")
            << "\n";
  if (!tokenizer_dir.empty())
    std::cout << "  tokenizer_src:     " << tokenizer_dir << "\n";
  std::cout << "\n";

  ModelBuilder mb;
  auto model = mb.build_model(config);

  if (config.use_inputs_embeds) {
    ov::pass::Serialize lang_serializer(
        (dest / "openvino_language_model.xml").string(),
        (dest / "openvino_language_model.bin").string());
    lang_serializer.run_on_model(model);

    auto emb_input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    emb_input->set_friendly_name("input");
    emb_input->output(0).set_names({"input"});

    auto emb_output = make_embedding(emb_input->output(0), config.vocab_size,
                                     config.hidden_size, "model.embed_tokens",
                                     config.precision);
    emb_output.set_names({"inputs_embeds"});

    auto emb_result = std::make_shared<ov::op::v0::Result>(emb_output);

    auto emb_model = std::make_shared<ov::Model>(ov::ResultVector{emb_result},
                                                 ov::ParameterVector{emb_input},
                                                 "text_embeddings_model");

    ov::pass::Serialize emb_serializer(
        (dest / "openvino_text_embeddings_model.xml").string(),
        (dest / "openvino_text_embeddings_model.bin").string());
    emb_serializer.run_on_model(emb_model);

    if (!tokenizer_dir.empty()) {
      symlink_vlm_assets(tokenizer_dir, dest);
    }
  } else {
    ov::pass::Serialize serializer((dest / "openvino_model.xml").string(),
                                   (dest / "openvino_model.bin").string());
    serializer.run_on_model(model);
  }

  std::string model_type_id = "llama";
  if (config.use_inputs_embeds) {
    model_type_id = (position_ids_str == "3d") ? "qwen2_5_vl" : "llava";
  }
  write_config_json(dest, config, context_len, model_type_id);

  if (!tokenizer_dir.empty()) {
    copy_tokenizer_files(tokenizer_dir, dest);
  }

  std::cout << "Model saved to: " << dest.string() << "/\n";
  std::cout << "Done.\n";
  return 0;
}

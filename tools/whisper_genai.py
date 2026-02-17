#!/usr/bin/env python3
"""Minimal whisper inference test using OpenVINO GenAI WhisperPipeline."""

import argparse
import json

import librosa
import openvino_genai


def main():
    parser = argparse.ArgumentParser(description="Run whisper inference on an audio file")
    parser.add_argument("model_dir", help="Path to OpenVINO whisper model directory")
    parser.add_argument("wav_file", help="Path to WAV audio file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    parser.add_argument("-c", "--config", help="JSON config file for device properties")
    args = parser.parse_args()

    ov_config = {}
    if args.config:
        with open(args.config) as f:
            ov_config = json.load(f)

    pipe = openvino_genai.WhisperPipeline(args.model_dir, args.device, **ov_config)

    gen_config = pipe.get_generation_config()
    gen_config.language = "<|en|>"
    gen_config.task = "transcribe"
    gen_config.return_timestamps = True

    raw_speech, _ = librosa.load(args.wav_file, sr=16000)
    result = pipe.generate(raw_speech.tolist(), gen_config)

    print(result)
    if result.chunks:
        for chunk in result.chunks:
            print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")


if __name__ == "__main__":
    main()

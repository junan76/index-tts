"""Run a short IndexTTS-2.5 synthesis and report resource use on AMD ROCm.

Run from the repository root, for example:
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python tools/rocm_synthesize.py
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from indextts.infer_v2_5 import IndexTTS2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--prompt", default="tests/sample_prompt.wav")
    parser.add_argument("--text", default="你好，这是在AMD显卡上进行的语音合成测试。")
    parser.add_argument("--output", default="outputs/rocm_smoke.wav")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of BF16")
    args = parser.parse_args()
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("An accessible AMD GPU and ROCm PyTorch are required.")
    if not Path(args.prompt).is_file():
        parser.error(f"Reference audio is missing: {args.prompt}")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=str(Path(args.model_dir) / "config.yaml"),
        model_dir=args.model_dir,
        use_bf16=not args.fp32,
        use_cuda_kernel=False,
        use_qwen_emo=False,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tts.infer(
        spk_audio_prompt=args.prompt,
        text=args.text,
        output_path=args.output,
        lang="zh",
        max_text_tokens_per_segment=60,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    audio, sample_rate = sf.read(args.output)
    if audio.size == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise RuntimeError("Output audio is empty, non-finite, or silent.")
    duration = len(audio) / sample_rate
    report = {
        "output": args.output,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "reported_architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "precision": "float32" if args.fp32 else "bfloat16",
        "load_seconds": load_seconds,
        "inference_seconds": elapsed,
        "audio_seconds": duration,
        "sample_rate": sample_rate,
        "rtf": elapsed / duration,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

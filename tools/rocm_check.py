"""Exercise ROCm kernels without downloading or loading IndexTTS weights."""

import argparse
import os

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip}", flush=True)
    print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '(unset)')}", flush=True)
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("A ROCm PyTorch build and an accessible AMD GPU are required.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; architecture: {getattr(props, 'gcnArchName', 'unknown')}", flush=True)
    print(f"Reported GPU memory: {props.total_memory / 1024**3:.2f} GiB", flush=True)
    print(f"Compiled architectures: {torch.cuda.get_arch_list()}", flush=True)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    a = torch.randn(128, 128).to(dtype)
    x = torch.randn(1, 4, 128).to(dtype)
    weight = torch.randn(8, 4, 3).to(dtype)
    q = torch.randn(1, 2, 32, 64).to(dtype)
    tolerance = {"float32": 1e-3, "float16": 2e-2, "bfloat16": 1e-1}[args.dtype]
    with torch.inference_mode():
        for name, operation, inputs in (
            ("matrix multiplication", lambda t: t @ t.T, (a,)),
            ("conv1d", F.conv1d, (x, weight)),
            ("attention", F.scaled_dot_product_attention, (q, q, q)),
        ):
            print(f"Testing {name} ({args.dtype})...", flush=True)
            expected = operation(*(t.float() for t in inputs))
            actual = operation(*(t.to("cuda") for t in inputs)).float().cpu()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        # Audio preprocessing uses float32 FFT, independent of model precision.
        print("Testing FFT (float32)...", flush=True)
        audio = torch.randn(2048)
        actual = torch.fft.rfft(audio.cuda()).cpu()
        torch.testing.assert_close(actual, torch.fft.rfft(audio), rtol=1e-3, atol=1e-3)
    print("PASS: tested kernels agree with CPU reference results.", flush=True)


if __name__ == "__main__":
    main()

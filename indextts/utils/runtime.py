"""Backend-specific restrictions for the portable PyTorch inference path."""

import warnings


def rocm_inference_options(use_cuda_kernel, use_deepspeed, use_accel):
    """Keep ROCm on the ordinary PyTorch path, before loading model weights."""
    import torch

    if not getattr(torch.version, "hip", None):
        return use_cuda_kernel, use_deepspeed, use_accel

    unsupported = []
    if use_deepspeed:
        unsupported.append("use_deepspeed")
    if use_accel:
        unsupported.append("use_accel")
    if unsupported:
        raise ValueError(
            "ROCm inference currently requires "
            + ", ".join(f"{option}=False" for option in unsupported)
            + ". Use the standard PyTorch inference path."
        )
    if use_cuda_kernel:
        warnings.warn(
            "The BigVGAN custom CUDA kernel is disabled on ROCm; "
            "using the PyTorch implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
    return False, False, False

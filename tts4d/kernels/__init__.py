"""Fast operators of the TTS layer: Triton kernels and hand-written (closed-form) backward passes.

The reference implementation of every operator is the plain PyTorch code in ``tts4d/tts_layer.py`` and
``tts4d/blocks.py``; ``TTSLayer(fast=True)`` / ``SelfAttention(fast=True)`` route through the functions below.

    memory.feature_map    phi = SiLU(x W0) * (x W2) / ||.||: ONE bf16 GEMM against [W0 | W2] with fp32 output and a
                          fused Triton tail (forward and backward)
    memory.moments        the five sufficient statistics (G, C, S, T, A) from bf16 GEMM operands with fp32
                          accumulation, hand-written backward
    memory.read_mm        the read  pc @ W  (bf16 operands, fp32 output and gradients)
    qkv.qkv_prep          SiLU + per-head L2 norm + 2-D / time RoPE of q and k, SiLU of v: one Triton kernel each way
    solve.centred_solve   centring + scaling of (G, C) fused with the ridge solve; returns (L^-1, W), closed-form
                          backward (Triton kernels for the dF x dF elementwise parts)
    qk_norm.qk_norm_packed  per-head RMSNorm of q and k of the within-view attention on the packed qkv tensor

Every module imports without Triton; the operators then fall back to PyTorch implementations of the same math.
"""
import torch

try:  # noqa: SIM105
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


def available(device=None):
    """True if the Triton kernels can run (Triton importable and a CUDA device)."""
    if not HAS_TRITON or not torch.cuda.is_available():
        return False
    return device is None or torch.device(device).type == "cuda"


def autocast_bf16(device_type="cuda"):
    """True inside ``torch.autocast(device_type, dtype=torch.bfloat16)``."""
    try:
        return bool(torch.is_autocast_enabled(device_type)) and torch.get_autocast_dtype(device_type) == torch.bfloat16
    except TypeError:  # older PyTorch: no device argument
        return bool(torch.is_autocast_enabled()) and torch.get_autocast_gpu_dtype() == torch.bfloat16

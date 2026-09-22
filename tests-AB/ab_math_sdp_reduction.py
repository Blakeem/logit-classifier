"""Measure what the sdp math reduction setting does to a bfloat16 attention output.

ComfyUI turns `allow_fp16_bf16_reduction_math_sdp` on at import, at
`comfy/model_management.py:569`. It governs the math attention backend, which this package
reaches only when `_usable_attention_backends` finds no CUDA kernel and `_pinned_forward`
enters no `sdpa_kernel` block. Forcing the math backend measures it without that hardware
and without the GPU, so this script needs neither weights nor a card.

    uv run python tests-AB/ab_math_sdp_reduction.py
"""

from __future__ import annotations

import argparse

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SHAPES = [
    ("one head group, short", (1, 4, 256, 128)),
    ("one head group, long", (1, 4, 1024, 128)),
    ("batched rows", (8, 4, 256, 128)),
]


def attention_under(allowed: bool, query, key, value):
    """One math-backend attention with the reduction setting held at `allowed`."""
    saved = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(allowed)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value)
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(saved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"torch {torch.__version__}, cpu, bfloat16")
    print(f"default on this build: {torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}")
    print()

    moved = 0
    for name, shape in SHAPES:
        torch.manual_seed(args.seed)
        query = torch.randn(*shape, dtype=torch.bfloat16)
        key = torch.randn(*shape, dtype=torch.bfloat16)
        value = torch.randn(*shape, dtype=torch.bfloat16)

        off = attention_under(False, query, key, value)
        on = attention_under(True, query, key, value)
        same = torch.equal(off, on)
        gap = float((off.float() - on.float()).abs().max())
        moved += not same
        print(f"  {name:24} {tuple(shape)!s:22} bitwise equal {same!s:5} "
              f"max abs diff {gap:.6g}")

    print()
    if moved:
        print(f"the setting moves the output on {moved} of {len(SHAPES)} shapes, "
              f"so the math backend needs it held")
        return 0
    print("the setting moved nothing, so holding it on the math path buys nothing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

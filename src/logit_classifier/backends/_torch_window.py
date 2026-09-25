"""The determinism window every backend's forward pass runs inside.

torch is imported inside the window rather than at module scope, so a backend over a
host's own model can import this without pulling torch into a core-only process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Two windows open on two threads would interleave their saves, and the second to
# exit would restore the pinned values rather than the host's. The service already
# serialises on its own GPU lock, which is always taken before this one.
_WINDOW = threading.RLock()


@contextmanager
def _determinism() -> Iterator[None]:
    """Hold the torch globals that decide bit-exact reductions, for one forward pass.

    Every alternative value buys speed by giving up bit-exactness, so none is tunable.
    torch exposes none of them as a call argument, so scoping them means setting them
    here and putting the host's values back after. A ComfyUI host sharing this process
    keeps its own settings everywhere outside the block.

    On the CUDA attention path only allow_bf16_reduced_precision_reduction moves a logit
    on either shipped model. The rest are kept because cudnn picks convolution algorithms
    by timing, which is specific to the card, and `ab_determinism_scope.py` measured one.

    The sdp and fp16 accumulation settings are held for a third reason, that ComfyUI turns
    each of them on and neither is reachable by that sweep. `ab_math_sdp_reduction.py`
    measures the sdp one on the math backend.

    Where torch has per-backend matmul slots, the coarse getter raises once a host has
    set a slot directly, so only the raw slots are saved, pinned and put back there.
    """
    import torch

    with _WINDOW:
        saved_benchmark = torch.backends.cudnn.benchmark
        saved_deterministic = torch.backends.cudnn.deterministic
        saved_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        # torch ships no stub for the mkldnn matmul slot, so it is reached by name.
        mkldnn: Any = getattr(torch.backends.mkldnn, "matmul", None)
        has_slots = hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
            mkldnn, "fp32_precision"
        )
        saved_matmul = None if has_slots else torch.get_float32_matmul_precision()
        saved_cuda = torch.backends.cuda.matmul.fp32_precision if has_slots else None
        saved_mkldnn = mkldnn.fp32_precision if has_slots else None
        # ComfyUI turns this on at import, at comfy/model_management.py:569. It governs
        # the math attention backend, which runs only when attention_backends is empty.
        has_sdp = hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp") and hasattr(
            torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed"
        )
        saved_sdp = bool(
            has_sdp and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        )
        # A bare --fast turns this on, since comfy/cli_args.py then enables every
        # PerformanceFeature. It is the fp16 sibling of the bf16 reduction above.
        has_fp16_acc = hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
        saved_fp16_acc = bool(
            has_fp16_acc and torch.backends.cuda.matmul.allow_fp16_accumulation
        )

        # Pinning sits inside the try, so a setter that raises part way still restores the host.
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            if has_slots:
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                mkldnn.fp32_precision = "ieee"
            else:
                torch.set_float32_matmul_precision("highest")
            if has_sdp:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            if has_fp16_acc:
                torch.backends.cuda.matmul.allow_fp16_accumulation = False
            yield
        finally:
            torch.backends.cudnn.benchmark = saved_benchmark
            torch.backends.cudnn.deterministic = saved_deterministic
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = saved_bf16
            if has_slots:
                torch.backends.cuda.matmul.fp32_precision = saved_cuda
                mkldnn.fp32_precision = saved_mkldnn
            elif saved_matmul is not None:
                torch.set_float32_matmul_precision(saved_matmul)
            if has_sdp:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(saved_sdp)
            if has_fp16_acc:
                torch.backends.cuda.matmul.allow_fp16_accumulation = saved_fp16_acc

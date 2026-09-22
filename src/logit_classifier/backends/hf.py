"""The transformers backend: model loading and the single-forward-pass branch scorer.

No token is ever generated. Every probability comes from the logit row at one
position, the position the answer prefill forces to be the answer.
"""

from __future__ import annotations

import copy
import os
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# cuBLAS needs a fixed workspace before torch initialises CUDA to keep its
# reductions reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.cache_utils import DynamicCache

from ..config import Config, canonical_model_id
from .base import BranchLogits, VisionUnsupportedError, verify_backend

# Expanding the shared prefix across batch rows costs this much per row, so the
# batch has to shrink as the context grows.
KV_BUDGET_BYTES = 6 * 1024**3

# A chunk's suffix tokens, rows times padded width, checked only when a branch widens
# the chunk it joins. The KV budget bounds what the prefix cache costs, and this bounds
# what a wide branch can charge the narrow ones beside it. Anything from 512 to 2048
# measured the same, and 4096 upward is clearly worse. ab_branch_packing.py.
CHUNK_TOKEN_CEILING = 2048

# torch 2.11 ships no FlashAttention kernel on Windows. Left to choose for itself the
# dispatcher then reaches the math backend, which builds the full attention matrix. On an
# 8k prefill that measured 147 seconds and 27.6 GB against 1.3 seconds and 9.2 GB here.
PREFERRED_ATTENTION = (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)

# When no preferred backend probes clean, the host's own enable flags would otherwise
# decide which kernel runs, so two hosts could answer one request differently. Pinning
# this order makes the choice a function of the hardware. Math is last and always works.
FALLBACK_ATTENTION = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _usable_attention_backends(device: torch.device, dtype: torch.dtype) -> tuple[SDPBackend, ...]:
    """Probe which preferred backends have a kernel for this dtype on this build.

    A backend that serves bfloat16 can be missing for another dtype, so the probe has to
    run at the dtype the model will use.
    """
    usable: list[SDPBackend] = []

    if device.type != "cuda":
        return ()
    # The probe draws from its own generator so that constructing a backend does not
    # shift the host's global RNG stream.
    generator = torch.Generator(device=device)
    query = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, generator=generator)
    key = torch.randn(1, 2, 64, 64, device=device, dtype=dtype, generator=generator)
    mask = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
    attention = torch.nn.functional.scaled_dot_product_attention
    for backend in PREFERRED_ATTENTION:
        try:
            # A rejected backend warns on the way out, which is the answer we came for.
            with warnings.catch_warnings(), sdpa_kernel(backend):
                warnings.simplefilter("ignore", UserWarning)
                attention(query, key, key, is_causal=True, enable_gqa=True)
                attention(query, key, key, attn_mask=mask, enable_gqa=True)
            usable.append(backend)
        except RuntimeError:
            continue
    return tuple(usable)


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
        try:
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


class HFBackend:
    """Reads logits off a Hugging Face model held in this process."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_id = config.model_id
        self.canonical_model_id = canonical_model_id(config.model_id)
        # None is what transformers already means by "use the HF cache", so one keyword
        # covers both the project folder and the global location with no branch here.
        cache_dir = str(config.models_dir) if config.models_dir else None

        loaded = AutoConfig.from_pretrained(config.model_id, cache_dir=cache_dir)
        self.sees_images = hasattr(loaded, "vision_config")
        self.processor = None

        if self.sees_images:
            self.processor = AutoProcessor.from_pretrained(config.model_id, cache_dir=cache_dir)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                cache_dir=cache_dir,
                dtype=getattr(torch, config.dtype),
                device_map=config.device,
            ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                cache_dir=cache_dir,
                dtype=getattr(torch, config.dtype),
                device_map=config.device,
            ).eval()

        self.label_ids = verify_backend(self)
        self._label_id_tensor = torch.tensor(self.label_ids, device=self.model.device)
        self._pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._kv_bytes_per_token = self._measure_kv_bytes_per_token()
        self.attention_backends = _usable_attention_backends(self.model.device, self.model.dtype)

    @contextmanager
    def _pinned_forward(self) -> Iterator[None]:
        """Establish the environment every forward pass in this class needs.

        The attention backend and the determinism globals are both process wide, so
        each pass sets them and puts them back rather than pinning them at load.
        Every self.model call belongs inside this block.
        """
        with _determinism():
            if not self.attention_backends:
                with sdpa_kernel(list(FALLBACK_ATTENTION), set_priority=True):
                    yield
                return
            with sdpa_kernel(list(self.attention_backends)):
                yield

    def _measure_kv_bytes_per_token(self) -> int:
        cfg = getattr(self.model.config, "text_config", self.model.config)
        heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        element = torch.empty((), dtype=getattr(torch, self.config.dtype)).element_size()
        return int(2 * cfg.num_hidden_layers * heads * head_dim * element)

    def render(self, system: str, user: str, prefill: str, *, open_ended: bool = False) -> str:
        """Apply the chat template, then append the answer prefill.

        open_ended returns only the user-message body, before the template closes the
        turn, which is exactly the span every branch shares.
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rendered: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not open_ended:
            return rendered + prefill
        return rendered[: rendered.rindex(user) + len(user)]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False)
        return ids

    def encode_prefix(self, text: str, image: Any = None) -> tuple[list[int], dict[str, Any]]:
        """Prefix token ids, plus the vision tensors its forward pass needs.

        The processor expands the single image marker into one token per patch, so
        the count is decided here rather than guessed. Only the prefix carries an
        image, which is what keeps every branch a plain text suffix.
        """
        if image is None:
            return self.encode(text), {}
        if not self.sees_images or self.processor is None:
            raise VisionUnsupportedError(
                f"{self.config.model_id} has no vision tower, so it cannot read an image"
            )
        batch = self.processor(text=[text], images=[image], return_tensors="pt",
                               add_special_tokens=False)
        ids: list[int] = batch["input_ids"][0].tolist()
        vision = {k: v.to(self.model.device) for k, v in batch.items()
                  if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}
        return ids, vision

    def _rows_per_chunk(self, prefix_len: int, suffix_len: int) -> int:
        if not self.config.batch_branches:
            return 1
        per_row = self._kv_bytes_per_token * (prefix_len + suffix_len)
        affordable = max(1, KV_BUDGET_BYTES // max(per_row, 1))
        return int(min(self.config.max_batch_rows, affordable))

    def _pack_chunks(self, suffix_ids: list[list[int]], prefix_len: int) -> list[list[int]]:
        """Group branch indices into chunks of similar suffix length.

        Every row in a chunk is left-padded to the chunk's longest suffix, and each
        padded token costs a full forward plus attention over the whole prefix. Taking
        branches in request order drags short ones to the longest width, which measured
        86.7 percent waste on a mixed request. The sort is stable, so equal lengths keep
        request order and the grouping stays a pure function of the request.
        """
        if not self.config.batch_branches:
            return [[index] for index in range(len(suffix_ids))]

        chunks: list[list[int]] = []
        current: list[int] = []

        for index in sorted(range(len(suffix_ids)), key=lambda i: len(suffix_ids[i])):
            width = len(suffix_ids[index])
            allowed = self._rows_per_chunk(prefix_len, width)
            rows = len(current) + 1
            # The ceiling exists to stop one wide branch dragging narrow ones out to
            # its width. A candidate no wider than the chunk adds no padding, so only
            # the row cap applies and a request of one shape chunks as it always did.
            widens = bool(current) and width > len(suffix_ids[current[-1]])
            over_ceiling = widens and rows * width > CHUNK_TOKEN_CEILING
            if current and (rows > allowed or over_ceiling):
                chunks.append(current)
                current = [index]
            else:
                current.append(index)
        if current:
            chunks.append(current)
        return chunks

    @torch.inference_mode()
    def score(
        self, prefix_ids: list[int], suffix_ids: list[list[int]], label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Score every branch against one shared prefix.

        The prefix is encoded once. Each chunk then copies that cache, broadcasts
        it across the chunk's rows, and reads the final position of every row in
        a single forward pass.
        """
        device = self.model.device
        scored: dict[int, BranchLogits] = {}
        rope_delta: torch.Tensor | None = None

        if not suffix_ids:
            return []

        prefix_tensor = torch.tensor([prefix_ids], device=device)
        seed = DynamicCache()
        with self._pinned_forward():
            self.model(
                input_ids=prefix_tensor,
                attention_mask=torch.ones_like(prefix_tensor),
                past_key_values=seed,
                use_cache=True,
                logits_to_keep=1,
                **(vision or {}),
            )
            # An image makes positions three dimensional and shifts every later token, so
            # the branches reuse the offset this pass wrote before another pass overwrites it.
            if vision:
                rope_delta = getattr(self.model.model, "rope_deltas", None)
                if rope_delta is not None:
                    rope_delta = rope_delta.clone()

        # The port promises one row per suffix in the order given, so the packed
        # chunks are scattered back rather than concatenated.
        for chunk in self._pack_chunks(suffix_ids, len(prefix_ids)):
            rows = self._score_chunk(
                prefix_ids, seed,
                [suffix_ids[i] for i in chunk], [label_counts[i] for i in chunk], rope_delta,
            )
            for slot, row in zip(chunk, rows, strict=True):
                scored[slot] = row
        return [scored[index] for index in range(len(suffix_ids))]

    def _score_chunk(
        self,
        prefix_ids: list[int],
        seed: DynamicCache,
        suffix_ids: list[list[int]],
        label_counts: list[int],
        rope_delta: torch.Tensor | None = None,
    ) -> list[BranchLogits]:
        device = self.model.device
        rows = len(suffix_ids)
        prefix_len = len(prefix_ids)
        width = max(len(s) for s in suffix_ids)
        results: list[BranchLogits] = []

        # Left-padding puts every row's final real token at the same index, so a
        # single kept position serves the whole batch.
        input_ids = torch.full((rows, width), self._pad_id, dtype=torch.long)
        attention = torch.zeros((rows, prefix_len + width), dtype=torch.long)
        attention[:, :prefix_len] = 1
        positions = torch.zeros((rows, width), dtype=torch.long)
        for row, suffix in enumerate(suffix_ids):
            pad = width - len(suffix)
            input_ids[row, pad:] = torch.tensor(suffix)
            attention[row, prefix_len + pad :] = 1
            positions[row, pad:] = torch.arange(prefix_len, prefix_len + len(suffix))

        placed = positions.to(device)
        if rope_delta is not None:
            placed = (placed + rope_delta.to(device)).unsqueeze(0).expand(3, rows, width)

        cache = copy.deepcopy(seed)
        cache.batch_repeat_interleave(rows)
        with self._pinned_forward():
            output = self.model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                position_ids=placed,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        final = output.logits[:, -1, :].float()

        # Index on device so only the handful of label logits crosses the bus.
        selected = final.index_select(1, self._label_id_tensor)
        full_norm = torch.logsumexp(final, dim=-1)
        del cache, output, final

        for row, count in enumerate(label_counts):
            z = selected[row, :count]
            mass = float(torch.exp(torch.logsumexp(z, dim=-1) - full_norm[row]))
            results.append(BranchLogits(z=z.double().cpu().numpy(), candidate_mass=mass))
        return results

"""A backend over a ComfyUI Qwen3-VL CLIP the host workflow already loaded.

No comfy module is imported at module scope. The CLIP is read by attribute, against
ComfyUI core's comfy/sd.py, comfy/sd1_clip.py and comfy/text_encoders/qwen3vl.py and
llama.py. torch and comfy are imported inside the calls that need them, since the host
already holds both.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

from ..config import canonical_model_id
from ..vision import ImageError
from ._torch_window import _determinism
from .base import (
    BackendContractError,
    BranchLogits,
    UnsupportedModelError,
    VisionUnsupportedError,
    verify_backend,
)

# ComfyUI names a Qwen3-VL text encoder by size, and FITTED_TEMPERATURES keys on the
# Hugging Face repo id.
_REPO_IDS = {
    "qwen3vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
}

# Core names every Qwen3-VL encoder qwen3vl_<size>, and only that family builds the 3D image
# positions the packed pass reads. Qwen2.5-VL and Qwen3.5 encoders pass every other check.
_FAMILY_PREFIX = "qwen3vl_"

# Core swaps an image entry in at this id and no other, at comfy/text_encoders/qwen3vl.py:194.
IMAGE_PAD_ID = 151655

_SUPPORTED = "a Qwen3-VL text encoder, such as the one Krea 2 loads"

# Bounds the L x L block mask and the masked attention's memory in one packed pass.
# Unmeasured beyond the prototype's largest pass, 1,610 tokens.
PACKED_TOKEN_CEILING = 4096


class _PackedPrefix(NamedTuple):
    """The prefix as core's text model encodes it, once per score call."""

    embeds: Any
    positions: Any
    visual_mask: Any
    deepstack: Any
    info: Any


def _clip_parts(clip: Any) -> tuple[Any, Any, str, str]:
    """Return the Hugging Face tokenizer, the transformer and the two encoder names.

    comfy.sd.CLIP defines generate and decode for every text encoder, so the encoder name,
    the generating transformer and the Qwen chat tokens together tell Qwen3-VL apart.
    """
    tokenizer_name = getattr(getattr(clip, "tokenizer", None), "clip", None)
    encoder_name = getattr(getattr(clip, "cond_stage_model", None), "clip", None)
    tokenizer: Any = None
    transformer: Any = None

    if isinstance(tokenizer_name, str):
        tokenizer = getattr(getattr(clip.tokenizer, tokenizer_name, None), "tokenizer", None)
    if isinstance(encoder_name, str):
        transformer = getattr(getattr(clip.cond_stage_model, encoder_name, None), "transformer", None)
    family = isinstance(encoder_name, str) and encoder_name.startswith(_FAMILY_PREFIX)
    generates = callable(getattr(clip, "generate", None)) and callable(getattr(clip, "decode", None))
    samples = callable(getattr(transformer, "sample_token", None)) and callable(
        getattr(transformer, "logits", None)
    )
    encodes = callable(getattr(tokenizer, "encode", None))
    chat_tokens = encodes and all(
        len(tokenizer.encode(token, add_special_tokens=False)) == 1
        for token in ("<|im_start|>", "<|im_end|>")
    )

    if not (family and generates and samples and chat_tokens):
        raise UnsupportedModelError(
            f"{type(clip).__name__} with text encoder {encoder_name!r} is not a generating Qwen3-VL "
            f"text model, so no logit row can be read from it. Load {_SUPPORTED}"
        )
    return tokenizer, transformer, str(tokenizer_name), str(encoder_name)


def _check_image(image: Any) -> None:
    import torch

    shape = tuple(getattr(image, "shape", ()))

    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise ImageError(
            "a ComfyUI IMAGE, a float torch tensor of shape [1, H, W, 3] in 0..1, is required, "
            f"got {type(image).__name__}"
        )
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise ImageError(f"a ComfyUI IMAGE of shape [1, H, W, 3] is required, got shape {shape}")


def _pack_passes(prefix_length: int, suffix_lengths: list[int], batch_branches: bool) -> list[list[int]]:
    """Group suffixes in request order into passes that stay under PACKED_TOKEN_CEILING.

    The grouping reads only lengths known before the first pass, so one request always
    packs one way (prime directive 3). A suffix too long to share a pass gets its own.
    """
    passes: list[list[int]] = []
    current: list[int] = []
    width = prefix_length

    for index, length in enumerate(suffix_lengths):
        fits = batch_branches and bool(current) and width + length <= PACKED_TOKEN_CEILING
        if current and not fits:
            passes.append(current)
            current = []
            width = prefix_length
        current.append(index)
        width += length
    if current:
        passes.append(current)
    return passes


def _block_mask(prefix_length: int, suffix_lengths: list[int], device: Any) -> Any:
    """Build a (1, L, L) mask where each suffix sees the prefix and itself, and no other suffix."""
    import torch

    total = prefix_length + sum(suffix_lengths)
    mask = torch.zeros((1, total, total), device=device)
    start = prefix_length

    mask[0, :prefix_length, :prefix_length] = torch.tril(torch.ones((prefix_length, prefix_length), device=device))
    for length in suffix_lengths:
        end = start + length
        mask[0, start:end, :prefix_length] = 1
        mask[0, start:end, start:end] = torch.tril(torch.ones((length, length), device=device))
        start = end
    return mask


def _pass_positions(prefix_positions: Any, suffix_lengths: list[int]) -> Any:
    """Prefix positions, then every suffix restarting right after the prefix's largest."""
    import torch

    rows = prefix_positions.shape[0]
    after = prefix_positions.max() + 1
    parts = [prefix_positions]

    for length in suffix_lengths:
        restart = torch.arange(length, device=prefix_positions.device) + after
        parts.append(restart.unsqueeze(0).expand(rows, length))
    return torch.cat(parts, dim=1)


class ComfyClipBackend:
    """Reads logits off a ComfyUI CLIP holding a Qwen3-VL text encoder.

    score packs every branch behind one shared prefix into one forward pass over core's
    transformer. If core's internals have moved, it falls back to one single-step
    clip.generate per branch, reading the row at sample_token.

    Packing can move the low bits of a question's probability with the questions sent
    beside it. batch_branches=False gives every branch its own pass, so a question's
    logits depend on it alone. LOGIT_BATCH_BRANCHES does not reach this backend, since it
    holds no Config.
    """

    def __init__(self, clip: Any, *, model_id: str | None = None, batch_branches: bool = True) -> None:
        tokenizer, transformer, tokenizer_name, encoder_name = _clip_parts(clip)

        self.clip = clip
        self._tokenizer = tokenizer
        self._transformer = transformer
        self._tokens_key = tokenizer_name
        self._batch_branches = batch_branches
        self._packed_failed = False
        self.model_id = model_id if model_id is not None else _REPO_IDS.get(encoder_name, encoder_name)
        self.canonical_model_id = canonical_model_id(self.model_id)
        self.sees_images = hasattr(transformer, "visual")
        self.label_ids = verify_backend(self)

    def render(self, system: str, user: str, prefill: str, *, open_ended: bool = False) -> str:
        """Render the Qwen chat template HFBackend renders, then append the prefill.

        Core's tokenizer appends an empty think block that HFBackend's template lacks, so
        the template is written here rather than taken from clip.tokenize.
        """
        opened = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}"

        if open_ended:
            return opened
        return f"{opened}<|im_end|>\n<|im_start|>assistant\n{prefill}"

    def encode(self, text: str) -> list[int]:
        """Token ids from the Hugging Face tokenizer itself.

        clip.tokenize drops a backslash before a parenthesis, reads embedding: as a file
        name and pads empty text, at comfy/sd1_clip.py:368-376, 584-616 and 668-669.
        """
        ids: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        return ids

    def encode_prefix(self, text: str, image: Any = None) -> tuple[list[int], dict[str, Any]]:
        """Prefix token ids, plus the ComfyUI IMAGE and the position score puts it back at.

        Core's vision tower expands the one image pad token into its patches, so the
        prefix carries exactly one.
        """
        ids: list[int] = []
        pads: list[int] = []

        if image is None:
            return self.encode(text), {}
        if not self.sees_images:
            raise VisionUnsupportedError(f"{self.model_id} has no vision tower, so it cannot read an image")
        _check_image(image)
        ids = self.encode(text)
        pads = [index for index, token in enumerate(ids) if token == IMAGE_PAD_ID]
        if len(pads) != 1:
            raise ImageError(
                f"the prefix holds {len(pads)} <|image_pad|> tokens, and one image needs exactly one"
            )
        return ids, {"image": image, "pad_index": pads[0]}

    def score(
        self,
        prefix_ids: list[int],
        suffix_ids: list[list[int]],
        label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Score every branch in packed passes, or per branch once packing has failed here.

        Only AttributeError and TypeError fall back, the errors a moved core internal
        raises. Anything else, CUDA out of memory included, reaches the caller.
        """
        rows: list[BranchLogits] = []

        if not self._packed_failed:
            try:
                return self._score_packed(prefix_ids, suffix_ids, label_counts, vision)
            except (AttributeError, TypeError) as error:
                # A fallback that fails too points at the input, not at core, so packing stays on.
                rows = self._score_by_generate(prefix_ids, suffix_ids, label_counts, vision)
                self._packed_failed = True
                warnings.warn(
                    f"packed scoring failed on this ComfyUI core ({type(error).__name__}: {error}), "
                    "so the slower per-branch generate path is used from now on",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return rows
        return self._score_by_generate(prefix_ids, suffix_ids, label_counts, vision)

    def _score_packed(
        self,
        prefix_ids: list[int],
        suffix_ids: list[list[int]],
        label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Encode the prefix once, then run each pass of packed suffixes as one forward.

        The call sequence is CLIP.generate's, at comfy/sd.py:475-484, minus the generate.
        """
        import comfy.model_management as model_management
        import comfy.ops
        import torch

        clip = self.clip
        encoder = clip.cond_stage_model
        text_model = getattr(encoder, encoder.clip)
        # Bare ids and a bare image dict, the form Qwen3VLClipModel.generate hands
        # process_tokens after stripping weights, at comfy/text_encoders/qwen3vl.py:133.
        prefix_tokens: list[Any] = list(prefix_ids)
        results: list[BranchLogits] = []

        if not suffix_ids:
            return results
        if vision:
            prefix_tokens[vision["pad_index"]] = {"type": "image", "data": vision["image"], "original_type": "image"}
        with _determinism(), torch.inference_mode():
            encoder.reset_clip_options()
            clip.load_model({self._tokens_key: [prefix_tokens]})
            device = clip.patcher.load_device
            encoder.set_clip_options({"layer": None, "execution_device": device})
            # BaseGenerate.generate picks its execution dtype this way, at llama.py:1115-1120.
            dtype = torch.bfloat16 if model_management.should_use_bf16(device) else torch.float32
            with model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(encoder, device):
                prefix = self._encode_packed_prefix(text_model, prefix_tokens, device)
                passes = _pack_passes(
                    prefix.embeds.shape[1], [len(suffix) for suffix in suffix_ids], self._batch_branches
                )
                for members in passes:
                    rows = self._packed_rows(prefix, [suffix_ids[index] for index in members], dtype)
                    for index, row in zip(members, rows, strict=True):
                        results.append(self._branch_logits(row, label_counts[index]))
        return results

    def _encode_packed_prefix(self, text_model: Any, prefix_tokens: list[Any], device: Any) -> _PackedPrefix:
        """Run the vision tower and embed the prefix, the only work every pass shares."""
        import torch

        embeds, _, _, info = text_model.process_tokens([prefix_tokens], device)
        positions, visual_mask, deepstack = self._transformer.build_image_inputs(embeds, info)

        if positions is None:
            positions = torch.arange(embeds.shape[1], device=device).unsqueeze(0)
        return _PackedPrefix(embeds, positions, visual_mask, deepstack, info)

    def _packed_rows(self, prefix: _PackedPrefix, suffixes: list[list[int]], dtype: Any) -> Any:
        """One forward over the prefix and these suffixes, returning each suffix's last row."""
        import torch

        transformer = self._transformer
        device = prefix.embeds.device
        prefix_length = prefix.embeds.shape[1]
        lengths = [len(suffix) for suffix in suffixes]
        flat_ids = [token for suffix in suffixes for token in suffix]
        ends: list[int] = []
        visual_mask = prefix.visual_mask

        for length in lengths:
            ends.append((ends[-1] if ends else prefix_length) + length)
        # The call process_tokens makes for text, at comfy/sd1_clip.py:212-213.
        suffix_embeds = transformer.get_input_embeddings()(
            torch.tensor([flat_ids], device=device, dtype=torch.long), out_dtype=torch.float32
        )
        embeds = torch.cat([prefix.embeds, suffix_embeds], dim=1).to(dtype)
        if visual_mask is not None:
            suffix_mask = torch.zeros((1, len(flat_ids)), dtype=torch.bool, device=device)
            visual_mask = torch.cat([visual_mask, suffix_mask], dim=1)
        # Core reshapes this to (B, 1, L, L) and adds its own causal mask, at llama.py:929-941.
        hidden = transformer.model.forward(
            None,
            embeds=embeds,
            attention_mask=_block_mask(prefix_length, lengths, device),
            position_ids=_pass_positions(prefix.positions, lengths),
            deepstack_embeds=prefix.deepstack,
            visual_pos_masks=visual_mask,
            embeds_info=prefix.info,
        )[0]
        last = torch.tensor([end - 1 for end in ends], device=device)
        # transformer.logits handles the tied embedding and core's weight casting.
        return transformer.logits(hidden[0, last].unsqueeze(1))[:, -1].float()

    def _score_by_generate(
        self,
        prefix_ids: list[int],
        suffix_ids: list[list[int]],
        label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Run one single-step generate per branch over the prefix plus that branch's suffix."""
        import torch

        prefix: list[tuple[Any, float]] = [(token, 1.0) for token in prefix_ids]
        results: list[BranchLogits] = []

        # Core's vision path reads only the entry its own tokenizer builds, at qwen3vl.py:192-197.
        if vision:
            entry = {"type": "image", "data": vision["image"], "original_type": "image"}
            prefix[vision["pad_index"]] = (entry, 1.0)
        with torch.inference_mode():
            for suffix, count in zip(suffix_ids, label_counts, strict=True):
                row = self._logit_row(prefix + [(token, 1.0) for token in suffix])
                results.append(self._branch_logits(row, count))
        return results

    def _logit_row(self, tokens: list[tuple[Any, float]]) -> Any:
        """Capture the full vocabulary row sample_token receives on the prefill step.

        The wrapper still returns the sampled token, since core copies it into
        decode_tokens right after the call, at comfy/text_encoders/llama.py:1170-1172.
        """
        transformer = self._transformer
        original = transformer.sample_token
        shadowed = "sample_token" in vars(transformer)
        previous = vars(transformer).get("sample_token")
        captured: list[Any] = []

        def capture(logits: Any, *args: Any, **kwargs: Any) -> Any:
            captured.append(logits)
            return original(logits, *args, **kwargs)

        # Inside the window, whose lock keeps two threads from interleaving the swap.
        with _determinism():
            transformer.sample_token = capture
            try:
                self.clip.generate({self._tokens_key: [tokens]}, do_sample=False, max_length=1)
            finally:
                if shadowed:
                    transformer.sample_token = previous
                else:
                    del transformer.sample_token

        if not captured:
            raise BackendContractError(
                f"{type(self.clip).__name__}.generate never reached sample_token, so no logit row was read"
            )
        return captured[0][0].float()

    def _branch_logits(self, row: Any, count: int) -> BranchLogits:
        import torch

        # Index on device so only the handful of label logits crosses the bus.
        index = torch.tensor(self.label_ids[:count], device=row.device)
        selected = row.index_select(0, index)
        full_norm = torch.logsumexp(row, dim=0)
        mass = float(torch.exp(torch.logsumexp(selected, dim=0) - full_norm))
        return BranchLogits(z=selected.double().cpu().numpy(), candidate_mass=mass)

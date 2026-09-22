"""Reading logits off a model you already loaded, instead of loading another one.

    uv run python examples/own_backend.py

This example needs no weights or GPU. It stands in a toy scorer for the one
method a real host would implement, so the port itself is what you see.

A host that already holds a language model passes its own object to Classifier and
the library never touches transformers. That is how a ComfyUI node runs this over a
text encoder the workflow already loaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from logit_classifier import (
    ANSWER_PREFILL,
    Backend,
    BranchLogits,
    Classifier,
    Config,
    parse_request,
    verify_backend,
)


class MyBackend:
    """The six members Backend asks for."""

    # The string the fitted temperature is looked up by. Report the model you
    # actually loaded, or the answer is calibrated for a different one.
    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    # Whether this model has a vision tower. False rejects a picture with a named
    # error rather than silently dropping it.
    sees_images = False

    def __init__(self) -> None:
        # verify_backend proves the two render contracts and returns one token id
        # per label, checked against the rendered prefill rather than the bare
        # letter. A prefill ending in a space merges into the letter after it.
        self.label_ids = verify_backend(self)

    def render(self, system: str, user: str, prefill: str, *, open_ended: bool = False) -> str:
        """Wrap the message bodies in your model's chat template.

        open_ended returns the span up to the end of the user body, which every
        branch of one request shares. The closed render must begin with it, character
        for character, or the shared prefix cannot be split off by offset.
        """
        body = f"<|system|>{system}<|user|>{user}"
        if open_ended:
            return body
        return f"{body}<|assistant|>{prefill}"

    def encode(self, text: str) -> list[int]:
        """Token ids for a fragment, with no special tokens added."""
        return [ord(character) for character in text]

    def encode_prefix(self, text: str, image: Any = None) -> tuple[list[int], dict[str, Any]]:
        """Prefix ids, plus whatever tensors the forward pass needs for an image.

        Whatever you put in the dict is handed back to score unchanged.
        """
        return self.encode(text), {}

    def score(
        self,
        prefix_ids: list[int],
        suffix_ids: list[list[int]],
        label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Read the label logits at each branch's final position.

        Replace this body with one forward pass over prefix_ids + suffix, reading
        the logit row at the last position and gathering self.label_ids from it.
        Return one row per suffix, in the order given, each label_counts[i] wide
        and ordered to match LABEL_ALPHABET.
        """
        rows: list[BranchLogits] = []
        for suffix, count in zip(suffix_ids, label_counts, strict=True):
            # A toy stand-in for a logit row, deterministic so the example repeats.
            seed = np.random.default_rng(abs(hash(tuple(suffix))) % (2**32))
            rows.append(BranchLogits(z=seed.normal(size=count), candidate_mass=0.9))
        return rows


def main() -> None:
    backend = MyBackend()

    # Runtime checkable, so a host can assert its own object before passing it.
    print(f"satisfies the port: {isinstance(backend, Backend)}")
    print(f"answer prefill:     {ANSWER_PREFILL!r}")
    print(f"label ids proven:   {len(backend.label_ids)}")

    # No model_id on Config, no weights loaded. The backend owns both.
    classifier = Classifier(Config(), backend=backend)
    print(f"temperature:        {classifier.temperature}  (from backend.model_id)")

    response, _ = classifier.classify(parse_request({
        "state": "the build failed again after the upgrade",
        "questions": {
            "area": {
                "type": "choice",
                "instructions": "Which area does this belong to",
                "criteria": {"build": None, "runtime": None, "docs": None},
            },
            "is_regression": {"type": "noul", "instructions": "This is a regression"},
        },
    }))

    print()
    for name, answer in response.answers.items():
        print(f"{name}: {answer}")


if __name__ == "__main__":
    main()

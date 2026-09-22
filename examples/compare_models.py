"""Answering the same questions with two models in one command.

    uv run python examples/compare_models.py

Each model is loaded, asked, then released before the next one loads, so a single
card only ever holds one set of weights. Drop the release if you have room for both
and want them side by side.

The fitted temperature follows the backend, not the config, so each model is scored
with its own value.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from logit_classifier import Classifier, Config, load_model, parse_request

# Weights land beside the project instead of in the global Hugging Face cache.
# Set LOGIT_MODELS_DIR, or HF_HOME, to keep them somewhere shared across projects.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]

REQUEST = {
    "state": "Shipped two days late and the box was crushed, but the product works fine.",
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "What is this review mainly about",
            "criteria": {"delivery": None, "quality": None, "price": None},
        },
        "mentions_damage": {
            "type": "noul",
            "instructions": "The review mentions physical damage",
        },
    },
}


def release() -> None:
    """Free the weights of a model the caller holds no reference to any more."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    config = Config(models_dir=MODELS_DIR)
    request = parse_request(REQUEST)

    for model_id in args.models:
        classifier = Classifier(config, backend=load_model(model_id, config))
        response, _ = classifier.classify(request)

        topic = response.answers["topic"]
        damage = response.answers["mentions_damage"]
        print(f"{model_id}")
        print(f"   temperature      {classifier.temperature}")
        print(f"   topic            {topic.choice}  confidence {topic.confidence:.4f}")
        print(f"   mentions_damage  {damage.noul:.4f}")
        del classifier, response, topic, damage
        release()


if __name__ == "__main__":
    main()

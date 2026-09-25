"""One request through the classifier, from the command line.

    uv run python examples/quickstart.py
    uv run python examples/quickstart.py --text "the payment failed again"
    uv run python examples/quickstart.py --image tests-AB/inputs/top-L.png

The model loads on the first call and answers every question below in one pass. A text
model rejects --image rather than ignoring it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from logit_classifier import Classifier, Config, load_model, parse_request

# Weights land beside the project instead of in the global Hugging Face cache.
# Edit MODELS_DIR to share one folder across projects.
# Neither LOGIT_MODELS_DIR nor HF_HOME reaches this script.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SAMPLE_TEXT = "I have been trying to connect my Stripe account for 3 days and it keeps failing."

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pre-purchase questions",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the writer appears",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
}

IMAGE_QUESTIONS = {
    "subject": {
        "type": "choice",
        "instructions": "What fills most of this picture",
        "criteria": {"a person": None, "a landscape": None, "an object": None, "text": None},
    },
    "is_night": {"type": "noul", "instructions": "This picture was taken at night"},
}


def report(answers: dict) -> None:
    for name, answer in answers.items():
        if answer.type == "noul":
            print(f"{name}: {answer.noul:.4f}")
            continue
        if answer.type == "score":
            print(f"{name}: level {answer.score:.2f}, confidence {answer.confidence:.4f}")
        else:
            print(f"{name}: {answer.choice}, confidence {answer.confidence:.4f}, abstain {answer.abstain}")
        for option, probability in answer.probabilities.items():
            print(f"    {option:<24} {probability:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one state a few questions.")
    parser.add_argument("--text", default=SAMPLE_TEXT, help="the text to classify")
    parser.add_argument("--image", default=None, help="a picture to classify instead of the text")
    parser.add_argument("--model", default=None, help="a model id, else the configured default")
    args = parser.parse_args()

    config = Config(models_dir=MODELS_DIR)
    backend = load_model(args.model, config)
    classifier = Classifier(config, backend=backend)
    state = {"image": args.image, "caption": args.text} if args.image else args.text
    questions = IMAGE_QUESTIONS if args.image else QUESTIONS

    print(f"model {backend.model_id}, reads images: {backend.sees_images}\n")
    response, _diagnostics = classifier.classify(
        parse_request({"state": state, "questions": questions})
    )
    report(response.answers)
    print(f"\n{response.usage.input_tokens} input tokens, {response.usage.output_tokens} branches")


if __name__ == "__main__":
    main()

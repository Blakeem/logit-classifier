"""Classifying a picture, and a picture with text beside it.

    uv run python examples/images.py --image tests-AB/inputs/top-L.png

The default model reads images. Put the picture under `image` or `screenshot` in a
mapping state, as a file path, a data URL or bare base64. Everything else in that
mapping is rendered as text beside it, so one request can ask about both.

A text only model raises VisionUnsupportedError rather than ignoring the picture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from logit_classifier import (
    Classifier,
    Config,
    VisionUnsupportedError,
    load_model,
    parse_request,
)

# Weights land beside the project instead of in the global Hugging Face cache.
# Set LOGIT_MODELS_DIR, or HF_HOME, to keep them somewhere shared across projects.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
QUESTIONS = {
    "subject": {
        "type": "choice",
        "instructions": "What fills most of this picture",
        "criteria": {"a person": None, "a landscape": None, "an object": None, "text": None},
    },
    "is_dark": {"type": "noul", "instructions": "This picture is dark or underexposed"},
    "quality": {
        "type": "score",
        "instructions": "Image quality",
        "criteria": ["Unusable", "Acceptable", "Sharp"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="a file path, a data URL or base64")
    parser.add_argument("--caption", default=None, help="text to read beside the picture")
    args = parser.parse_args()

    config = Config(models_dir=MODELS_DIR)
    classifier = Classifier(config, backend=load_model(config=config))
    if not classifier.backend.sees_images:
        print("this model has no vision tower, so it cannot read a picture")
        return

    # A bare string state is text only. A mapping carrying `image` or `screenshot`
    # is a picture, plus whatever other keys you put beside it.
    state: object = {"image": args.image}
    if args.caption:
        state = {"image": args.image, "caption": args.caption}

    try:
        response, _ = classifier.classify(parse_request({"state": state, "questions": QUESTIONS}))
    except VisionUnsupportedError as error:
        print(f"this model cannot read pictures: {error}")
        return

    subject = response.answers["subject"]
    print(f"subject    {subject.choice}  confidence {subject.confidence:.4f}")
    print(f"is_dark    {response.answers['is_dark'].noul:.4f}")
    # Legend keys are the level index as a string, matching the response JSON.
    quality = response.answers["quality"]
    print(f"quality    {quality.legend[str(round(quality.score))]}  score {quality.score:.2f}")


if __name__ == "__main__":
    main()

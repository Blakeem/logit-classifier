"""Every question type, and every field it answers with.

    uv run python examples/question_types.py

A choice picks one option. A score places the state on an ordered scale. A noul
answers one true or false statement. All three are answered in a single forward
pass when they arrive in one request.
"""

from __future__ import annotations

from pathlib import Path

from logit_classifier import Classifier, Config, load_model, parse_request

# Weights land beside the project instead of in the global Hugging Face cache.
# Set LOGIT_MODELS_DIR, or HF_HOME, to keep them somewhere shared across projects.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

REVIEW = "Shipped two days late and the box was crushed, but the product itself works fine."

REQUEST = {
    "state": REVIEW,
    "questions": {
        # A choice declares named options. Descriptions are optional, and null
        # means the name speaks for itself.
        "topic": {
            "type": "choice",
            "instructions": "What is this review mainly about",
            "criteria": {
                "delivery": "Shipping speed, packaging or damage in transit",
                "quality": "How well the product itself works",
                "price": None,
            },
        },
        # A score declares levels in order. The answer is a weighted position on
        # that scale, not a pick, so 0.0 is the first level and 2.0 is the last.
        "sentiment": {
            "type": "score",
            "instructions": "Overall sentiment of the review",
            "criteria": ["Negative", "Mixed", "Positive"],
        },
        # A noul is one statement. Ask one per fact, rather than one choice over
        # several facts, when more than one can be true at once.
        "mentions_damage": {
            "type": "noul",
            "instructions": "The review mentions physical damage",
        },
    },
}


def main() -> None:
    config = Config(models_dir=MODELS_DIR)
    classifier = Classifier(config, backend=load_model(config=config))
    response, _ = classifier.classify(parse_request(REQUEST))

    choice = response.answers["topic"]
    print("choice")
    print(f"  choice        {choice.choice}")
    print(f"  confidence    {choice.confidence:.4f}   how peaked the distribution is")
    print(f"  abstain       {choice.abstain}   how much it wants none of these")
    for option, probability in choice.probabilities.items():
        print(f"    {option:<12} {probability:.4f}")

    score = response.answers["sentiment"]
    print("\nscore")
    print(f"  score         {score.score:.4f}   0.0 is the first level, {len(score.legend) - 1}.0 the last")
    print(f"  confidence    {score.confidence:.4f}   how tightly the mass sits around one level")
    print(f"  legend        {score.legend}")
    for level, probability in score.probabilities.items():
        print(f"    {score.legend[level]:<12} {probability:.4f}")

    noul = response.answers["mentions_damage"]
    print("\nnoul")
    print(f"  noul          {noul.noul:.4f}   probability the statement is true")

    print(f"\n{response.usage.input_tokens} input tokens, {response.usage.output_tokens} branches")
    print("Every answer above came from one forward pass over the shared state.")


if __name__ == "__main__":
    main()

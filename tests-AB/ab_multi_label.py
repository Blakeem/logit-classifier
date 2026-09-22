"""Which shape tags an image crop with several true answers at once.

One choice over every fragment, against one binary question per fragment. The tiles in
inputs/ are a 2 by 3 split of a generated 768x1024 image, and the labels were read off
them by eye. Fragment 6 is true in all six tiles, so it cannot be ranked and is excluded.

    uv run python tests-AB/ab_multi_label.py
"""

from __future__ import annotations

import base64
import time

import numpy as np
from ab_env import INPUTS, auc

from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import parse_request

FRAGMENTS = [
    "a woman's face in close-up",
    "a large tree trunk with textured bark and ivy",
    "a red moon glowing orange-red",
    "a starry night sky",
    "tree branches arching overhead",
    "a misty forest floor",
    "the woman's dark flowing hair",
    "distant trees silhouetted against the sky",
]

TRUTH = {
    "top-L": [0, 1, 0, 1, 1, 0, 1, 1],
    "top-R": [0, 0, 1, 1, 0, 0, 1, 0],
    "mid-L": [1, 1, 0, 0, 1, 1, 1, 0],
    # Her face spans the tile boundary, so half of it is here. Labelled absent by eye at
    # first, and corrected after every method disagreed at 1.00.
    "mid-R": [1, 0, 0, 0, 0, 0, 1, 0],
    "low-L": [0, 0, 0, 0, 0, 1, 1, 0],
    "low-R": [0, 0, 0, 0, 0, 0, 1, 0],
}

# Blake asked whether an image model handles other words better than yes and no.
BINARY_WORDS = {"yes/no": ("yes", "no"),
                "present/absent": ("present", "absent"),
                "visible/hidden": ("visible", "hidden")}
PROMPT = ", ".join(FRAGMENTS)


def as_data_url(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((INPUTS / f"{name}.png").read_bytes()).decode()


def main() -> None:
    classifier = Classifier(Config(use_prior_debias=False))
    if not classifier.backend.sees_images:
        raise SystemExit(f"{classifier.config.model_id} has no vision tower")

    nouls = {f"f{i}": {"type": "noul", "instructions": f"This crop visibly contains {f}"}
             for i, f in enumerate(FRAGMENTS)}
    choice = {"tags": {"type": "choice",
                       "instructions": "Which of these is most visible in this crop",
                       "criteria": dict.fromkeys(FRAGMENTS)}}
    binaries = {f"{key}|{i}": {"type": "choice",
                               "instructions": f"Statement: this crop visibly contains {fragment}",
                               "criteria": {positive: None, negative: None}}
                for key, (positive, negative) in BINARY_WORDS.items()
                for i, fragment in enumerate(FRAGMENTS)}

    methods = ["noul", "choice", *BINARY_WORDS]
    scores = {m: [] for m in methods}
    labels, timings = [], {m: [] for m in methods}

    def ask(state, questions):
        return classifier.classify(parse_request(
            {"state": state, "questions": questions}))[0].answers

    for name, truth in TRUTH.items():
        state = {"image": as_data_url(name), "prompt_for_the_whole_image": PROMPT}

        began = time.perf_counter()
        answered = ask(state, nouls)
        timings["noul"].append((time.perf_counter() - began) * 1000)
        scores["noul"].append([answered[f"f{i}"].noul for i in range(len(FRAGMENTS))])

        began = time.perf_counter()
        picked = ask(state, choice)["tags"]
        timings["choice"].append((time.perf_counter() - began) * 1000)
        scores["choice"].append([picked.probabilities[f] for f in FRAGMENTS])

        began = time.perf_counter()
        answered = ask(state, binaries)
        spent = (time.perf_counter() - began) * 1000
        for key, (positive, _) in BINARY_WORDS.items():
            scores[key].append([answered[f"{key}|{i}"].probabilities[positive]
                                for i in range(len(FRAGMENTS))])
            timings[key].append(spent / len(BINARY_WORDS))
        labels.append(truth)

    labels = np.array(labels, dtype=float)
    rankable = [i for i in range(len(FRAGMENTS)) if 0 < labels[:, i].sum() < len(TRUTH)]
    print(f"{len(TRUTH)} tiles, {int(labels.sum())} true pairs of {labels.size}, "
          f"{len(rankable)} fragments rankable\n")
    print(f"{'method':<16} {'AUC across tiles':>17} {'AUC within a tile':>18} {'best F1':>9}")
    print("-" * 64)

    for method in methods:
        table = np.array(scores[method])
        flat, truth = table[:, rankable].ravel(), labels[:, rankable].ravel()
        within = [auc(table[row, rankable][labels[row, rankable] == 1],
                      table[row, rankable][labels[row, rankable] == 0])
                  for row in range(len(TRUTH))]
        within = [v for v in within if not np.isnan(v)]

        best = 0.0
        for threshold in np.unique(flat):
            chosen = flat >= threshold
            precision = truth[chosen].sum() / max(chosen.sum(), 1)
            recall = truth[chosen].sum() / truth.sum()
            if precision + recall > 0:
                best = max(best, 2 * precision * recall / (precision + recall))
        print(f"{method:<16} {auc(flat[truth == 1], flat[truth == 0]):>17.4f} "
              f"{np.mean(within):>18.4f} {best:>9.4f}")

    print(f"\n8 questions on one tile took {np.median(timings['noul']):.0f} ms. "
          f"{len(binaries)} took {np.median(timings['yes/no']) * len(BINARY_WORDS):.0f} ms.")


if __name__ == "__main__":
    main()

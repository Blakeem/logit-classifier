"""Does the wording of a boolean's two sides change the answer?

Banking77 gives free balanced labels. Half the rows are asked about their own intent,
which is true, and half about a random other intent, which is false. Every wording is
asked of the same row in one request, so they share the state and the batch.

    uv run python tests-AB/ab_noul_wording.py
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
from ab_env import auc, banking77, stratified

from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import parse_request

WORDINGS = {
    "yes / no": ("yes", "no"),
    "true / false": ("true", "false"),
    "correct / incorrect": ("correct", "incorrect"),
    "present / absent": ("present", "absent"),
    "agree / disagree": ("agree", "disagree"),
}
# The same pair with the sides swapped, to tell word bias from label position bias.
SWAPPED = ("no", "yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = banking77()
    picked = stratified(data["label"], args.per_label, args.seed)
    classifier = Classifier(Config(use_prior_debias=False, abstain=False))
    print(f"{len(picked)} rows, {len(WORDINGS) + 2} arms\n", flush=True)

    rows, started = [], time.perf_counter()
    for position, index in enumerate(picked, 1):
        gold = data["names"][data["label"][index]]
        truth = position % 2 == 0
        subject = gold if truth else random.Random(index).choice(
            [n for n in data["names"] if n != gold])
        statement = f"This message is about {subject}"

        questions = {key: {"type": "choice", "instructions": f"Statement: {statement}",
                           "criteria": {positive: None, negative: None}}
                     for key, (positive, negative) in WORDINGS.items()}
        questions["no / yes, sides swapped"] = {
            "type": "choice", "instructions": f"Statement: {statement}",
            "criteria": {SWAPPED[0]: None, SWAPPED[1]: None}}
        questions["shipped noul"] = {"type": "noul", "instructions": statement}

        answers = classifier.classify(parse_request(
            {"state": data["text"][index], "questions": questions}))[0].answers

        scores = {key: answers[key].probabilities[positive]
                  for key, (positive, _) in WORDINGS.items()}
        # Always read the probability of the true side, whichever position it sits in.
        scores["no / yes, sides swapped"] = answers["no / yes, sides swapped"].probabilities["yes"]
        scores["shipped noul"] = answers["shipped noul"].noul
        rows.append({"truth": truth, "scores": scores})
        if position % 150 == 0:
            print(f"  {position}/{len(picked)}  {time.perf_counter() - started:.0f}s", flush=True)

    print(f"\n{'wording':<26} {'mean if true':>12} {'mean if false':>13} {'AUC':>7} "
          f"{'acc at 0.5':>11}")
    print("-" * 74)
    for key in [*WORDINGS, "no / yes, sides swapped", "shipped noul"]:
        yes = np.array([r["scores"][key] for r in rows if r["truth"]])
        no = np.array([r["scores"][key] for r in rows if not r["truth"]])
        accuracy = (np.mean(yes > 0.5) + np.mean(no <= 0.5)) / 2
        print(f"{key:<26} {yes.mean():>12.4f} {no.mean():>13.4f} "
              f"{auc(yes, no):>7.4f} {accuracy:>11.4f}")


if __name__ == "__main__":
    main()

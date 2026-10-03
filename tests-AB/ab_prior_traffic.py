"""Does the running prior help or hurt a noul when the traffic is skewed?

In the noul layout `A` is always yes and `B` is always no, so the binary prior bucket learns
the yes rate of past requests rather than letter bias. This harness scores Banking77
statements ("This message is about <intent>", true for the row's own intent and false for a
random other one) as a stream of requests at several true rates. The prior-on arm runs each
stream in order through one classifier, the way the service learns from its traffic. The
prior-off arm scores every statement once, since its answers do not depend on order. Both
arms read the same statements, and the first PRIOR_MIN_OBSERVATIONS requests of each stream
are left out of the metrics, since the prior stays unused until then.

    uv run python tests-AB/ab_prior_traffic.py
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
from ab_env import auc, banking77, stratified

from logit_classifier.classifier import Classifier
from logit_classifier.config import PRIOR_MIN_OBSERVATIONS, Config
from logit_classifier.schema import parse_request

TRUE_RATES = (0.5, 0.8, 0.95, 0.2)
# The verification operating points FINDINGS ships, beside the plain decision point.
THRESHOLDS = (0.5, 0.9999, 0.99998)


def statement_pool(data: dict, picked: list[int]) -> list[dict]:
    """A true and a false statement for every picked row."""
    pool: list[dict] = []

    for index in picked:
        gold = data["names"][data["label"][index]]
        other = random.Random(index).choice([name for name in data["names"] if name != gold])
        pool.append({"state": data["text"][index], "statement": f"This message is about {gold}", "truth": True})
        pool.append({"state": data["text"][index], "statement": f"This message is about {other}", "truth": False})
    return pool


def stream(pool: list[dict], true_rate: float, length: int, seed: int) -> list[int]:
    """Pool indices for one request stream at `true_rate`, drawn without replacement per class."""
    rng = random.Random(seed)
    trues = [i for i, item in enumerate(pool) if item["truth"]]
    falses = [i for i, item in enumerate(pool) if not item["truth"]]
    rng.shuffle(trues)
    rng.shuffle(falses)
    true_count = round(length * true_rate)
    order = trues[:true_count] + falses[:length - true_count]
    rng.shuffle(order)
    return order


def noul(classifier: Classifier, item: dict) -> tuple[float, float]:
    """The statement's noul and the request's wall time in seconds."""
    request = parse_request({"state": item["state"], "questions": {"q": {"type": "noul",
                                                                          "instructions": item["statement"]}}})
    started = time.perf_counter()
    answer = classifier.classify(request)[0].answers["q"]
    return float(answer.noul), time.perf_counter() - started


def metrics(scores: list[float], truths: list[bool]) -> dict:
    yes = np.array([s for s, t in zip(scores, truths, strict=True) if t])
    no = np.array([s for s, t in zip(scores, truths, strict=True) if not t])
    row: dict = {"true": len(yes), "false": len(no), "auc": auc(yes, no) if len(yes) and len(no) else float("nan")}

    for threshold in THRESHOLDS:
        row[f"recall@{threshold}"] = float(np.mean(yes >= threshold)) if len(yes) else float("nan")
        row[f"fpr@{threshold}"] = float(np.mean(no >= threshold)) if len(no) else float("nan")
    return row


def print_row(label: str, row: dict) -> None:
    cells = " ".join(f"{row[f'recall@{t}']:>8.3f} {row[f'fpr@{t}']:>7.3f}" for t in THRESHOLDS)
    print(f"{label:<22} {row['true']:>5} {row['false']:>5} {row['auc']:>7.4f} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--length", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = banking77()
    picked = stratified(data["label"], args.per_label, args.seed)
    pool = statement_pool(data, picked)
    off = Classifier(Config(use_prior_debias=False, abstain=False))
    off_scores: list[float] = []
    off_seconds: list[float] = []
    on_seconds: list[float] = []
    results: list[tuple[float, dict, dict]] = []

    print(f"{len(pool)} statements, streams of {args.length}, the first {PRIOR_MIN_OBSERVATIONS} unscored\n",
          flush=True)
    for item in pool:
        score, seconds = noul(off, item)
        off_scores.append(score)
        off_seconds.append(seconds)
    for true_rate in TRUE_RATES:
        # One fresh classifier per stream, on the backend already loaded, so each stream starts from no prior.
        on = Classifier(Config(use_prior_debias=True, calibration_path=None, abstain=False), off.backend)
        order = stream(pool, true_rate, args.length, args.seed)
        on_scores: list[float] = []
        for index in order:
            score, seconds = noul(on, pool[index])
            on_scores.append(score)
            on_seconds.append(seconds)
        scored = order[PRIOR_MIN_OBSERVATIONS:]
        truths = [pool[i]["truth"] for i in scored]
        results.append((true_rate,
                        metrics([off_scores[i] for i in scored], truths),
                        metrics(on_scores[PRIOR_MIN_OBSERVATIONS:], truths)))
        print(f"  stream at {true_rate:.0%} true done, prior {on.priors.stats()}", flush=True)

    header = " ".join(f"{'rec@' + str(t):>8} {'fpr':>7}" for t in THRESHOLDS)
    print(f"\n{'arm':<22} {'true':>5} {'false':>5} {'AUC':>7} {header}")
    print("-" * (42 + 17 * len(THRESHOLDS)))
    for true_rate, off_row, on_row in results:
        print_row(f"{true_rate:.0%} true, prior off", off_row)
        print_row(f"{true_rate:.0%} true, prior on", on_row)
    print(f"\nmean request: prior off {1000 * np.mean(off_seconds):.1f} ms, "
          f"prior on {1000 * np.mean(on_seconds):.1f} ms")


if __name__ == "__main__":
    main()

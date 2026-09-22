"""Fit a model's temperature on held out Banking77 rows.

Both shipped temperatures came from here. Logits are collected once and cached, then
every temperature is swept offline, so a re-sweep costs no GPU time.

    uv run python tests-AB/ab_temperature.py
    uv run python tests-AB/ab_temperature.py --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from ab_env import CACHE, auc, banking77, expected_calibration_error, split_by_label, stratified

from logit_classifier.classifier import Classifier
from logit_classifier.config import ANSWER_PREFILL, Config
from logit_classifier.prompt import SYSTEM_PROMPT, branch_content, build_branches, prefix_content
from logit_classifier.schema import parse_request
from logit_classifier.scoring import choice_confidence, combine_escape, restricted_softmax

INSTRUCTIONS = "Which banking support intent does this message express"
GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]


def collect(model: str, per_label: int, seed: int) -> dict:
    """Raw label logits per row, plus the option layout needed to rebuild a distribution."""
    data = banking77()
    criteria = dict.fromkeys(data["names"])
    classifier = Classifier(Config(model_id=model, use_prior_debias=False, permutations=1))
    picked = stratified(data["label"], per_label, seed)
    print(f"collecting {len(picked)} rows on {model}", flush=True)

    probe = parse_request(
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": INSTRUCTIONS,
                                           "criteria": criteria}}})
    shape = build_branches(probe.questions, "joint", classifier.config.abstain)
    layout = {"names": list(criteria), "groups": [list(b.targets) for b in shape]}

    records, started = [], time.perf_counter()
    for position, index in enumerate(picked, 1):
        request = parse_request(
            {"state": data["text"][index],
             "questions": {"q": {"type": "choice", "instructions": INSTRUCTIONS,
                                 "criteria": criteria}}})
        branches = build_branches(request.questions, "joint", classifier.config.abstain)
        prefix_text = classifier.backend.render(
            SYSTEM_PROMPT, prefix_content(request.state), ANSWER_PREFILL, open_ended=True)
        prefix_ids, vision = classifier.backend.encode_prefix(prefix_text, None)
        suffixes = [classifier.backend.encode(
            classifier.backend.render(SYSTEM_PROMPT, branch_content(request.state, b),
                                      ANSWER_PREFILL)[len(prefix_text):])
            for b in branches]
        scored = classifier.backend.score(prefix_ids, suffixes,
                                          [b.label_count for b in branches], vision)
        records.append({"gold": data["names"][data["label"][index]],
                        "logits": [s.z.tolist() for s in scored]})
        if position % 200 == 0:
            print(f"  {position}/{len(picked)}  {time.perf_counter() - started:.0f}s", flush=True)
    return {"layout": layout, "records": records, "model": model}


def measure(records: list[dict], layout: dict, temperature: float) -> dict:
    correct, confidences, abstains, nll = [], [], [], 0.0

    for record in records:
        parts = [restricted_softmax(np.array(z), None, temperature) for z in record["logits"]]
        flat, declined = combine_escape(parts)
        probabilities, cursor = {}, 0
        for group in layout["groups"]:
            for target in group:
                probabilities[layout["names"][target]] = float(flat[cursor])
                cursor += 1
        winner = max(probabilities, key=probabilities.get)
        correct.append(winner == record["gold"])
        confidences.append(choice_confidence(np.array(list(probabilities.values()))))
        abstains.append(declined)
        nll -= np.log(max(probabilities.get(record["gold"], 1e-12), 1e-12))
    return {"acc": sum(correct) / len(correct),
            "ece": expected_calibration_error(correct, confidences),
            "conf": float(np.mean(confidences)),
            "nll": nll / len(records),
            "abstain_auc": auc(np.array([a for a, c in zip(abstains, correct, strict=True) if not c]),
                               np.array([a for a, c in zip(abstains, correct, strict=True) if c]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Config().model_id)
    parser.add_argument("--per-label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recollect", action="store_true")
    args = parser.parse_args()

    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"temperature-{args.model.replace('/', '-')}-{args.per_label}-{args.seed}.json"
    if args.recollect or not cached.exists():
        cached.write_text(json.dumps(collect(args.model, args.per_label, args.seed)),
                          encoding="utf-8")
    blob = json.loads(cached.read_text(encoding="utf-8"))
    fit, test = split_by_label(blob["records"])
    print(f"\n{blob['model']}  fit {len(fit)} rows, test {len(test)} rows")
    print(f"{'T':>6} | {'fit NLL':>9} {'fit ECE':>8} | {'test acc':>9} {'test ECE':>9} {'conf':>7}")
    print("-" * 60)

    best_nll = best_ece = None
    for temperature in GRID:
        fitted, held = measure(fit, blob["layout"], temperature), measure(test, blob["layout"], temperature)
        print(f"{temperature:>6.2f} | {fitted['nll']:>9.4f} {fitted['ece']:>8.4f} | "
              f"{held['acc']:>9.4f} {held['ece']:>9.4f} {held['conf']:>7.4f}")
        if best_nll is None or fitted["nll"] < best_nll[0]:
            best_nll = (fitted["nll"], temperature, held)
        if best_ece is None or fitted["ece"] < best_ece[0]:
            best_ece = (fitted["ece"], temperature, held)

    print(f"\nfit on NLL -> T={best_nll[1]}  test acc {best_nll[2]['acc']:.4f}  "
          f"ECE {best_nll[2]['ece']:.4f}")
    print(f"fit on ECE -> T={best_ece[1]}  test acc {best_ece[2]['acc']:.4f}  "
          f"ECE {best_ece[2]['ece']:.4f}   <- what config.py ships")
    print("\nAccuracy climbing with temperature here is the split path artefact, not a gain.")


if __name__ == "__main__":
    main()

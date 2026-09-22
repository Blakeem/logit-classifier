"""Warm-model latency and throughput, measured on the real backend.

Two shapes matter for bulk tagging. The state is encoded once per record, and every
question against that record adds one branch. So the cost per tag falls as the number of
questions per record rises.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from ab_env import FIXTURES  # noqa: F401  sets sys.path

from logit_classifier.backends.hf import HFBackend
from logit_classifier.config import ANSWER_PREFILL, Config
from logit_classifier.prompt import SYSTEM_PROMPT, branch_content, build_branches, prefix_content
from logit_classifier.schema import SystemOneRequest, parse_request

FILLER = (
    "The customer contacted support about a recurring billing problem on their account. "
    "They described the steps they had already taken and attached a screenshot of the error. "
    "The agent confirmed the issue was reproducible and escalated it to engineering. "
)

TOPICS = [
    ("department", ["billing", "technical", "sales", "account"]),
    ("sentiment", ["positive", "neutral", "negative"]),
    ("urgency", ["low", "medium", "high"]),
    ("channel", ["email", "chat", "phone"]),
    ("language", ["english", "german", "french", "spanish"]),
    ("topic", ["refund", "bug", "pricing", "access", "other"]),
    ("resolved", ["yes", "no"]),
    ("escalate", ["yes", "no"]),
]


def make_state(backend: HFBackend, target_tokens: int) -> str:
    text = FILLER
    while len(backend.encode(text)) < target_tokens:
        text += FILLER
    ids = backend.encode(text)[:target_tokens]
    return backend.tokenizer.decode(ids)


def make_request(state: str, question_count: int) -> SystemOneRequest:
    questions = {}
    for index in range(question_count):
        name, options = TOPICS[index % len(TOPICS)]
        questions[f"{name}_{index}"] = {
            "type": "choice",
            "instructions": f"Classify the {name} of this record",
            "criteria": dict.fromkeys(options),
        }
    return parse_request({"state": state, "questions": questions})


def time_once(backend: HFBackend, request: SystemOneRequest) -> tuple[float, int, int]:
    branches = build_branches(request.questions)
    prefix_text = backend.render(SYSTEM_PROMPT, prefix_content(request.state), ANSWER_PREFILL,
                                open_ended=True)

    torch.cuda.synchronize()
    started = time.perf_counter()
    prefix_ids = backend.encode(prefix_text)
    suffixes = [
        backend.encode(backend.render(SYSTEM_PROMPT, branch_content(request.state, b),
                                    ANSWER_PREFILL)[len(prefix_text):])
        for b in branches
    ]
    backend.score(prefix_ids, suffixes, [b.label_count for b in branches])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0, len(prefix_ids), sum(len(s) for s in suffixes)


def measure(backend: HFBackend, state: str, question_count: int, repeats: int) -> dict:
    request = make_request(state, question_count)
    time_once(backend, request)
    samples = [time_once(backend, request) for _ in range(repeats)]
    times = [s[0] for s in samples]
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "prefix_tokens": samples[0][1],
        "suffix_tokens": samples[0][2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    backend = HFBackend(Config())
    parameters = sum(p.numel() for p in backend.model.parameters())
    print(f"model {backend.config.model_id}  {parameters / 1e9:.2f}B params  {backend.model.dtype}")
    print(f"device {torch.cuda.get_device_name(0)}\n")

    short_state = make_state(backend, 150)

    print("A. Questions per record, state fixed at ~150 tokens")
    header = f"{'questions':>9} | {'total ms':>8} | {'ms/tag':>7} | {'tags/sec':>8} | {'records/hr':>10}"
    print(header)
    print("-" * len(header))
    for count in [1, 2, 4, 8, 16, 32]:
        result = measure(backend, short_state, count, args.repeats)
        total = result["median_ms"]
        print(f"{count:>9} | {total:>8.1f} | {total / count:>7.1f} | "
              f"{count / (total / 1000):>8.1f} | {3600 / (total / 1000):>10,.0f}")

    print("\nB. State length, 8 questions per record")
    header = f"{'state tok':>9} | {'total ms':>8} | {'ms/tag':>7} | {'tags/sec':>8} | {'records/hr':>10}"
    print(header)
    print("-" * len(header))
    for tokens in [128, 512, 2048, 8192, 32768]:
        state = make_state(backend, tokens)
        result = measure(backend, state, 8, max(3, args.repeats // 2))
        total = result["median_ms"]
        prefill = result["prefix_tokens"] + result["suffix_tokens"]
        flops = 2 * parameters * prefill
        print(f"{result['prefix_tokens']:>9} | {total:>8.1f} | {total / 8:>7.1f} | "
              f"{8 / (total / 1000):>8.1f} | {3600 / (total / 1000):>10,.0f}"
              f"   [{prefill:>6} tok prefilled, {flops / (total / 1000) / 1e12:>5.1f} TFLOP/s]")

    print("\nC. Single question, single record, the worst case for amortising")
    result = measure(backend, short_state, 1, args.repeats)
    print(f"   median {result['median_ms']:.1f} ms, floor {result['min_ms']:.1f} ms")


if __name__ == "__main__":
    main()

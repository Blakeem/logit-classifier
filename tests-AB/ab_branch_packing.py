"""Measure packing branches by suffix length against taking them in request order.

Every row in a chunk is left-padded to that chunk's longest suffix, and each padded
token costs a full forward plus attention over the whole prefix. Request order drags
short branches to the longest width. This measures what sorting by length buys, what
it costs in low bits, and where the chunk token ceiling belongs.

    uv run python tests-AB/ab_branch_packing.py
    uv run python tests-AB/ab_branch_packing.py --ceiling-sweep
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from logit_classifier.backends import hf
from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import parse_request


def choice(count: int, instructions: str = "Pick the closest match") -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": {f"option {i}": f"the {i}th kind of thing" for i in range(count)}}


def noul(index: int) -> dict:
    return {"type": "noul", "instructions": f"Statement {index} is true of this state"}


def score(levels: int = 5) -> dict:
    return {"type": "score", "instructions": "How strongly this applies",
            "criteria": [f"level {i}" for i in range(levels)]}


SHORT_STATE = "The integration keeps failing and I am losing sales."
LONG_STATE = SHORT_STATE + " " + ("Here is more detail about the problem. " * 90)
# A low ceiling makes more chunks, and every chunk copies the prefix cache, so the
# cost of a low ceiling shows up behind a long prefix rather than a short one.
HUGE_STATE = SHORT_STATE + " " + ("Here is a great deal more context to carry. " * 700)

# Each shape names what it is meant to expose.
SHAPES = {
    "mixed, four choices plus nouls plus scores": {
        "state": LONG_STATE,
        "questions": {
            **{f"c{n}": choice(n) for n in (3, 8, 20, 45)},
            **{f"n{i}": noul(i) for i in range(12)},
            **{f"s{i}": score() for i in range(4)},
        },
    },
    "skewed, one wide choice beside twenty nouls": {
        "state": LONG_STATE,
        "questions": {"wide": choice(45), **{f"n{i}": noul(i) for i in range(20)}},
    },
    "uniform, 256 nouls at the Jev cap": {
        "state": SHORT_STATE,
        "questions": {f"n{i}": noul(i) for i in range(256)},
    },
    "split, one 77 option choice": {
        "state": SHORT_STATE,
        "questions": {"intent": choice(77, "Which banking support intent does this express")},
    },
    "tile fragments, 24 nouls": {
        "state": SHORT_STATE,
        "questions": {f"f{i}": noul(i) for i in range(24)},
    },
    # A uniform request whose branches are wider than the ceiling divided by the row
    # cap. Nothing here needs packing, so the answer must not move.
    "uniform, 30 three option choices": {
        "state": SHORT_STATE,
        "questions": {f"c{i}": choice(3) for i in range(30)},
    },
    "uniform, 12 twenty option choices": {
        "state": SHORT_STATE,
        "questions": {f"c{i}": choice(20) for i in range(12)},
    },
}


def request_order_chunks(self, suffixes, prefix_len):
    """The grouping this backend used before packing landed."""
    rows = self._rows_per_chunk(prefix_len, max(len(s) for s in suffixes))
    return [list(range(start, min(start + rows, len(suffixes))))
            for start in range(0, len(suffixes), rows)]


class Run:
    """One classify call, with its logits, its wall clock, its VRAM and its tokens."""

    def __init__(self, classifier, request):
        backend = classifier.backend
        self.tokens = 0
        self.chunks = 0
        original = backend._score_chunk

        def counted(prefix_ids, seed, suffixes, label_counts, rope_delta=None):
            self.tokens += len(suffixes) * max(len(s) for s in suffixes)
            self.chunks += 1
            return original(prefix_ids, seed, suffixes, label_counts, rope_delta)

        backend._score_chunk = counted
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            response, _ = classifier.classify(request)
            torch.cuda.synchronize()
        finally:
            backend._score_chunk = original
        self.ms = (time.perf_counter() - started) * 1000
        self.peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        self.answers = response.to_dict()["answers"]


def best_of(classifier, captured, request, grouping, rounds: int = 3):
    """Fastest of several runs under one grouping, with that run's logits.

    The first call on a shape pays CUDA warm-up, which alone reads as a 2x win and
    has nothing to do with packing, so the first round is discarded.
    """
    type(classifier.backend)._pack_chunks = grouping
    best = None
    rows = np.empty(0)

    for attempt in range(rounds):
        captured.clear()
        run = Run(classifier, request)
        if attempt and (best is None or run.ms < best.ms):
            best = run
            rows = np.concatenate([r.ravel() for r in captured])
    return best, rows


def capture_logits(classifier):
    """Wrap score so every raw logit row is kept for a bitwise comparison."""
    captured: list[np.ndarray] = []
    original = classifier.backend.score

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.extend(np.asarray(row.z, dtype=np.float64) for row in out)
        return out

    classifier.backend.score = wrapped
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config().model_id)
    parser.add_argument("--ceiling-sweep", action="store_true")
    parser.add_argument("--ceiling", type=int, default=None,
                        help="override CHUNK_TOKEN_CEILING for the shape table")
    args = parser.parse_args()

    shipped_ceiling = hf.CHUNK_TOKEN_CEILING
    if args.ceiling is not None:
        hf.CHUNK_TOKEN_CEILING = args.ceiling
        print(f"chunk token ceiling: {args.ceiling}")

    config = Config(model_id=args.model, use_prior_debias=False, permutations=1)
    print(f"loading {config.model_id}", flush=True)
    classifier = Classifier(config)
    captured = capture_logits(classifier)
    packed_chunks = type(classifier.backend)._pack_chunks

    print()
    header = f"{'shape':44} {'ms':>9} {'peak GB':>8} {'tokens':>8} {'chunks':>7}"
    print(header)
    print("-" * len(header))

    identical = 0
    for name, body in SHAPES.items():
        request = parse_request(body)

        before, before_z = best_of(classifier, captured, request, request_order_chunks)
        after, after_z = best_of(classifier, captured, request, packed_chunks)

        same = bool(np.array_equal(before_z, after_z))
        identical += same
        drift = 0.0 if same else float(np.max(np.abs(before_z - after_z)))
        speed = before.ms / after.ms if after.ms else 0.0

        print(f"{name[:44]:44} {before.ms:9.1f} {before.peak_gb:8.2f} {before.tokens:8d} "
              f"{before.chunks:7d}  order")
        print(f"{'':44} {after.ms:9.1f} {after.peak_gb:8.2f} {after.tokens:8d} "
              f"{after.chunks:7d}  packed")
        print(f"{'':44} {speed:8.2f}x {before.peak_gb - after.peak_gb:7.2f} "
              f"{before.tokens - after.tokens:8d}  saved, "
              f"{'bitwise identical' if same else f'max drift {drift:.4g}'}")
        print()

    print(f"{identical} of {len(SHAPES)} shapes came back bitwise identical")

    if args.ceiling_sweep:
        sweeps = {
            "mixed shape, roughly 800 token prefix":
                SHAPES["mixed, four choices plus nouls plus scores"],
            # Uniform widths never reach the ceiling, so a sweep over them proves
            # nothing. Mixed widths behind a long prefix is where an extra chunk is
            # most expensive, since every chunk copies the whole prefix cache.
            "long prefix, mixed widths, where an extra chunk costs most":
                {"state": HUGE_STATE,
                 "questions": {"wide": choice(45), "mid": choice(8),
                               **{f"n{i}": noul(i) for i in range(20)}}},
        }
        for title, body in sweeps.items():
            print()
            print(f"chunk token ceiling sweep, {title}:")
            request = parse_request(body)
            for ceiling in (512, 1024, 2048, 4096, 8192, 1 << 30):
                hf.CHUNK_TOKEN_CEILING = ceiling
                run, _ = best_of(classifier, captured, request, packed_chunks)
                label = "none" if ceiling > 1 << 20 else str(ceiling)
                print(f"  ceiling {label:>6}  {run.ms:8.1f} ms  {run.peak_gb:5.2f} GB  "
                      f"{run.tokens:6d} tokens  {run.chunks:3d} chunks")
        hf.CHUNK_TOKEN_CEILING = shipped_ceiling
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

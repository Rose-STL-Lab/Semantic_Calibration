#!/usr/bin/env python
"""The batched evaluation path must reproduce the one-question path exactly.

    python tests/test_question_batch.py
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import load_config  # noqa: E402
from csr.judge import Judge  # noqa: E402
from csr.semantic import semantic_confidence, semantic_confidence_batch  # noqa: E402

FAILURES = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILURES.append(name)


class HashJudge(Judge):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_retries = 1
        self.calls = 0

    # verdict depends only on the rendered prompt (template, question, A, B), as a
    # temperature-0 judge does; h % 11 forces a fixed share of unparsables
    def _call_once(self, messages):
        self.calls += 1
        h = int(hashlib.md5(messages[-1]["content"].encode()).hexdigest(), 16)
        if h % 11 == 0:
            return "???"
        return "Correct: YES" if h % 3 == 0 else "Correct: NO"


def make_questions(rng, n_questions):
    pool = ["Paris", "paris", "Lyon", "Bob Day", "Lucy Gichuhi", "1996", "the Sahara Hotel", ""]
    out = []
    for q in range(n_questions):
        k = rng.choice([0, 1, 2, 8, 8, 8, 16])
        answers = [rng.choice(pool) for _ in range(k)]
        gold = rng.choice([rng.choice(pool[:7]), [rng.choice(pool[:7]), rng.choice(pool[:7])]])
        out.append((f"Question {q}: which one?", answers, gold))
    return out


def main():
    rng = random.Random(0)
    for dataset in ("hotpot", "trivia", "nq_open"):
        cfg = load_config("CSR", "qwen", dataset)
        kind = cfg["dataset"].get("correctness_prompt", "judge")
        qs = make_questions(rng, 40)
        print(f"\n[{dataset}] correctness_prompt={kind}, {len(qs)} questions")

        # a fresh judge per path, so neither reuses the other's cache
        seq_judge, bat_judge = HashJudge(cfg), HashJudge(cfg)
        seq_clusters = [semantic_confidence(a, q, seq_judge) for q, a, _ in qs]
        seq_labels = [seq_judge.score_against_gold(a, g, q, kind) for q, a, g in qs]

        bat_clusters = semantic_confidence_batch([a for _, a, _ in qs], [q for q, _, _ in qs], bat_judge)
        bat_labels = bat_judge.score_against_gold_batch([a for _, a, _ in qs], [g for _, _, g in qs],
                                                        [q for q, _, _ in qs], kind)

        check("cluster ids identical for every question", seq_clusters == bat_clusters)
        check("correctness labels identical for every rollout", seq_labels == bat_labels)
        check("unparsable verdicts actually occurred",
              seq_judge.same_stats["unparsed"] + seq_judge.correctness_stats["unparsed"] > 0)
        check("same number of unparsed verdicts on both paths",
              (seq_judge.same_stats["unparsed"], seq_judge.correctness_stats["unparsed"])
              == (bat_judge.same_stats["unparsed"], bat_judge.correctness_stats["unparsed"]))
        check("K=0 and K=1 questions present", any(len(a) == 0 for _, a, _ in qs)
              and any(len(a) == 1 for _, a, _ in qs))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

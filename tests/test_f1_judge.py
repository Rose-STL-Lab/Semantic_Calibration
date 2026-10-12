#!/usr/bin/env python
"""Token-F1 equivalence (csr/f1_judge.py); needs no GPU, API key or judge server.

    python tests/test_f1_judge.py
"""

from __future__ import annotations

import os
import re
import string
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import load_config  # noqa: E402
from csr.f1_judge import (  # noqa: E402
    F1EquivalenceJudge, f1_score, f1_score_aliases, normalize_answer,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


# the official TriviaQA scorer (same rule as rd_util/EvaluationMetrics.py),
# transcribed here so the test does not import the module it is checking
def ref_normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    return white_space_fix(
        remove_articles(handle_punc(str(s or "").lower().replace("_", " ")))).strip()


def ref_f1(prediction, ground_truth):
    p = ref_normalize(prediction).split()
    g = ref_normalize(ground_truth).split()
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(p)
    recall = 1.0 * num_same / len(g)
    return (2 * precision * recall) / (precision + recall)


SAMPLES = [
    ("Paris", "Paris"),
    ("paris", "  PARIS  "),
    ("The Eiffel Tower", "Eiffel Tower"),
    ("New_York", "new york"),
    ("Bob Day", "Lucy Gichuhi"),
    ("a dog", "the dog"),
    ("don't know", "dont know"),
    ("1999", "the year 1999"),
    ("", "something"),
    ("something", ""),
    ("George Walker Bush", "George Bush"),
    ("carbon dioxide (CO2)", "carbon dioxide"),
    ("It's about 12 miles", "12 miles"),
]


def test_matches_triviaqa_reference():
    print("\n[F1 agrees with the official TriviaQA implementation]")
    worst, worst_case = 0.0, None
    for a, b in SAMPLES:
        got, want = f1_score(a, b), ref_f1(a, b)
        if abs(got - want) > worst:
            worst, worst_case = abs(got - want), (a, b, got, want)
    check(f"F1 matches on all {len(SAMPLES)} sample pairs",
          worst < 1e-12, f"worst {worst:.3e} at {worst_case}")

    norm_ok = all(normalize_answer(a) == ref_normalize(a) for a, _ in SAMPLES)
    check("normalize_answer matches the reference on every sample", norm_ok)

    # The vendored normalizer lower-cases and strips punctuation, articles and
    # underscores -- the properties the F1 depends on.
    check("underscores become spaces", normalize_answer("New_York") == "new york")
    check("articles are dropped", normalize_answer("the dog") == "dog")
    check("punctuation is dropped", normalize_answer("don't!") == "don t")


def test_f1_values():
    print("\n[F1 values]")
    check("identical strings score 1.0", abs(f1_score("Paris", "paris") - 1.0) < 1e-12)
    check("disjoint strings score 0.0", f1_score("Bob Day", "Lucy Gichuhi") == 0.0)
    check("empty prediction scores 0.0", f1_score("", "anything") == 0.0)
    # "eiffel tower" vs "the eiffel tower" -> articles removed, so identical
    check("articles do not cost F1",
          abs(f1_score("The Eiffel Tower", "Eiffel Tower") - 1.0) < 1e-12)
    # 1 shared token of 3 predicted and 2 gold: P=1/3, R=1/2, F1=0.4
    check("partial overlap is the harmonic mean",
          abs(f1_score("big red dog", "red cat") - 0.4) < 1e-9,
          f"{f1_score('big red dog', 'red cat')}")
    check("F1 is symmetric",
          all(abs(f1_score(a, b) - f1_score(b, a)) < 1e-12 for a, b in SAMPLES))
    check("aliases take the maximum",
          abs(f1_score_aliases("Paris", ["London", "Paris", "Rome"]) - 1.0) < 1e-12)
    check("empty alias list scores 0.0", f1_score_aliases("Paris", []) == 0.0)
    check("None prediction scores 0.0", f1_score_aliases(None, ["Paris"]) == 0.0)


















def make_equivalence_judge(threshold=0.5):
    """The CSR-F1 configuration: token F1 for rollout-rollout equivalence only.

    The ``llm`` slot is a bare object: reclustering touches nothing but
    ``symmetric_batched``, which is exactly what scripts/f1_judge_eval.py relies on.
    """
    return F1EquivalenceJudge(object(), {"judge": {"f1_threshold": threshold}},
                              chat_end_markers=())


def test_equivalence_threshold_is_inclusive():
    print("\n[F1EquivalenceJudge threshold]")
    # "red cat" vs "red dog": 1 shared token out of 2 and 2 -> F1 exactly 0.5
    exact_half = f1_score("red cat", "red dog")
    check("this pair sits exactly on the threshold", abs(exact_half - 0.5) < 1e-12,
          str(exact_half))
    j = make_equivalence_judge(0.5)
    labels, unparsed, missing = j.symmetric_batched(["red cat"], ["red dog"], ["q"])
    # F1Judge uses `>` like RD; F1EquivalenceJudge uses `>=`, as the CSR-F1 scripts do.
    check("F1 == threshold counts as EQUIVALENT (`>=`)", labels == [1], str(labels))
    check("equivalence verdicts are never unparsable", unparsed == 0 and missing == [])

    j_high = make_equivalence_judge(0.6)
    check("raising the threshold flips it",
          j_high.symmetric_batched(["red cat"], ["red dog"], ["q"])[0] == [0])


def test_equivalence_is_symmetric():
    print("\n[F1EquivalenceJudge symmetry]")
    j = make_equivalence_judge()
    a, b = "The Eiffel Tower", "Eiffel Tower"
    fwd = j.symmetric_batched([a], [b], ["q"])[0]
    rev = j.symmetric_batched([b], [a], ["q"])[0]
    check("equivalence is symmetric by construction", fwd == rev == [1],
          f"{fwd} vs {rev}")


def test_equivalence_stats_and_delegation():
    print("\n[F1EquivalenceJudge stats and delegation]")
    j = make_equivalence_judge()
    j.symmetric_batched(["Paris", "paris"], ["Paris", "Lyon"], ["q", "q"])
    stats = j.f1_same_stats
    check("f1_same_stats counts every pair", stats["total"] == 2, str(stats))
    check("one pair matched, one did not",
          stats["parsed_yes"] == 1 and stats["parsed_no"] == 1, str(stats))
    check("f1 equivalence never reports unparsed", stats["unparsed"] == 0, str(stats))
    # correctness is NOT replaced: it is delegated to the wrapped LLM judge.
    check("correctness is delegated, not overridden",
          "symmetric_batched" in vars(type(j))
          and "correctness_batched" not in vars(type(j)))


def main():
    print("=" * 70)
    print("token-F1 judge -- csr/f1_judge.py")
    print("=" * 70)

    test_matches_triviaqa_reference()
    test_f1_values()
    test_equivalence_threshold_is_inclusive()
    test_equivalence_is_symmetric()
    test_equivalence_stats_and_delegation()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

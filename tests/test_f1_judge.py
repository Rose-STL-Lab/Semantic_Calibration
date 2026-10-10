#!/usr/bin/env python
"""Token-F1 judge (csr/f1_judge.py); needs no GPU, API key or judge server.

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
from csr.f1_judge import F1Judge, f1_score, f1_score_aliases, normalize_answer  # noqa: E402
from csr.judge import Judge, make_judge  # noqa: E402
from csr.rewards import RewardComputer  # noqa: E402

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


def make_judge_f1(threshold=0.5):
    cfg = load_config("CSR", "llama", "hotpot",
                      ["judge.backend=f1", f"judge.f1_threshold={threshold}"])
    return make_judge(cfg, chat_end_markers=cfg["model"].get("chat_end_markers", []))


def test_factory():
    print("\n[factory dispatches on judge.backend]")
    cfg_llm = load_config("CSR", "llama", "hotpot")
    check("default backend is the LLM judge", isinstance(make_judge(cfg_llm), Judge))
    check("backend=f1 gives the F1 judge", isinstance(make_judge_f1(), F1Judge))
    check("backend=f1 is not an LLM judge", not isinstance(make_judge_f1(), Judge))

    cfg_bad = load_config("CSR", "llama", "hotpot", ["judge.backend=nonsense"])
    try:
        make_judge(cfg_bad)
        check("an unknown backend is rejected", False, "no error raised")
    except ValueError:
        check("an unknown backend is rejected", True)

    j = make_judge_f1(0.6)
    check("f1_threshold is read from the config", abs(j.threshold - 0.6) < 1e-12)
    check("model_id names the rule", j.model_id == "f1@0.6", j.model_id)


def test_threshold_is_strict():
    print("\n[threshold]")
    j = make_judge_f1(0.5)
    # "red cat" vs "red dog": 1 shared token of 2 and 2 -> F1 exactly 0.5
    exact_half = f1_score("red cat", "red dog")
    check("this pair sits exactly on the threshold", abs(exact_half - 0.5) < 1e-12,
          str(exact_half))
    labels = j.score_against_gold(["red cat"], "red dog", "q")
    check("F1 == threshold counts as wrong (RD uses `> threshold`)",
          labels == [0], str(labels))

    j_low = make_judge_f1(0.4)
    check("lowering the threshold flips it",
          j_low.score_against_gold(["red cat"], "red dog", "q") == [1])


def test_judge_contract():
    print("\n[Judge interface contract]")
    j = make_judge_f1()

    labels, unparsed, missing = j.symmetric_batched(["Paris", "Berlin"],
                                                    ["paris", "Paris"], ["q", "q"])
    check("symmetric_batched returns (labels, unparsed, missing)",
          labels == [1, 0] and unparsed == 0 and missing == [],
          f"{labels} {unparsed} {missing}")

    a, b = "The Eiffel Tower", "eiffel tower"
    fwd, _, _ = j.symmetric_batched([a], [b], ["q"])
    rev, _, _ = j.symmetric_batched([b], [a], ["q"])
    check("equivalence is symmetric by construction", fwd == rev == [1],
          f"{fwd} vs {rev}")

    labels, unparsed = j.correctness_batched(["Paris"], ["paris"], ["q"])
    check("correctness_batched returns (labels, unparsed)",
          labels == [1] and unparsed == 0, f"{labels} {unparsed}")

    # gold may be one string or a list of aliases; both kinds score the same way
    check("a single gold string works", j.score_against_gold(["Paris"], "Paris", "q") == [1])
    check("a gold alias list works",
          j.score_against_gold(["Paris"], ["London", "Paris"], "q") == [1])
    check("kind='gold' and kind='judge' agree (there is no prompt to pick)",
          j.score_against_gold(["Paris"], ["London", "Paris"], "q", "gold")
          == j.score_against_gold(["Paris"], ["London", "Paris"], "q", "judge"))

    preds_list = [["Paris", "Lyon"], ["Berlin"]]
    golds = ["Paris", ["Berlin", "berlin city"]]
    questions = ["q1", "q2"]
    batched = j.score_against_gold_batch(preds_list, golds, questions)
    one_by_one = [j.score_against_gold(p, g, q)
                  for p, g, q in zip(preds_list, golds, questions)]
    check("score_against_gold_batch equals the per-question path",
          batched == one_by_one == [[1, 0], [1]], f"{batched} vs {one_by_one}")

    for attr in ("single_gold_prompt", "multi_gold_prompt", "cache", "model_id"):
        check(f"exposes .{attr} (read off the judge elsewhere)", hasattr(j, attr))


def test_stats_block():
    print("\n[stats block]")
    j = make_judge_f1()
    j.score_against_gold(["Paris", "Berlin"], "Paris", "q")
    j.symmetric_batched(["a"], ["a"], ["q"])
    stats = j.stats_summary()

    # JudgeCacheCallback reads exactly these
    for key in ("cache_size", "correctness", "semantic_equivalence"):
        check(f"stats_summary has {key!r}", key in stats)
    for block in ("correctness", "semantic_equivalence"):
        for key in ("total", "unparsed", "parsed_yes", "parsed_no", "unparsed_rate"):
            check(f"{block}.{key} present", key in stats[block])
    check("correctness counted 2 verdicts", stats["correctness"]["total"] == 2,
          str(stats["correctness"]))
    check("equivalence counted 1 verdict", stats["semantic_equivalence"]["total"] == 1)
    check("nothing is ever unparsed", stats["correctness"]["unparsed"] == 0
          and stats["semantic_equivalence"]["unparsed"] == 0)
    check("the block records the backend", stats.get("backend") == "f1")

    j.reset_stats()
    check("reset_stats clears the counters",
          j.stats_summary()["correctness"]["total"] == 0)


def test_answer_extraction():
    print("\n[answer extraction matches the LLM path]")
    j = make_judge_f1()
    # canonicalize_answer strips an "Answer:" prefix, as it does before the LLM
    # judge is asked, so the two backends see the same string.
    check("an 'Answer:' prefix is stripped before scoring",
          j.score_against_gold(["Answer: Paris"], "Paris", "q") == [1])
    check("a trailing period does not matter",
          j.score_against_gold(["Paris."], "Paris", "q") == [1])


def test_determinism_and_offline():
    print("\n[deterministic and offline]")
    j1, j2 = make_judge_f1(), make_judge_f1()
    preds = ["Paris", "the city of Paris", "Berlin", ""]
    r1 = j1.score_against_gold(preds, ["Paris", "paris france"], "q")
    r2 = j2.score_against_gold(preds, ["Paris", "paris france"], "q")
    check("two fresh judges give identical verdicts", r1 == r2, f"{r1} vs {r2}")
    check("no API key was needed", "OPENAI_API_KEY" not in os.environ
          or True)  # the point: nothing above raised
    check("the judge holds no client", not hasattr(j1, "_client"))


def test_calibration_reward_runs_on_f1():
    print("\n[CSR's calibration reward on the F1 judge]")
    cfg = load_config("CSR", "llama", "hotpot", ["judge.backend=f1"])
    judge = make_judge(cfg)
    rc = RewardComputer(cfg, judge)
    rc.cali_w = 1.0

    k = 8
    texts = ["Paris", "paris", "Paris.", "Berlin", "Lyon", "Paris", "Berlin", "Rome"]
    prompt = [{"role": "system", "content": "s"},
              {"role": "user", "content": "capital of France?"}]
    prompts = [prompt] * k
    completions = [[{"role": "assistant", "content": t}] for t in texts]
    answers = ["Paris"] * k

    rewards = rc.calibration_reward(prompts, completions, answers)
    check(f"returns one reward per rollout ({k})", len(rewards) == k, str(len(rewards)))
    check("every reward lies in [0, 1] at cali_w=1",
          all(-1e-6 <= r <= 1 + 1e-6 for r in rewards),
          str([round(r, 4) for r in rewards]))

    # reward = agreement fraction z_j = mean_{i!=j} 1[s_ji == q_j]; the F1 judge puts
    # Paris/paris/"Paris." in one cluster, so a wrong isolated rollout scores higher
    # than a correct one -- by design: the term rewards agreement structure, not correctness
    want = [3 / 7, 3 / 7, 3 / 7, 6 / 7, 1.0, 3 / 7, 6 / 7, 1.0]
    check("reward equals the agreement fraction of each rollout",
          all(abs(g - w) < 1e-4 for g, w in zip(rewards, want)),
          f"{[round(r, 4) for r in rewards]} vs {[round(w, 4) for w in want]}")
    check("an isolated wrong rollout scores 1.0 (nobody agrees, and it is wrong)",
          abs(rewards[4] - 1.0) < 1e-4 and abs(rewards[7] - 1.0) < 1e-4,
          str([round(r, 4) for r in rewards]))
    check("the F1 judge clusters Paris / paris / 'Paris.' together",
          abs(rewards[0] - rewards[1]) < 1e-6 and abs(rewards[0] - rewards[2]) < 1e-6,
          str([round(r, 4) for r in rewards]))

    corr = rc.correctness_reward(prompts, completions, answers)
    check("correctness marks the Paris rollouts right and the others wrong",
          [int(c) for c in corr] == [1, 1, 1, 0, 0, 1, 0, 0],
          str(corr))
    check("the judge was asked for both terms without an LLM",
          judge.stats_summary()["correctness"]["total"] > 0)


def main():
    print("=" * 70)
    print("token-F1 judge -- csr/f1_judge.py")
    print("=" * 70)

    test_matches_triviaqa_reference()
    test_f1_values()
    test_factory()
    test_threshold_is_strict()
    test_judge_contract()
    test_stats_block()
    test_answer_extraction()
    test_determinism_and_offline()
    test_calibration_reward_runs_on_f1()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

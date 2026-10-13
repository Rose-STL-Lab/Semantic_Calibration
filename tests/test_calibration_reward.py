#!/usr/bin/env python
# CSR's calibration reward, driven by a stub judge -- no GPU and no API key.
# python tests/test_calibration_reward.py

from __future__ import annotations

import io
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from csr.config import load_config  # noqa: E402
from csr.judge import Judge  # noqa: E402
from csr.rewards import CE_MAX, CE_MIN, DELTA_CE, RewardComputer  # noqa: E402


def standardize(values):
    # GRPO's per-group advantage: (r - mean) / std, or None if degenerate
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var < 1e-18:
        return None
    sd = math.sqrt(var)
    return [(v - mean) / sd for v in values]

K = 8
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


class StubJudge:
    # answers pair queries from equivalent(j, i); records every call

    def __init__(self, correct, equivalent, missing_pairs=()):
        self.correct = correct              # list[int], one per rollout
        self.equivalent = equivalent        # fn(text_a, text_b) -> 0/1
        self.missing_pairs = set(missing_pairs)
        self.pair_calls: list[tuple[str, str]] = []
        self.correctness_calls = 0

    def score_against_gold(self, preds, gold, question, kind="judge"):
        self.correctness_calls += 1
        return list(self.correct)

    def symmetric_batched(self, a_list, b_list, questions):
        labels, missing = [], []
        for idx, (a, b) in enumerate(zip(a_list, b_list)):
            self.pair_calls.append((a, b))
            if idx in self.missing_pairs:
                labels.append(0)
                missing.append(idx)
            else:
                labels.append(int(self.equivalent(a, b)))
        return labels, len(missing), missing


def make_computer(judge, cali_w=1.0):
    cfg = load_config("CSR", "llama", "hotpot")
    rc = RewardComputer(cfg, judge)
    rc.cali_w = cali_w
    return rc


def make_batch(texts):
    prompt = [{"role": "system", "content": "s"},
              {"role": "user", "content": "capital of France?"}]
    prompts = [prompt] * len(texts)
    completions = [[{"role": "assistant", "content": t}] for t in texts]
    answers = ["Paris"] * len(texts)
    return prompts, completions, answers


def test_pair_enumeration():
    print("\n[pair enumeration]")
    texts = [f"answer {i}" for i in range(K)]
    judge = StubJudge([1] * K, lambda a, b: 0)
    rc = make_computer(judge)
    rc.calibration_reward(*make_batch(texts))

    n_pairs = len(judge.pair_calls)
    check(f"K={K} requests K(K-1)/2 = {K * (K - 1) // 2} pair judgments",
          n_pairs == K * (K - 1) // 2, f"got {n_pairs}")

    unordered = {frozenset(p) for p in judge.pair_calls}
    check("every unordered pair appears exactly once",
          len(unordered) == n_pairs == K * (K - 1) // 2,
          f"{len(unordered)} distinct of {n_pairs}")

    check("no self-pairs", all(a != b for a, b in judge.pair_calls))

    peers = {t: 0 for t in texts}
    for a, b in judge.pair_calls:
        peers[a] += 1
        peers[b] += 1
    check(f"each rollout has exactly K-1 = {K - 1} peer contributions",
          all(v == K - 1 for v in peers.values()), str(peers))


def test_permutation_equivariance():
    print("\n[permutation equivariance]")
    texts = ["A", "A", "B", "C", "A", "B", "D", "A"]
    eq = lambda a, b: int(a == b)  # noqa: E731
    correct = [1, 1, 0, 0, 1, 0, 0, 1]

    rc = make_computer(StubJudge(correct, eq))
    base = rc.calibration_reward(*make_batch(texts))

    order = [3, 0, 5, 7, 2, 6, 1, 4]
    permuted_texts = [texts[i] for i in order]
    permuted_correct = [correct[i] for i in order]
    rc2 = make_computer(StubJudge(permuted_correct, eq))
    permuted = rc2.calibration_reward(*make_batch(permuted_texts))

    expected = [base[i] for i in order]
    check("permuting rollouts only permutes the reward vector",
          all(abs(a - b) < 1e-6 for a, b in zip(permuted, expected)),
          f"{[round(x, 4) for x in permuted]} vs {[round(x, 4) for x in expected]}")


def reward_for_mismatches(m: int, correct: bool) -> float:
    # reward of rollout 0 when exactly m of its K-1 peers mismatch its label
    texts = [f"r{i}" for i in range(K)]

    if correct:
        # peers 1..K-1-m agree with rollout 0, the last m do not
        agreeing = set(range(1, K - m))
    else:
        # for an incorrect rollout a *mismatch* is a peer that agrees with it
        agreeing = set(range(1, 1 + m))

    def eq(a, b):
        ia, ib = texts.index(a), texts.index(b)
        if 0 in (ia, ib):
            other = ib if ia == 0 else ia
            return int(other in agreeing)
        return 0

    labels = [1 if correct else 0] + [0] * (K - 1)
    rc = make_computer(StubJudge(labels, eq))
    return rc.calibration_reward(*make_batch(texts))[0]


def test_reward_levels():
    print("\n[reward levels: r_cal = normalised mean pairwise BCE = 1 - m/(K-1)]")
    # a binary verdict makes each pair's BCE CE_MIN or CE_MAX, so mapping the mean by
    # the two endpoints gives exactly 1 - m/(K-1) -- no offset, no clamp
    expected = [(CE_MAX - (CE_MIN + (m / (K - 1)) * DELTA_CE)) / DELTA_CE for m in range(K)]

    for correct in (True, False):
        got = [reward_for_mismatches(m, correct) for m in range(K)]
        label = "correct" if correct else "incorrect"
        check(f"[{label}] reward equals 1 - m/(K-1) for m=0..{K - 1}",
              all(abs(g - e) < 1e-4 for g, e in zip(got, expected)),
              f"{[round(x, 4) for x in got]}")
        check(f"[{label}] all rewards finite",
              all(math.isfinite(x) for x in got))
        check(f"[{label}] range is exactly [0, 1]: m=0 -> 1, m={K-1} -> 0",
              abs(got[0] - 1.0) < 1e-4 and abs(got[K - 1]) < 1e-4,
              f"m=0 -> {got[0]:.6f}, m={K-1} -> {got[K-1]:.6f}")
        check(f"[{label}] {K} distinct levels",
              len({round(x, 6) for x in got}) == K)
        check(f"[{label}] strictly decreasing in mismatch count",
              all(got[i] > got[i + 1] for i in range(K - 1)),
              f"{[round(x, 4) for x in got]}")
        check(f"[{label}] no clamp(0,5) dead zone: every m>=3 is distinct",
              len({round(x, 6) for x in got[3:]}) == K - 3,
              f"{[round(x, 4) for x in got[3:]]}")
        check(f"[{label}] levels are evenly spaced by 1/{K - 1} = {1/(K-1):.4f}",
              all(abs((got[i] - got[i + 1]) - 1 / (K - 1)) < 1e-4
                  for i in range(K - 1)),
              f"{[round(got[i]-got[i+1], 4) for i in range(K-1)]}")

    print(f"    levels: {[round(x, 4) for x in expected]}")


def test_no_offset_no_clamp():
    print("\n[no offset / no clamp / exact affine normalisation]")
    src = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "csr", "rewards.py"), encoding="utf-8").read()
    body = src[src.index("def calibration_reward"):src.index("def correctness_reward")]
    for pattern, name in [("5.0 -", "a `5.0 -` offset"),
                          ("clamp(0.0, 5", "a clamp(0, 5)"),
                          ("clamp(0.0, 1.0)", "a clamp(0, 1)"),
                          (".clamp(", "a clamp on the reward"),
                          ("torch.clip", "a clip on the reward"),
                          ("relu", "a hinge")]:
        check(f"calibration_reward contains no {name}", pattern not in body,
              f"found {pattern!r}")
    check("it maps the mean BCE by its two endpoints: (CE_MAX - mean_ce) / DELTA_CE",
          "(CE_MAX - mean_ce) / DELTA_CE" in body)

    # the reward of a fully-matching rollout must be exactly 1, not 5 or ~0
    r = reward_for_mismatches(0, True)
    check("a fully matching rollout scores 1.0 (5.0 if offset, ~0 if raw -BCE)",
          abs(r - 1.0) < 1e-4, f"got {r:.9f}")
    r = reward_for_mismatches(K - 1, True)
    check("a fully mismatching rollout scores 0.0, never negative",
          abs(r) < 1e-4 and r >= -1e-7, f"got {r:.9f}")


def test_affine_equivalence_under_grpo():
    print("\n[affine equivalence: raw -BCE at lambda == normalised at DELTA_CE*lambda]")
    # r_raw = -CE_MAX + DELTA_CE * z is affine in z, so after standardising, raw -BCE at
    # lambda and the normalised reward at DELTA_CE * lambda agree elementwise
    rng = random.Random(0)
    worst = 0.0
    for _ in range(200):
        q = [rng.randint(0, 1) for _ in range(K)]
        z = [rng.randint(0, K - 1) / (K - 1) for _ in range(K)]
        lam = rng.uniform(0.02, 0.4)
        r_raw = [q[i] + lam * (-CE_MAX + DELTA_CE * z[i]) for i in range(K)]
        r_z = [q[i] + (DELTA_CE * lam) * z[i] for i in range(K)]
        a1, a2 = standardize(r_raw), standardize(r_z)
        if a1 is None:
            continue
        worst = max(worst, max(abs(x - y) for x, y in zip(a1, a2)))
    check("advantages agree elementwise over 200 random groups",
          worst < 1e-9, f"worst |diff| = {worst:.3e}")
    check(f"raw 0.05 is {DELTA_CE*0.05:.6f} on the [0,1] scale",
          abs(DELTA_CE * 0.05 - 0.690775) < 1e-5, f"DELTA_CE = {DELTA_CE:.6f}")
    check(f"raw 0.1 -> 0.2 is {DELTA_CE*0.1:.4f} -> {DELTA_CE*0.2:.4f}",
          abs(DELTA_CE * 0.1 - 1.3816) < 1e-3 and abs(DELTA_CE * 0.2 - 2.7631) < 1e-3)


def trl_advantage(rewards):
    # TRL 0.24's default scale_rewards='group': centre, divide by unbiased std + 1e-4
    r = torch.tensor(rewards, dtype=torch.float32)
    return ((r - r.mean()) / (r.std() + 1e-4)).tolist()


def test_scale_identity_through_trl():
    print("\n[scale identity: raw -BCE @0.05 == normalised @0.690775, via TRL's advantage]")
    # rebuild the pre-reparameterisation reward (-mean_ce) from the same verdicts and run
    # both through TRL's advantage: the optimisation target must be unchanged
    rng = random.Random(1234)
    lam_raw = 0.05
    lam_z = lam_raw * DELTA_CE
    pairs = [(j, i) for j in range(K) for i in range(j + 1, K)]
    worst, n_groups, n_mixed = 0.0, 0, 0
    for _ in range(300):
        cid = [rng.randrange(rng.randint(1, K)) for _ in range(K)]
        q = [rng.randint(0, 1) for _ in range(K)]
        missing = set(rng.sample(range(len(pairs)), rng.randint(0, 3)))
        texts = [f"r{i}" for i in range(K)]

        def eq(a, b, cid=cid, texts=texts):
            return int(cid[texts.index(a)] == cid[texts.index(b)])

        rc = make_computer(StubJudge(q, eq, missing_pairs=missing), cali_w=lam_z)
        new = rc.calibration_reward(*make_batch(texts))

        # reference: raw -mean_ce, the same pair evidence and the same fallback
        match, cnt = [0.0] * K, [0.0] * K
        for idx, (j, i) in enumerate(pairs):
            if idx in missing:
                continue
            s = eq(texts[j], texts[i])
            for owner in (j, i):
                cnt[owner] += 1
                match[owner] += float(s == q[owner])
        raw = [None] * K
        for j in range(K):
            if cnt[j] > 0:
                raw[j] = -(CE_MIN * match[j] + CE_MAX * (cnt[j] - match[j])) / cnt[j]
        have = [r for r in raw if r is not None]
        fb = sum(have) / len(have) if have else -CE_MAX / 2
        raw = [lam_raw * (r if r is not None else fb) for r in raw]

        a_new = trl_advantage([q[j] + new[j] for j in range(K)])
        a_raw = trl_advantage([q[j] + raw[j] for j in range(K)])
        worst = max(worst, max(abs(x - y) for x, y in zip(a_new, a_raw)))
        n_groups += 1
        n_mixed += int(0 < sum(q) < K)
    check(f"advantages identical over {n_groups} random groups ({n_mixed} mixed) under TRL's formula",
          worst < 1e-5, f"worst |diff| = {worst:.3e}")
    check("every normalised reward lies in [0, 1] before weighting",
          all(0.0 - 1e-6 <= v / lam_z <= 1.0 + 1e-6 for v in new), str(new))


def test_missing_verdicts_dropped():
    print("\n[missing verdicts]")
    texts = [f"r{i}" for i in range(K)]
    labels = [1] + [0] * (K - 1)          # rollout 0 is the correct one

    # rollout 0 disagrees with peers 1 and 2, agrees with 3..7  ->  m=2
    def eq(a, b):
        ia, ib = texts.index(a), texts.index(b)
        if 0 in (ia, ib):
            other = ib if ia == 0 else ia
            return int(other not in (1, 2))
        return 0

    rc = make_computer(StubJudge(labels, eq))
    full = rc.calibration_reward(*make_batch(texts))[0]

    # pair index 0 is (0, 1) -- one of rollout 0's two *mismatching* pairs
    rc2 = make_computer(StubJudge(labels, eq, missing_pairs={0}))
    dropped = rc2.calibration_reward(*make_batch(texts))[0]

    # kept: 2 mismatching peers of 7 -> 5/7.  dropped: 1 mismatching of 6 -> 5/6.
    want_full = (CE_MAX - (CE_MIN + (2 / 7) * DELTA_CE)) / DELTA_CE
    want_drop = (CE_MAX - (CE_MIN + (1 / 6) * DELTA_CE)) / DELTA_CE
    # counted silently as NO it would stay at 2/7
    check("an unreadable pair leaves the denominator, it is not counted as NO",
          abs(full - want_full) < 1e-3 and abs(dropped - want_drop) < 1e-3,
          f"full={full:.4f} (want {want_full:.4f})  dropped={dropped:.4f} (want {want_drop:.4f})")

    # every pair of rollout 0 unreadable -> no evidence -> group mean, never 0.0
    all_pairs = [(j, i) for j in range(K) for i in range(j + 1, K)]
    touching_zero = {idx for idx, pair in enumerate(all_pairs) if 0 in pair}
    rc3 = make_computer(StubJudge(labels, eq, missing_pairs=touching_zero))
    values = rc3.calibration_reward(*make_batch(texts))
    peer_mean = sum(values[1:]) / (K - 1)
    check("a rollout with no readable pair gets the group mean, not the worst score 0",
          abs(values[0] - peer_mean) < 1e-4 and values[0] > 1e-3,
          f"got {values[0]:.6f}, peer mean {peer_mean:.6f}")

    check("no reward is non-finite in any of the three cases",
          all(math.isfinite(x) for x in [full, dropped] + list(values)))


def test_single_rollout():
    print("\n[degenerate group]")
    rc = make_computer(StubJudge([1], lambda a, b: 1))
    reward = rc.calibration_reward(*make_batch(["only"]))
    # 0.5 is the midpoint of [0, 1]; 1.0 would be the *best* score
    check("K=1 falls back to the neutral midpoint 0.5 of [0, 1]",
          abs(reward[0] - 0.5) < 1e-4, str(reward))


def test_cali_w_applied_once():
    print("\n[cali_w]")
    texts = [f"r{i}" for i in range(K)]
    eq = lambda a, b: 1  # noqa: E731
    rc1 = make_computer(StubJudge([1] * K, eq), cali_w=1.0)
    rc2 = make_computer(StubJudge([1] * K, eq), cali_w=0.2)
    r1 = rc1.calibration_reward(*make_batch(texts))[0]
    r2 = rc2.calibration_reward(*make_batch(texts))[0]
    # rewards in [0, 1]; float32 output
    check("reward scales linearly with cali_w, applied exactly once",
          abs(r2 - 0.2 * r1) < 1e-4, f"{r2} vs {0.2 * r1}")


class RecordingJudge(Judge):
    # judge with the API replaced by a scripted responder

    def __init__(self, cfg, responses):
        super().__init__(cfg)
        self.responses = list(responses)
        self.calls: list = []

    def _call_once(self, messages):
        self.calls.append(messages[-1]["content"])
        return self.responses.pop(0) if self.responses else ""


def test_symmetric_cache():
    print("\n[judge orientation]")
    cfg = load_config("CSR", "llama", "hotpot")

    # A and B keep the caller's roles: upstream judge_yesno_batched keys the cache on
    # (q, a, b) with no canonical reordering, so each unordered pair is asked once, i < j
    judge = RecordingJudge(cfg, ["Correct: YES", "Correct: YES"])
    labels_ab, _, _ = judge.symmetric_batched(["Paris"], ["the capital, Paris city"], ["q"])
    labels_ba, _, _ = judge.symmetric_batched(["the capital, Paris city"], ["Paris"], ["q"])
    check("(A,B) and (B,A) are separate cache entries, as upstream",
          len(judge.calls) == 2, f"{len(judge.calls)} calls")
    check("the caller's order reaches the prompt unchanged",
          judge.calls[0].index("Paris") < judge.calls[0].index("the capital")
          and judge.calls[1].index("the capital") < judge.calls[1].index("Paris"),
          judge.calls[0][:160])
    check("both orientations still parse to the same verdict here",
          labels_ab == labels_ba == [1], f"{labels_ab} vs {labels_ba}")

    # kind="judge" (HotpotQA): correctness reuses the equivalence prompt, with the
    # prediction in the Response 1 slot and the gold in Response 2
    shared = RecordingJudge(cfg, ["Correct: YES"])
    corr = shared.score_against_gold(["Paris"], "the capital, Paris city", "q", "judge")
    sent = shared.calls[-1]
    check("judge-kind correctness puts the prediction in the Response 1 slot",
          sent.index("Paris\n") < sent.index("the capital"), sent[:200])
    check("and it reads as correct", corr == [1], f"{corr}")
    check("the prompt is the original one, with no symmetry sentence",
          "Response 1:" in shared.calls[-1]
          and "symmetrically" not in shared.calls[-1]
          and "Answer A" not in shared.calls[-1], shared.calls[-1][:160])

    # kind="gold" (OOD): directional single/multi-gold prompts picked by gold count,
    # prediction always in the A slot
    listed = RecordingJudge(cfg, ["Correct: YES"])
    listed.score_against_gold(["zzz"], ["aaa", "bbb"], "q", "gold")
    sent = listed.calls[-1]
    check("gold-kind with several golds uses the multi-gold prompt",
          "Candidate Correct Answers" in sent, sent[:120])
    check("multi-gold keeps the prediction in the generated-answer slot",
          sent.index("zzz") < sent.index("aaa"), sent[:160])

    single = RecordingJudge(cfg, ["Correct: YES"])
    single.score_against_gold(["zzz"], ["aaa"], "q", "gold")
    sent = single.calls[-1]
    check("gold-kind with one gold uses the single-gold prompt",
          "Gold Correct Answer" in sent and "Candidate" not in sent, sent[:120])
    check("single-gold keeps the prediction in the generated-answer slot",
          sent.index("zzz") < sent.index("aaa"), sent[:160])

    equiv_j = RecordingJudge(cfg, ["Correct: YES"])
    equiv_j.symmetric_batched(["zzz"], ["aaa"], ["q"])
    sent = equiv_j.calls[-1]
    check("rollout-vs-rollout uses the original judge prompt",
          "Response 1:" in sent and "Gold Correct Answer" not in sent, sent[:120])


def test_unparsed_not_cached():
    print("\n[unparsed handling]")
    cfg = load_config("CSR", "llama", "hotpot")

    # both attempts of the first batch are unreadable (max_retries=2), then a later one is fine
    judge = RecordingJudge(cfg, ["???", "???", "Correct: YES"])
    labels, unparsed, missing = judge.symmetric_batched(["X"], ["Y"], ["q"])
    check("unparsable verdict is reported as missing",
          unparsed == 1 and missing == [0], f"unparsed={unparsed} missing={missing}")
    check("its label is 0 but carries no information", labels == [0], str(labels))
    check("it is NOT written to the cache", len(judge.cache) == 0,
          f"cache size {len(judge.cache)}")

    labels2, unparsed2, _ = judge.symmetric_batched(["X"], ["Y"], ["q"])
    check("a later batch re-asks instead of reusing a fabricated NO",
          labels2 == [1] and unparsed2 == 0, f"{labels2} unparsed={unparsed2}")

    stats = judge.stats_summary()
    check("the unparsed rate is recorded for the run log",
          stats["semantic_equivalence"]["unparsed"] >= 1,
          str(stats["semantic_equivalence"]))


def main():
    print("=" * 70)
    print("CSR calibration reward")
    print("=" * 70)

    test_pair_enumeration()
    test_permutation_equivariance()
    test_reward_levels()
    test_no_offset_no_clamp()
    test_affine_equivalence_under_grpo()
    test_scale_identity_through_trl()
    test_missing_verdicts_dropped()
    test_single_rollout()
    test_cali_w_applied_once()
    test_symmetric_cache()
    test_unparsed_not_cached()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

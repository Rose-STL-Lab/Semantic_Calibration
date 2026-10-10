"""GRPO reward functions for RLVR (correctness only) and CSR (correctness plus a
semantic calibration term over the K rollouts of a prompt group)."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import torch

from .text import extract_completion_text, extract_user_question, normalize_for_equiv


# Reachable endpoints of the per-pair BCE.  r_cal = (CE_MAX - mean_ce) / DELTA_CE is an
# exact affine map, not upstream's clamp(5 - ce, 0, 5) hinge (which collapses every
# rollout with >=3 mismatching peers onto 0), so for binary verdicts r_cal is exactly the
# agreement fraction 1 - m/(K-1) with m mismatching peers: one of 0, 1/(K-1), ..., 1.
EPS = 1e-6
CE_MIN = -math.log1p(-EPS)      # 1.0000e-06, reached when the verdict matches
CE_MAX = -math.log(EPS)         # 13.8155,    reached when it does not
DELTA_CE = CE_MAX - CE_MIN      # slope of r_cal


def group_by_prompt(prompts) -> OrderedDict:
    groups: OrderedDict = OrderedDict()
    for i, prompt in enumerate(prompts):
        groups.setdefault(str(prompt), []).append(i)
    return groups


def gold_for_row(answer, index: int, batch: int, n: int):
    if isinstance(answer, list):
        if len(answer) in (batch, n):
            return answer[index]
        raise ValueError(f"unexpected answer length {len(answer)} (batch={batch}, n={n})")
    return answer


class RewardComputer:
    """Judge-backed GRPO rewards, shared between RLVR and CSR."""

    def __init__(self, cfg, judge, correctness_prompt_kind: str = "judge"):
        train_cfg = cfg["method"]["train"]
        self.judge = judge
        self.kind = correctness_prompt_kind
        self.markers = cfg["model"].get("chat_end_markers", [])
        self.corr_w = float(train_cfg.get("corr_w", 1.0))
        self.cali_w_start = float(train_cfg.get("cali_w_start", 0.0))
        self.cali_w_end = float(train_cfg.get("cali_w_end", 0.0))
        # "linear" default keeps existing runs; start == end makes any schedule constant
        self.cali_w_schedule = str(train_cfg.get("cali_w_schedule", "linear")).lower()
        self.cali_w_sigmoid_slope = float(train_cfg.get("cali_w_sigmoid_slope", 10.0))
        if self.cali_w_schedule not in ("linear", "sigmoid", "constant"):
            raise ValueError(f"unknown method.train.cali_w_schedule "
                             f"{self.cali_w_schedule!r}; expected linear, sigmoid or constant")
        self.cali_w = self.cali_w_start
        self._verdict_cache: dict[bytes, list[int]] = {}

    # -- calibration weight schedule -------------------------------------
    def _schedule_fraction(self, p: float) -> float:
        """Fraction of the start -> end travel done at ``p``; the sigmoid is rescaled to
        hit exactly 0 and 1 (raw slope 10 would start at 0.0067, missing cali_w_start)."""
        if self.cali_w_schedule == "linear":
            return p
        if self.cali_w_schedule == "constant":
            return 0.0
        a = self.cali_w_sigmoid_slope
        lo, hi = 1.0 / (1.0 + math.exp(a / 2.0)), 1.0 / (1.0 + math.exp(-a / 2.0))
        return (1.0 / (1.0 + math.exp(-a * (p - 0.5))) - lo) / (hi - lo)

    def set_progress(self, progress: float) -> float:
        """Move cali_w from cali_w_start to cali_w_end over [0, 1] of training."""
        p = min(max(float(progress), 0.0), 1.0)
        frac = self._schedule_fraction(p)
        self.cali_w = self.cali_w_start + frac * (self.cali_w_end - self.cali_w_start)
        return self.cali_w

    # -- helpers ----------------------------------------------------------
    def _batch_key(self, prompts, completions, answer) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        n, batch = len(completions), len(prompts)
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            question = extract_user_question(prompt)
            answer_text = normalize_for_equiv(completion, self.markers)
            gold = normalize_for_equiv(gold_for_row(answer, i, batch, n), self.markers)
            for part in (question, answer_text, gold):
                digest.update((part or "").encode("utf-8", "ignore"))
                digest.update(b"\x00")
        return digest.digest()

    def _judge_group(self, answers, gold, question) -> list[int]:
        return self.judge.score_against_gold(answers, gold, question, self.kind)

    # -- reward functions -------------------------------------------------
    @torch.no_grad()
    def calibration_reward(self, prompts, completions, answer, **_):
        """CSR's semantic calibration term, scaled by the current cali_w."""
        n, batch = len(completions), len(prompts)
        if n != batch:
            raise ValueError(f"expected one prompt per completion, got n={n}, batch={batch}")

        texts = [extract_completion_text(c, self.markers) for c in completions]
        rewards = torch.zeros(n, dtype=torch.float32)
        verdicts = [0] * n

        for _, indices in group_by_prompt(prompts).items():
            group = [normalize_for_equiv(texts[i], self.markers) for i in indices]
            gold = gold_for_row(answer, indices[0], batch, n)
            question = extract_user_question(prompts[indices[0]])
            k = len(group)

            q_labels = self._judge_group(group, gold, question)
            for j, row in enumerate(indices):
                verdicts[row] = int(q_labels[j])
            q = torch.tensor(q_labels, dtype=torch.float32)

            if k <= 1:
                # no peers: neutral 0.5, and a K=1 group has zero advantage anyway
                r_cal = torch.full((k,), 0.5, dtype=torch.float32)
            else:
                a_list, b_list, pair_idx = [], [], []
                for j in range(k):
                    for i in range(j + 1, k):      # unordered: k(k-1)/2 pairs
                        a_list.append(group[j])
                        b_list.append(group[i])
                        pair_idx.append((j, i))

                same, _unparsed, missing = self.judge.symmetric_batched(
                    a_list, b_list, [question] * len(a_list))
                dropped = set(missing)

                # one verdict per pair, credited to both endpoints against each one's own q
                owners, verdicts_ji = [], []
                for idx, (j, i) in enumerate(pair_idx):
                    if idx in dropped:
                        continue        # unreadable verdict carries no evidence
                    verdict = float(same[idx])
                    owners.extend((j, i))
                    verdicts_ji.extend((verdict, verdict))

                # float64: in float32 the s=1,q=0 branch leaves a 9.5e-4 residue that
                # would blur the exact 0/1 endpoints
                owner_idx = torch.tensor(owners, dtype=torch.long)
                s = torch.tensor(verdicts_ji, dtype=torch.float64)
                q_pair = q.to(torch.float64)[owner_idx]

                s_soft = EPS + (1.0 - 2.0 * EPS) * s
                ce_pair = -(q_pair * torch.log(s_soft)
                            + (1.0 - q_pair) * torch.log1p(-s_soft))

                ce_sum = torch.zeros(k, dtype=torch.float64)
                count = torch.zeros(k, dtype=torch.float64)
                ce_sum.scatter_add_(0, owner_idx, ce_pair)
                count.scatter_add_(0, owner_idx, torch.ones_like(ce_pair))

                # clamp_min keeps the division finite so no NaN can reach the advantages
                has_evidence = count > 0
                mean_ce = ce_sum / count.clamp_min(1.0)
                per_peer = (CE_MAX - mean_ce) / DELTA_CE
                if bool(has_evidence.any()):
                    # every pair dropped -> group mean, so no evidence means zero
                    # advantage rather than a worst score
                    fallback = per_peer[has_evidence].mean()
                else:
                    fallback = torch.tensor(0.5, dtype=torch.float64)
                r_cal = torch.where(has_evidence, per_peer, fallback).to(torch.float32)

            for j, row in enumerate(indices):
                rewards[row] = r_cal[j]

        # fail here: a NaN reward otherwise surfaces much later as grad_norm=NaN
        if not torch.isfinite(rewards).all():
            raise ValueError(f"[calibration_reward] non-finite reward: {rewards.tolist()}")

        # hand the verdicts to correctness_reward so the judge is not paid twice
        self._verdict_cache[self._batch_key(prompts, completions, answer)] = verdicts
        return (rewards * float(self.cali_w)).tolist()


    @torch.no_grad()
    def correctness_reward(self, prompts, completions, answer, **_):
        """Binary correctness, ``corr_w`` if the judge says YES else 0."""
        n, batch = len(completions), len(prompts)
        if n != batch:
            raise ValueError(f"expected one prompt per completion, got n={n}, batch={batch}")

        cached = self._verdict_cache.pop(self._batch_key(prompts, completions, answer), None)
        if cached is not None:
            return [self.corr_w if int(y) == 1 else 0.0 for y in cached]

        answers = [normalize_for_equiv(c, self.markers) for c in completions]
        questions = [extract_user_question(p) for p in prompts]
        # the alias list must stay intact: normalize_for_equiv keeps only the first
        # alias, so "Mumbai" would miss gold ["Bombay", "Mumbai"]
        golds = [gold_for_row(answer, i, batch, n) for i in range(n)]

        # score_against_gold_batch renders each gold with format_gold and picks the
        # equivalence / single-gold / multi-gold template per row, exactly as the
        # evaluation path does
        before = int(self.judge.correctness_stats["unparsed"])
        rows = self.judge.score_against_gold_batch(
            preds_list=[[a] for a in answers], golds=golds,
            questions=questions, kind=self.kind)
        labels = [row[0] for row in rows]
        unparsed = int(self.judge.correctness_stats["unparsed"]) - before
        if unparsed:
            print(f"[judge] {unparsed}/{len(labels)} correctness verdicts unparsed")
        return [self.corr_w if y == 1 else 0.0 for y in labels]

    # -- TRL wiring -------------------------------------------------------
    def reward_funcs(self, reward: str) -> list:
        """Closures, not bound methods: TRL names each logged mean after ``__name__``.
        TRL sums them (corr_w * q + cali_w * r_cal) under its default scale_rewards="group",
        which this reward relies on and never overrides."""

        def correctness_reward(prompts, completions, answer, **kwargs):
            return self.correctness_reward(prompts, completions, answer, **kwargs)

        def calibration_reward(prompts, completions, answer, **kwargs):
            return self.calibration_reward(prompts, completions, answer, **kwargs)

        if reward == "correctness":
            return [correctness_reward]
        if reward == "calibration_correctness":
            # calibration runs first: it judges correctness and caches the verdicts
            return [calibration_reward, correctness_reward]
        raise ValueError(f"unknown reward {reward!r}; "
                         f"expected 'correctness' or 'calibration_correctness'")

# GRPO reward functions for RLVR (correctness only) and CSR (correctness plus a
# semantic calibration term over the K rollouts of a prompt group).

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import torch

from .parsing import strip_prompt_prefix, extract_user_question, normalize_for_equiv


# reachable endpoints of the per-pair BCE; r_cal = (CE_MAX - mean_ce) / DELTA_CE maps
# them affinely onto [0, 1], so for binary verdicts r_cal is the agreement fraction.
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


def row_questions(prompts, question=None) -> list[str]:
    # prefer the dataset's question column: on context datasets the user turn starts
    # with the passages, which the judge's truncation would keep instead of the question
    if question is not None:
        if len(question) != len(prompts):
            raise ValueError(f"expected one question per prompt, got {len(question)} "
                             f"for {len(prompts)} prompts")
        return [str(q or "").strip() for q in question]
    return [extract_user_question(p) for p in prompts]


class RewardComputer:
    def __init__(self, cfg, judge, correctness_prompt_kind: str = "judge"):
        train_cfg = cfg["method"]["train"]
        self.judge = judge
        self.kind = correctness_prompt_kind
        self.markers = cfg["model"].get("chat_end_markers", [])
        self.corr_w = float(train_cfg.get("corr_w", 1.0))
        self.cali_w_start = float(train_cfg.get("cali_w_start", 0.0))
        self.cali_w_end = float(train_cfg.get("cali_w_end", 0.0))
        # start == end makes any schedule constant
        self.cali_w_schedule = str(train_cfg.get("cali_w_schedule", "linear")).lower()
        self.cali_w_sigmoid_slope = float(train_cfg.get("cali_w_sigmoid_slope", 10.0))
        if self.cali_w_schedule not in ("linear", "sigmoid"):
            raise ValueError(f"unknown method.train.cali_w_schedule "
                             f"{self.cali_w_schedule!r}; expected linear or sigmoid "
                             f"(equal endpoints already give a constant weight)")
        self.cali_w = self.cali_w_start
        self._verdict_cache: dict[bytes, list[int]] = {}

    # -- calibration weight schedule -------------------------------------
    def _schedule_fraction(self, p: float) -> float:
        # fraction of the start -> end travel at p; the sigmoid is rescaled to hit 0 and 1
        if self.cali_w_schedule == "linear":
            return p
        a = self.cali_w_sigmoid_slope
        lo, hi = 1.0 / (1.0 + math.exp(a / 2.0)), 1.0 / (1.0 + math.exp(-a / 2.0))
        return (1.0 / (1.0 + math.exp(-a * (p - 0.5))) - lo) / (hi - lo)

    def set_progress(self, progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        frac = self._schedule_fraction(p)
        self.cali_w = self.cali_w_start + frac * (self.cali_w_end - self.cali_w_start)
        return self.cali_w

    # -- helpers ----------------------------------------------------------
    def _batch_key(self, prompts, completions, answer, question=None) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        n, batch = len(completions), len(prompts)
        questions = row_questions(prompts, question)
        for i, completion in enumerate(completions):
            q_text = questions[i]
            answer_text = normalize_for_equiv(completion, self.markers)
            gold = normalize_for_equiv(gold_for_row(answer, i, batch, n), self.markers)
            for part in (q_text, answer_text, gold):
                digest.update((part or "").encode("utf-8", "ignore"))
                digest.update(b"\x00")
        return digest.digest()

    def _questions(self, prompts, question) -> list[str]:
        questions = row_questions(prompts, question)
        if not getattr(self, "_logged_question", False):
            # one line at the start of training, to confirm what the judge is shown
            source = "dataset column" if question is not None else "prompt user turn"
            print(f"[reward] judge question from {source}: {questions[0][:200]!r}", flush=True)
            self._logged_question = True
        return questions

    def _judge_group(self, answers, gold, question) -> list[int]:
        return self.judge.score_against_gold(answers, gold, question, self.kind)

    # -- reward functions -------------------------------------------------
    @torch.no_grad()
    def calibration_reward(self, prompts, completions, answer, question=None, **_):
        n, batch = len(completions), len(prompts)
        if n != batch:
            raise ValueError(f"expected one prompt per completion, got n={n}, batch={batch}")

        texts = [strip_prompt_prefix(c, self.markers) for c in completions]
        questions = self._questions(prompts, question)
        rewards = torch.zeros(n, dtype=torch.float32)
        verdicts = [0] * n

        for _, indices in group_by_prompt(prompts).items():
            group = [normalize_for_equiv(texts[i], self.markers) for i in indices]
            gold = gold_for_row(answer, indices[0], batch, n)
            q_text = questions[indices[0]]
            k = len(group)

            q_labels = self._judge_group(group, gold, q_text)
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
                    a_list, b_list, [q_text] * len(a_list))
                dropped = set(missing)

                # one verdict per pair, credited to both endpoints against each one's own q
                owners, verdicts_ji = [], []
                for idx, (j, i) in enumerate(pair_idx):
                    if idx in dropped:
                        continue        # unreadable verdict carries no evidence
                    verdict = float(same[idx])
                    owners.extend((j, i))
                    verdicts_ji.extend((verdict, verdict))

                # float64: float32 leaves a residue at the s=1, q=0 endpoint
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

                # clamp_min keeps the division finite; no NaN reaches the advantages
                has_evidence = count > 0
                mean_ce = ce_sum / count.clamp_min(1.0)
                per_peer = (CE_MAX - mean_ce) / DELTA_CE
                if bool(has_evidence.any()):
                    # rollouts with no readable pair get the group mean (zero advantage)
                    fallback = per_peer[has_evidence].mean()
                else:
                    fallback = torch.tensor(0.5, dtype=torch.float64)
                r_cal = torch.where(has_evidence, per_peer, fallback).to(torch.float32)

            for j, row in enumerate(indices):
                rewards[row] = r_cal[j]

        # fail here: a NaN reward otherwise surfaces much later as grad_norm=NaN
        if not torch.isfinite(rewards).all():
            raise ValueError(f"[calibration_reward] non-finite reward: {rewards.tolist()}")

        # hand the verdicts to correctness_reward; the judge is not paid twice
        self._verdict_cache[self._batch_key(prompts, completions, answer, question)] = verdicts
        return (rewards * float(self.cali_w)).tolist()

    @torch.no_grad()
    def correctness_reward(self, prompts, completions, answer, question=None, **_):
        n, batch = len(completions), len(prompts)
        if n != batch:
            raise ValueError(f"expected one prompt per completion, got n={n}, batch={batch}")

        cached = self._verdict_cache.pop(
            self._batch_key(prompts, completions, answer, question), None)
        if cached is not None:
            return [self.corr_w if int(y) == 1 else 0.0 for y in cached]

        answers = [normalize_for_equiv(c, self.markers) for c in completions]
        questions = self._questions(prompts, question)
        # keep the alias list intact; format_gold renders it as the eval path does
        golds = [gold_for_row(answer, i, batch, n) for i in range(n)]

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
        # closures, not bound methods: TRL names each logged reward after __name__.
        # TRL sums them (corr_w * q + cali_w * r_cal) with its default scale_rewards="group".

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

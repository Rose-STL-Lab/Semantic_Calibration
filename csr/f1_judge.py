"""Token-F1 judge: the same interface as the LLM judge, with no model behind it.

correct(a, gold) = max_alias F1(a, alias) > threshold, equivalent(a, b) = F1(a, b)
> threshold -- deterministic, so ``judge.backend: f1`` needs no server and no API key.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

from .text import canonicalize_answer

# TriviaQA official eval code, as vendored by RD in rd_util/EvaluationMetrics.py.
# Verbatim, and not csr.text.normalize_answer (different step order), so this judge
# and RD's training loop tokenize identically.
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(str(s or "")))))).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_score_aliases(prediction, aliases: Sequence[str]) -> float:
    """Best F1 over every acceptable answer, as RD's f1_score_aliases does."""
    if prediction is None:
        return 0.0
    scores = [f1_score(prediction, a) for a in aliases if a is not None]
    return max(scores) if scores else 0.0


def _as_aliases(gold) -> list[str]:
    """A gold field is either one string or a list of acceptable answers."""
    if gold is None:
        return []
    if isinstance(gold, (list, tuple)):
        return [str(g) for g in gold if g is not None and str(g).strip() != ""]
    return [str(gold)]


def new_stats() -> dict:
    return {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []}


class F1Judge:
    """Drop-in replacement for ``Judge`` that scores with token F1.

    Verdicts are never unparsable, so ``unparsed`` is always 0 and ``missing`` [].
    """

    def __init__(self, cfg, chat_end_markers: Sequence[str] = ()):
        judge_cfg = cfg["judge"] if "judge" in cfg else cfg
        self.threshold = float(judge_cfg.get("f1_threshold", 0.5))
        # Strictly greater than, as in RD: at the default 0.5 an answer sharing
        # exactly half its tokens with the gold is wrong, not right.
        self.model_id = f"f1@{self.threshold:g}"
        self.chat_end_markers = list(chat_end_markers or ())
        # No prompts and no network; kept as attributes only because callers read them.
        self.prompt = None
        self.single_gold_prompt = None
        self.multi_gold_prompt = None
        self.base_url = None
        # Caching buys nothing here; the attribute stays for len(judge.cache) callers.
        self.cache: dict = {}
        self.same_stats = new_stats()
        self.correctness_stats = new_stats()

    # -- scoring ---------------------------------------------------------
    def _clean(self, text) -> str:
        """Same answer extraction the LLM judge applies before it asks."""
        return canonicalize_answer(text, self.chat_end_markers)

    def _verdict(self, a, aliases: Sequence[str]) -> int:
        return int(f1_score_aliases(self._clean(a),
                                    [self._clean(x) for x in aliases]) > self.threshold)

    @staticmethod
    def _record(stats: dict, labels: Sequence[int]) -> None:
        stats["total"] += len(labels)
        yes = sum(1 for y in labels if y == 1)
        stats["parsed_yes"] += yes
        stats["parsed_no"] += len(labels) - yes

    # -- the Judge API ---------------------------------------------------
    def yes_no_batched(self, preds, golds, questions, use_cache: bool = True,
                       shortcut: bool = True, stats: dict | None = None,
                       return_missing: bool = False, template=None):
        assert len(preds) == len(golds) == len(questions)
        stats = stats if stats is not None else self.same_stats
        labels = [self._verdict(a, _as_aliases(b)) for a, b in zip(preds, golds)]
        self._record(stats, labels)
        return (labels, 0, []) if return_missing else (labels, 0)

    def symmetric_batched(self, a_list, b_list, questions):
        """One equivalence verdict per pair; F1 is symmetric, so orientation cannot matter."""
        return self.yes_no_batched(a_list, b_list, questions,
                                   stats=self.same_stats, return_missing=True)

    def correctness_batched(self, preds, golds, questions, template=None):
        return self.yes_no_batched(preds, golds, questions,
                                   stats=self.correctness_stats)

    def score_against_gold(self, preds: Sequence[str], gold, question: str,
                           kind: str = "judge") -> list[int]:
        """``kind`` is accepted and ignored: it chooses a prompt, and there is none."""
        aliases = _as_aliases(gold)
        labels = [self._verdict(p, aliases) for p in preds]
        self._record(self.correctness_stats, labels)
        return labels

    def score_against_gold_batch(self, preds_list, golds, questions,
                                 kind: str = "judge") -> list[list[int]]:
        return [self.score_against_gold(preds, gold, question, kind)
                for preds, gold, question in zip(preds_list, golds, questions)]

    # -- reporting -------------------------------------------------------
    def stats_summary(self) -> dict:
        def block(stats):
            total = int(stats["total"])
            return {
                "total": total,
                "parsed_yes": int(stats["parsed_yes"]),
                "parsed_no": int(stats["parsed_no"]),
                "unparsed": 0,
                "unparsed_rate": 0.0,
            }

        return {
            "model_id": self.model_id,
            "backend": "f1",
            "f1_threshold": self.threshold,
            "correctness": block(self.correctness_stats),
            "semantic_equivalence": block(self.same_stats),
            "cache_size": 0,
        }

    def reset_stats(self) -> None:
        self.same_stats = new_stats()
        self.correctness_stats = new_stats()


class F1EquivalenceJudge:
    """LLM judge for correctness, token F1 for rollout-rollout equivalence (CSR-F1).

    Note ``>=`` here, as in the original CSR-F1 scripts, where ``F1Judge`` uses ``>``
    like RD. Everything except ``symmetric_batched`` forwards to the LLM judge.
    """

    def __init__(self, llm_judge, cfg, chat_end_markers: Sequence[str] = ()):
        judge_cfg = cfg["judge"] if "judge" in cfg else cfg
        self.llm = llm_judge
        self.threshold = float(judge_cfg.get("f1_threshold", 0.5))
        self.chat_end_markers = list(chat_end_markers or ())
        self.f1_same_stats = new_stats()

    def __getattr__(self, name):
        # only reached for attributes not defined here
        return getattr(self.llm, name)

    def symmetric_batched(self, a_list, b_list, questions):
        assert len(a_list) == len(b_list) == len(questions)
        labels = [
            int(f1_score(canonicalize_answer(a, self.chat_end_markers),
                         canonicalize_answer(b, self.chat_end_markers)) >= self.threshold)
            for a, b in zip(a_list, b_list)
        ]
        F1Judge._record(self.f1_same_stats, labels)
        return labels, 0, []

    def stats_summary(self) -> dict:
        summary = self.llm.stats_summary()
        stats = self.f1_same_stats
        summary["semantic_equivalence"] = {
            "backend": "f1",
            "f1_threshold": self.threshold,
            "total": int(stats["total"]),
            "parsed_yes": int(stats["parsed_yes"]),
            "parsed_no": int(stats["parsed_no"]),
            "unparsed": 0,
            "unparsed_rate": 0.0,
        }
        return summary

    def reset_stats(self) -> None:
        self.llm.reset_stats()
        self.f1_same_stats = new_stats()

# Token-F1 equivalence: equivalent(a, b) = F1(a, b) >= threshold. Deterministic and
# offline, wrapped around the LLM judge so correctness labels stay the LLM's.
# The CSR-F1 configuration: --set judge.equivalence=f1

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

from .parsing import canonicalize_answer

# TriviaQA official eval code, as vendored by RD in rd_util/EvaluationMetrics.py.
# Verbatim, and not csr.parsing.normalize_answer (different step order), so this judge
# and RD's training loop tokenize identically.
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def normalize_answer(s: str) -> str:
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
    # best F1 over the aliases, as RD does
    if prediction is None:
        return 0.0
    scores = [f1_score(prediction, a) for a in aliases if a is not None]
    return max(scores) if scores else 0.0


def _record(stats: dict, labels: Sequence[int]) -> None:
    stats["total"] += len(labels)
    yes = sum(1 for y in labels if y == 1)
    stats["parsed_yes"] += yes
    stats["parsed_no"] += len(labels) - yes


def new_stats() -> dict:
    return {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0}


class F1EquivalenceJudge:
    # LLM judge for correctness, token F1 for rollout-rollout equivalence (CSR-F1).
    # Note >=, as in the original CSR-F1 scripts (RD's exact match uses >).
    # Everything but symmetric_batched forwards to the LLM judge.

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
        _record(self.f1_same_stats, labels)
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

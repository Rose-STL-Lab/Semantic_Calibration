"""RLCR baseline: reward functions and its HotpotQA prompt construction.

RLCR emits <think>/<answer>/<analysis>/<confidence>, is scored by exact match
rather than the LLM judge, and takes the supporting paragraphs in the prompt.
"""

from __future__ import annotations

import random
import re
import string
from typing import List, Tuple

FORMAT_PATTERNS = {
    "tbac": r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z",
    "ta": r".*?</think>\s*<answer>.*?</answer>\s*\Z",
    "tac": r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z",
    "tabc": r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z",
}


def normalize_answer(s: str) -> str:
    text = str(s or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_any(prediction: str, gold) -> float:
    """``exact_match_score`` against a gold that may be a list of aliases; a list
    stringified whole would never match. Mirrors rd_util/EvaluationMetrics.py."""
    if isinstance(gold, (list, tuple)):
        return max((exact_match_score(prediction, g) for g in gold), default=0.0)
    return exact_match_score(prediction, gold)


def completion_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        text = completion[0].get("content", "")
    elif isinstance(completion, list) and completion:
        text = completion[0]
    else:
        text = completion
    return str(text or "").strip()


def last_tag_text(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL | re.MULTILINE)
    return matches[-1].strip() if matches else ""


def _confidence_of(content: str) -> float | None:
    raw = last_tag_text(content, "confidence")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class RLCRRewards:
    def __init__(self, format_pattern: str = "tabc"):
        if format_pattern not in FORMAT_PATTERNS:
            raise ValueError(f"unsupported format pattern {format_pattern!r}; "
                             f"choose from {sorted(FORMAT_PATTERNS)}")
        self.pattern = FORMAT_PATTERNS[format_pattern]

    def format_reward(self, completions, **_) -> list[float]:
        """1.0 when the full tag structure is present and confidence is in [0, 1]."""
        rewards = []
        for completion in completions:
            content = completion_text(completion)
            if not re.match(self.pattern, content, re.DOTALL | re.MULTILINE):
                rewards.append(0.0)
                continue
            confidence = _confidence_of(content)
            rewards.append(1.0 if confidence is not None and 0.0 <= confidence <= 1.0 else 0.0)
        return rewards

    def accuracy_reward(self, completions, answer, source=None, **_) -> list[float]:
        """Exact match on the <answer> tag; zero unless the format is valid."""
        golds = list(answer)
        sources = list(source) if source is not None else ["hotpot"] * len(completions)
        formats = self.format_reward(completions)
        rewards = []
        for completion, gold, src, ok in zip(completions, golds, sources, formats):
            if ok == 0.0 or src != "hotpot":
                rewards.append(0.0)
                continue
            rewards.append(exact_match_any(last_tag_text(completion_text(completion), "answer"),
                                           gold))
        return rewards

    def brier_reward(self, completions, answer, source=None, **_) -> list[float]:
        """1 - (correctness - confidence)^2, the calibration term."""
        correctness = self.accuracy_reward(completions, answer, source=source)
        formats = self.format_reward(completions)
        rewards = []
        for completion, correct, ok in zip(completions, correctness, formats):
            if ok == 0.0:
                rewards.append(0.0)
                continue
            confidence = _confidence_of(completion_text(completion))
            rewards.append(0.0 if confidence is None else 1.0 - (correct - confidence) ** 2)
        return rewards

    def mean_confidence_reward(self, completions, **_) -> list[float]:
        """Logged with a near-zero weight, purely to monitor confidence drift."""
        return [min(max(_confidence_of(completion_text(c)) or 0.0, 0.0), 1.0)
                for c in completions]

    def confidence_one_or_zero(self, completions, **_) -> list[float]:
        """Logged with a near-zero weight: fraction collapsing to exactly 0 or 1."""
        rewards = []
        for completion in completions:
            confidence = min(max(_confidence_of(completion_text(completion)) or 0.0, 0.0), 1.0)
            rewards.append(1.0 if abs(confidence - 1.0) < 0.01 or abs(confidence) < 0.01 else 0.0)
        return rewards

    def reward_funcs(self) -> list:
        """Closures in the order ``reward_weights`` in configs/methods/rlcr.yaml expects."""

        def format_reward(completions, **kwargs):
            return self.format_reward(completions, **kwargs)

        def accuracy_reward(completions, answer, source=None, **kwargs):
            return self.accuracy_reward(completions, answer, source=source, **kwargs)

        def brier_reward(completions, answer, source=None, **kwargs):
            return self.brier_reward(completions, answer, source=source, **kwargs)

        def mean_confidence_reward(completions, **kwargs):
            return self.mean_confidence_reward(completions, **kwargs)

        def confidence_one_or_zero(completions, **kwargs):
            return self.confidence_one_or_zero(completions, **kwargs)

        return [format_reward, accuracy_reward, brier_reward,
                mean_confidence_reward, confidence_one_or_zero]


# HotpotQA prompt construction
ANSWER_TAG_INSTRUCTION = (
    "Your answer will be verified with exact match score. To ensure correct verification, "
    "only provide the answer within the <answer> </answer> tags. Do not put any sentences "
    "or reasoning process within the <answer> </answer> tags."
)


def _sample_removed_titles(example, rng: random.Random) -> Tuple[int, List[str]]:
    """Drop paragraphs so the model sometimes lacks the evidence it needs: genuinely
    unanswerable questions are the point of training a confidence head. RLCR training
    only -- the shared context table in csr/data.py keeps every paragraph.
    """
    supporting = list(example["supporting_facts"]["title"])
    all_titles = list(example["context"]["title"])
    non_supporting = list(set(all_titles) - set(supporting))

    choice = rng.choice([0, 1, 2])
    try:
        if choice == 1 and supporting and non_supporting:
            return choice, [rng.choice(supporting), rng.choice(non_supporting)]
        if choice == 2 and supporting:
            return choice, supporting
        if len(non_supporting) >= 2:
            return 0, rng.sample(non_supporting, 2)
    except (IndexError, ValueError):
        pass
    return 0, []


def build_problem(example, rng: random.Random) -> str:
    _, removed = _sample_removed_titles(example, rng)
    sections = []
    index = 0
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        if title in removed:
            continue
        sections.append(f"Paragraph {index} \n\n {chr(10).join(sentences)} \n\n")
        sections.append("--------------------------------\n\n")
        index += 1
    return (
        f"Question: {example['question']} \n\n"
        f"{ANSWER_TAG_INSTRUCTION} \n\n"
        f"Supporting Information: {''.join(sections)} \n\n"
    )


def load_hotpot(cfg, split: str = "train", size: int | None = None):
    """RLCR's own HotpotQA view: paragraphs in the prompt, tags in the answer."""
    from datasets import load_dataset

    seed = int(cfg["seed"])
    system_prompt = cfg["method"]["system_prompt"]
    user_template = cfg["method"].get("user_template", "\n\nPROBLEM: {question}\n\n")
    ds_cfg = cfg["dataset"]

    raw = load_dataset(ds_cfg["hf"]["path"], ds_cfg["hf"]["config"])
    split_name = ds_cfg["train_split"] if split == "train" else ds_cfg["eval_split"]

    def convert(example):
        example_id = str(example.get("id", ""))
        # per-example RNG so paragraph dropping is deterministic across runs
        rng = random.Random(seed + sum(ord(c) for c in example_id))
        problem = build_problem(example, rng)
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(question=problem)},
            ],
            "answer": example.get("answer", ""),
            "source": "hotpot",
            "question": example.get("question", ""),
            "id": example_id,
        }

    dataset = raw[split_name].map(convert, remove_columns=raw[split_name].column_names)
    dataset = dataset.shuffle(seed=seed)
    if size is None:
        size = ds_cfg["train_size"] if split == "train" else ds_cfg["eval_size"]
    if size:
        dataset = dataset.select(range(0, min(int(size), len(dataset))))
    print(f"[data] rlcr hotpot/{split_name}: {len(dataset)} examples")
    return dataset

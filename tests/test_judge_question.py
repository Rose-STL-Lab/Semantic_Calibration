"""Every judge call made by the CSR reward must carry that row's own question.

On context datasets the user turn is "Context: ... Question: q", so the reward reads the
dataset's question column. The pairwise equivalence call once received the whole column
instead of the group's question; this check fails on that.

    python tests/test_judge_question.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import load_config  # noqa: E402
from csr.rewards import RewardComputer  # noqa: E402

QUESTIONS = ["Who wrote Hamlet?", "Capital of France?"]
GOLDS = ["Shakespeare", "Paris"]


class RecordingJudge:
    correctness_stats = {"unparsed": 0}

    def __init__(self):
        self.seen = []

    def score_against_gold(self, answers, gold, question, kind):
        self.seen.append(question)
        return [int(a == gold) for a in answers]

    def score_against_gold_batch(self, preds_list, golds, questions, kind):
        self.seen.extend(questions)
        return [[1] for _ in preds_list]

    def symmetric_batched(self, a_list, b_list, questions):
        self.seen.extend(questions)
        return [int(a == b) for a, b in zip(a_list, b_list)], 0, []


def main():
    cfg = load_config("CSR", "qwen", "hotpot_context")
    judge = RecordingJudge()
    rewards = RewardComputer(cfg, judge)
    prompts, completions, answer, question = [], [], [], []
    for q, gold in zip(QUESTIONS, GOLDS):
        for k in range(4):
            prompts.append([{"role": "user",
                             "content": f"Context:\n{'passage ' * 300}\n\nQuestion: {q}"}])
            completions.append(gold if k % 2 == 0 else "something else")
            answer.append(gold)
            question.append(q)
    rewards.calibration_reward(prompts, completions, answer, question=question)
    rewards.correctness_reward(prompts, completions, answer, question=question)
    bad = [q for q in judge.seen if q not in QUESTIONS]
    assert judge.seen, "the judge was never called"
    assert not bad, f"judge received {bad[0]!r} instead of one row's question"
    print("all checks passed")


if __name__ == "__main__":
    main()

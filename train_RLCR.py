#!/usr/bin/env python
# coding: utf-8

"""
RLCR HotpotQA training script aligned to the RLUQ-style Unsloth/GRPO structure,
while preserving RLCR's original prompt, rewards, and GRPO hyperparameters.
"""

import argparse
import csv
import os
import random
import re
import string
from typing import List, Tuple

import unsloth
import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainerCallback

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 0) Env / seed / device
# ============================================================
HF_CACHE = "./hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device_count =", torch.cuda.device_count())
print("current_device =", torch.cuda.current_device())
print("name0 =", torch.cuda.get_device_name(0))


# ============================================================
# 1) RLCR prompt / experiment setup
# ============================================================
TABC_LONG_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "Here are some guidelines for the analysis: "
    "1. Your task is to point out things where the model could be wrong in its thinking, or things where there might be ambiguity in the solution steps, or in the reasoning process itself.\n"
    "2. You should not suggest ways of fixing the response, your job is only to reason about uncertainties.\n"
    "3. For some questions, the response might be correct. In these cases, It is also okay to have only a small number of uncertainties and then explictly say that I am unable to spot more uncertainties.\n"
    "4. Uncertainties might be different from errors. For example, uncertainties may arise from ambiguities in the question, or from the application of a particular lemma/proof. \n"
    "5. If there are alternate potential approaches that may lead to different answers, you should mention them.\n"
    "6. List out plausible uncertainties, do not make generic statements, be as specific about uncertainties as possible.\n"
    "7. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

FORMAT_PATTERN = "tabc"
REWARD_WEIGHTS = [0.5, 0.5, 0.5, 0.000001, 0.000001]
EXPERIMENT = "RLCR"


# ============================================================
# 2) Reward helpers
# ============================================================
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def extract_completion_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        text = completion[0].get("content", "")
    elif isinstance(completion, list) and completion:
        text = completion[0]
    else:
        text = completion
    return str(text or "").strip()


def _last_tag_text(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL | re.MULTILINE)
    return matches[-1].strip() if matches else ""


def format_reward(completions, **kwargs):
    if FORMAT_PATTERN == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif FORMAT_PATTERN == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif FORMAT_PATTERN == "tac":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif FORMAT_PATTERN == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    else:
        raise ValueError(f"Unsupported format pattern: {FORMAT_PATTERN}")

    contents = [extract_completion_text(c) for c in completions]
    rewards = []
    for content in contents:
        matched = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        if not matched:
            rewards.append(0.0)
            continue
        confidence_text = _last_tag_text(content, "confidence")
        if not confidence_text:
            rewards.append(0.0)
            continue
        try:
            confidence = float(confidence_text)
            rewards.append(1.0 if 0.0 <= confidence <= 1.0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, answer, source=None, **kwargs):
    contents = [extract_completion_text(c) for c in completions]
    golds = list(answer)
    sources = list(source) if source is not None else [None] * len(contents)
    fmt = format_reward(completions)
    rewards = []
    for content, gold, src, fr in zip(contents, golds, sources, fmt):
        if fr == 0.0:
            rewards.append(0.0)
            continue
        pred = _last_tag_text(content, "answer")
        if src == "hotpot":
            rewards.append(exact_match_score(pred, gold))
        else:
            rewards.append(0.0)
    return rewards


def brier_reward(completions, answer, source=None, **kwargs):
    contents = [extract_completion_text(c) for c in completions]
    correctness = accuracy_reward(completions, answer, source=source)
    fmt = format_reward(completions)
    rewards = []
    for content, corr, fr in zip(contents, correctness, fmt):
        if fr == 0.0:
            rewards.append(0.0)
            continue
        confidence_text = _last_tag_text(content, "confidence")
        try:
            conf = float(confidence_text)
            rewards.append(1.0 - (corr - conf) ** 2)
        except Exception:
            rewards.append(0.0)
    return rewards


def mean_confidence_reward(completions, answer=None, **kwargs):
    contents = [extract_completion_text(c) for c in completions]
    rewards = []
    for content in contents:
        confidence_text = _last_tag_text(content, "confidence")
        try:
            conf = max(0.0, min(float(confidence_text), 1.0))
        except Exception:
            conf = 0.0
        rewards.append(conf)
    return rewards


def confidence_one_or_zero(completions, answer=None, **kwargs):
    contents = [extract_completion_text(c) for c in completions]
    rewards = []
    for content in contents:
        confidence_text = _last_tag_text(content, "confidence")
        try:
            conf = max(0.0, min(float(confidence_text), 1.0))
        except Exception:
            conf = 0.0
        rewards.append(1.0 if abs(conf - 1.0) < 0.01 or abs(conf - 0.0) < 0.01 else 0.0)
    return rewards


# ============================================================
# 3) Dataset
# ============================================================
def _sample_removed_titles(example, rng: random.Random) -> Tuple[int, List[str]]:
    supporting_titles = list(example["supporting_facts"]["title"])
    all_titles = list(example["context"]["title"])
    non_supporting_titles = list(set(all_titles) - set(supporting_titles))
    choice = rng.choice([0, 1, 2])
    try:
        if choice == 1 and supporting_titles and non_supporting_titles:
            return choice, [rng.choice(supporting_titles), rng.choice(non_supporting_titles)]
        if choice == 2 and supporting_titles:
            return choice, supporting_titles
        if len(non_supporting_titles) >= 2:
            return 0, rng.sample(non_supporting_titles, 2)
    except Exception:
        pass
    return 0, []


def _build_hotpot_problem(example, rng: random.Random) -> str:
    sys = (
        "Your answer will be verified with exact match score. To ensure correct verification, only provide the answer "
        "within the <answer> </answer> tags. Do not put any sentences or reasoning process within the <answer> </answer> tags."
    )
    _, removed_titles = _sample_removed_titles(example, rng)
    sections = []
    para_idx = 0
    for title, sentence_cluster in zip(example["context"]["title"], example["context"]["sentences"]):
        if title in removed_titles:
            continue
        sentence_cluster_str = "\n".join(sentence_cluster)
        sections.append(f"Paragraph {para_idx} \n\n {sentence_cluster_str} \n\n")
        sections.append("--------------------------------\n\n")
        para_idx += 1
    info = "".join(sections)
    return (
        f"Question: {example['question']} \n\n"
        f"{sys} \n\n"
        f"Supporting Information: {info} \n\n"
    )


def convert_to_item(ex):
    example_id = str(ex.get("id", ""))
    local_seed = SEED + sum(ord(c) for c in example_id)
    rng = random.Random(local_seed)
    problem = _build_hotpot_problem(ex, rng)
    user_format = f"\n\nPROBLEM: {problem}\n\n"
    return {
        "prompt": [
            {"role": "system", "content": TABC_LONG_PROMPT},
            {"role": "user", "content": user_format},
        ],
        "answer": ex.get("answer", ""),
        "source": "hotpot",
        "question": ex.get("question", ""),
        "id": example_id,
    }


raw = load_dataset("hotpot_qa", "distractor")
train_ds = raw["train"].map(convert_to_item).shuffle(seed=SEED).select(range(min(10000, len(raw["train"]))))
validation_ds = raw["validation"].map(convert_to_item).shuffle(seed=SEED).select(range(min(400, len(raw["validation"]))))
print(f"train size={len(train_ds)}, validation size={len(validation_ds)}")


# ============================================================
# 4) Policy model
# ============================================================
max_seq_length = 4096
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    # target_modules=["q_proj", "v_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
).to(DEVICE)


# ============================================================
# 5) GRPO config (preserve RLCR params)
# ============================================================
training_args = GRPOConfig(
    beta=0.0,
    loss_type="grpo",
    shuffle_dataset=True,
    learning_rate=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.0,
    warmup_ratio=0.05,
    lr_scheduler_type="constant_with_warmup",
    optim="paged_adamw_8bit",
    logging_steps=5,
    logging_strategy="steps",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    num_iterations=1,
    max_prompt_length=3072,
    max_completion_length=1024,
    num_train_epochs=1,
    max_steps=-1,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=10,
    eval_strategy="no",
    use_vllm=True,
    vllm_gpu_memory_utilization=0.4,
    eval_on_start=False,
    max_grad_norm=1.0,
    report_to="none",
    output_dir=f"outputs_{EXPERIMENT}",
    bf16=True,
    remove_unused_columns=False,
    temperature=0.7,
    reward_weights=REWARD_WEIGHTS,
    scale_rewards=False,
    seed=SEED,
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward,
        accuracy_reward,
        brier_reward,
        mean_confidence_reward,
        confidence_one_or_zero,
    ],
    args=training_args,
    train_dataset=train_ds,
)


# ============================================================
# 6) Logging
# ============================================================
class MetricCSVCallback(TrainerCallback):
    def __init__(self, outdir="outputs", csv_name="metrics_log.csv"):
        self.csv_path = os.path.join(outdir, csv_name)
        os.makedirs(outdir, exist_ok=True)
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._csv_header = None
        print(f"[MetricCSVCallback] CSV -> {self.csv_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": int(state.global_step)}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
        if self._csv_writer is None:
            self._csv_header = ["step"] + sorted([k for k in row.keys() if k != "step"])
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header, extrasaction="ignore")
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass


trainer.add_callback(MetricCSVCallback(outdir=f"outputs_{EXPERIMENT}", csv_name="metrics_log.csv"))


# ============================================================
# 7) Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from-checkpoint", default="your checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("Starting training from scratch.")
    # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.train()
    save_dir = f"grpo_saved_lora_qa_{EXPERIMENT}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved LoRA adapter to {save_dir}")


if __name__ == "__main__":
    main()

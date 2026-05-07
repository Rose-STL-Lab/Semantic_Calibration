#!/usr/bin/env python
# coding: utf-8

# ============================================================
# NQ Open verbal-confidence evaluation for RewardingDoubt
# - ONE generation per question
# - Output format: Answer: <answer>, Confidence: <0-10>
# - Standalone script aligned with other RD eval methods
# ============================================================

import os
import argparse

# --- HF cache / vLLM env ---
HF_CACHE = "./hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# ===== 0. Imports =====
import csv
import json
import random
import re
import string
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from unsloth import FastLanguageModel


# ===== 0. Seed / Device =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "unsloth/meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
SAMPLE_SIZE = 1000
EVAL_MAX_NEW_TOKENS = 256
ECE_BINS = 10


# ============================================================
# 1) Policy Model (Unsloth LoRA / merged checkpoint)
# ============================================================
def unsloth_load_checkpoint(checkpoint_path: Optional[str], base_model: str = BASE_MODEL, max_seq_length: int = MAX_SEQ_LENGTH):
    model_name = checkpoint_path or base_model
    if checkpoint_path:
        model_name = os.path.abspath(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        fix_tokenizer=False,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if checkpoint_path:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ============================================================
# 2) Judge / Correctness
# ============================================================
JUDGE_SYSTEM = "You are a helpful assistant that helps people find information."
JUDGE_PROMPT = """Given a question and two potential answers, you need to analyze whether the two answers convey the same meaning. Respond with \"yes\" or \"no\".
Your response should be completely based on the similarity between the two answers.
DO NOT use your own knowledge base when do the comparison.

Question:
{q}
LLM Response 1:
{a}
LLM Response 2:
{b}

Your response should use the following format:
Correct: <yes or no>
""".rstrip()

NQ_OPEN_CORRECTNESS_JUDGE_PROMPT = """Given a question, one generated answer, and the gold correct answer, determine whether the generated answer matches the gold answer in meaning. Respond with \"yes\" or \"no\".
Your response should be completely based on the similarity between the generated answer and the gold correct answer.
DO NOT use your own knowledge base when do the comparison.

Question:
{q}
Generated Answer:
{a}
Gold Correct Answer:
{b}

Your response should use the following format:
Correct: <yes or no>
""".rstrip()

NQ_OPEN_MULTI_CORRECTNESS_JUDGE_PROMPT = """Given a question, one generated answer, and a list of candidate correct answers, you need to analyze whether the generated answer matches any candidate correct answer. Respond with \"yes\" or \"no\".
Your response should be completely based on the similarity between the generated answer and the candidate correct answers.
DO NOT use your own knowledge base when do the comparison.

Question:
{q}
Generated Answer:
{a}
Candidate Correct Answers:
{b}

Your response should use the following format:
Correct: <yes or no>
""".rstrip()


def make_openai_client():
    try:
        return AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
        )
    except KeyError:
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _parse_yes_no(text: str) -> int:
    txt = (text or "").strip()
    m = re.search(r"(?im)^\s*CORRECT\s*:\s*(YES|NO)\b", txt)
    if m:
        return 1 if m.group(1).upper() == "YES" else 0
    return 1 if re.search(r"\byes\b", txt, flags=re.I) else 0


def _clean_api_text(text) -> str:
    return str(text or "").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def api_yesno_judge(questions: List[str], golds: List[object], preds: List[Optional[str]]):
    client = make_openai_client()
    model_name = "gpt-4.1-nano"
    labels = []
    for question, gold_answers, pred in zip(questions, golds, preds):
        pred_text = normalize_for_equiv(_clean_api_text(pred or "I don't know"))

        if isinstance(gold_answers, list):
            gold_text = "; ".join(normalize_for_equiv(str(x)) for x in gold_answers if x)
            prompt_template = NQ_OPEN_MULTI_CORRECTNESS_JUDGE_PROMPT
        else:
            gold_text = normalize_for_equiv(_clean_api_text(gold_answers or ""))
            prompt_template = NQ_OPEN_CORRECTNESS_JUDGE_PROMPT

        if canonicalize_answer(pred_text) and canonicalize_answer(pred_text) == canonicalize_answer(gold_text):
            labels.append(1)
            continue

        prompt = _clean_api_text(prompt_template.format(q=_clean_api_text(question), a=pred_text, b=gold_text))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=20,
        )
        labels.append(_parse_yes_no(response.choices[0].message.content or ""))
    return labels


# ============================================================
# 3) Text extraction / canonicalization
# ============================================================
def extract_completion_text(c) -> str:
    if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
        text = c[0]["content"]
    elif isinstance(c, list) and c and isinstance(c[0], str):
        text = c[0]
    elif isinstance(c, str):
        text = c
    else:
        text = str(c)

    text = (text or "").strip()
    cleaned_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^\s*(cutting knowledge date|today date)\s*:", s, flags=re.I):
            continue
        if re.match(r"^\s*(system|user|assistant)\s*$", s, flags=re.I):
            continue
        s = re.sub(r"\.(system|user|assistant)\s*$", "", s, flags=re.I).strip()
        cleaned_lines.append(s)

    text = "\n".join([x for x in cleaned_lines if x]).strip()
    m = re.search(r"Answer:\s*(.*?)(?:,\s*Confidence:\s*[-+]?\d+(?:\.\d+)?|$)", text, flags=re.I | re.S)
    if m:
        text = m.group(1).strip()
    elif "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
        text = re.sub(r",\s*Confidence:\s*.*$", "", text, flags=re.I | re.S).strip()
    return text


def normalize_for_equiv(text: str, max_chars: int = 2000) -> str:
    t = extract_completion_text(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def canonicalize_answer(text: str) -> str:
    t = normalize_for_equiv(text)
    t = re.sub(r"\s*(explanation\s*:.*)$", "", t, flags=re.I | re.S).strip()
    t = re.sub(r"^\s*(answer\s*:\s*|final\s*answer\s*:\s*|the\s*answer\s*is\s*|answer\s*is\s*)", "", t, flags=re.I).strip()
    t = re.sub(r"[ \t]*\.\s*$", "", t).strip()
    return t


def extract_answer(text: str) -> Optional[str]:
    ans = canonicalize_answer(text)
    return ans if ans else None


def parse_verbal_confidence(text: str):
    match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)", str(text or ""), flags=re.I)
    if not match:
        return None, None
    raw = match.group(1)
    try:
        value = float(raw)
    except Exception:
        return None, raw
    if not (0.0 <= value <= 10.0):
        return None, raw
    conf = value / 10.0
    return float(min(max(conf, 0.0), 1.0)), raw


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "` "]).replace(" ", ""))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(str(s or "")))))).strip()


# ============================================================
# 4) Scoring helpers
# ============================================================
def exact_match_aliases(prediction: Optional[str], aliases: List[str]) -> float:
    if prediction is None:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(alias) for alias in aliases))


# ============================================================
# 5) Prompting
# ============================================================
SYSTEM_PROMPT = (
    "You will get questions. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. "
    "A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. "
    "The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else."
)


def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ============================================================
# 6) Dataset
# ============================================================
raw = load_dataset("google-research-datasets/nq_open")
if "validation" in raw:
    eval_raw = raw["validation"]
elif "test" in raw:
    eval_raw = raw["test"]
else:
    eval_raw = raw[list(raw.keys())[0]]


def convert_to_item(ex):
    ans = ex.get("answer", [])
    if isinstance(ans, str):
        aliases = [ans] if ans.strip() else []
    elif isinstance(ans, list):
        aliases = [a for a in ans if isinstance(a, str) and a.strip()]
    else:
        aliases = []
    return {
        "question": ex.get("question", ""),
        "answers": aliases,
    }


eval_ds = eval_raw.map(convert_to_item)
eval_ds = eval_ds.filter(lambda x: len(x["answers"]) > 0 and bool(x["question"]))
eval_ds = eval_ds.shuffle(seed=SEED)
eval_ds = eval_ds.select(range(min(SAMPLE_SIZE, len(eval_ds))))
print(f"validation size={len(eval_ds)}")


# ============================================================
# 7) Utils: CSV
# ============================================================
def append_row_csv(csv_path: str, header: list[str], row: dict):
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if need_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def append_cases_jsonl(jsonl_path: str, rows: list[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_rows(jsonl_path: str) -> list[dict]:
    rows = []
    if not jsonl_path or (not os.path.exists(jsonl_path)) or os.path.getsize(jsonl_path) == 0:
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# 8) Metrics
# ============================================================
def get_brier(correctness, confidence):
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    return float(np.mean((confidence - correctness) ** 2)) if len(confidence) else float("nan")


def get_ece(correctness, confidence, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            ece += float((np.sum(mask) / len(confidence)) * np.abs(np.mean(confidence[mask]) - np.mean(correctness[mask])))
    return float(ece)


def get_auroc(correctness, confidence):
    if len(np.unique(correctness)) < 2:
        return float("nan")
    return float(roc_auc_score(correctness, confidence))


def summarize_partial(records, correctness, confidences):
    return {
        "accuracy": float(np.mean(correctness)) if correctness else float("nan"),
        "avg_confidence": float(np.mean(confidences)) if confidences else float("nan"),
        "confidence_format_adherence": float(np.mean([r["format_ok"] for r in records])) if records else float("nan"),
        "brier_score": get_brier(correctness, confidences),
        "ece": get_ece(correctness, confidences, ECE_BINS),
        "auroc": get_auroc(correctness, confidences),
    }


# ============================================================
# 9) Generation helpers
# ============================================================
def generate_one(model, tokenizer, messages, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if hasattr(model, "fast_generate") and getattr(model, "vllm_engine", None) is not None:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=1,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            seed=SEED,
        )
        outs = model.fast_generate([prompt], sampling_params=sampling_params)
        return str(outs[0].outputs[0].text)

    model_dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(model_dev)
    temp = float(temperature)
    do_sample = temp > 0.0
    gen_kwargs = dict(
        **enc,
        do_sample=do_sample,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = float(top_p)
    out = model.generate(**gen_kwargs)
    gen_ids = out[0, enc["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def generate_and_verbal_confidence_for_eval(model, tokenizer, question: str):
    raw_text = generate_one(model, tokenizer, build_prompt(question), EVAL_MAX_NEW_TOKENS, 0.0, 1.0)
    answer = extract_answer(raw_text)
    confidence, confidence_raw = parse_verbal_confidence(raw_text)
    return {
        "prediction_text": raw_text,
        "extracted_answer": answer,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "confidence_raw": confidence_raw,
        "format_ok": 1.0 if (answer and confidence is not None) else 0.0,
    }


# ============================================================
# 10) Evaluation
# ============================================================
def score_prediction(prediction: Optional[str], aliases: List[str], question: str) -> float:
    if prediction is None:
        return 0.0
    return float(api_yesno_judge([question], [aliases], [prediction])[0])

def evaluate_model_on_dataset(ds, model, tokenizer, jsonl_path: str | None = None, jsonl_flush_every: int = 20):
    n = len(ds)

    records = []
    judge_q, judge_a, judge_p = [], [], []
    pending_records = []
    pending_write = []
    start_idx = 0

    if jsonl_path is not None:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        existing_records = load_jsonl_rows(jsonl_path)
        if existing_records:
            existing_records = existing_records[:n]
            records.extend(existing_records)
            start_idx = len(existing_records)
            tqdm.write(f"[Resume] loaded {start_idx} existing cases from {jsonl_path}")

    progress = tqdm(range(start_idx, n), desc="RD Eval NQ Open (verbal)", total=n)

    for i in progress:
        ex = ds[i]
        out = generate_and_verbal_confidence_for_eval(model=model, tokenizer=tokenizer, question=ex["question"])
        judge_q.append(ex["question"])
        judge_a.append(ex["answers"])
        judge_p.append(out["extracted_answer"])

        rec = {
            "idx": i,
            "question": ex["question"],
            "gold_answers": ex["answers"],
            "correct": 0,
            **out,
        }
        records.append(rec)
        pending_records.append(rec)

        if len(pending_records) == 5:
            judged = api_yesno_judge(judge_q, judge_a, judge_p)
            for rec_j, lab in zip(pending_records, judged):
                rec_j["correct"] = int(lab)

            pending_write.extend(pending_records)
            if jsonl_path is not None and len(pending_write) >= jsonl_flush_every:
                append_cases_jsonl(jsonl_path, pending_write)
                pending_write = []

            partial_corr = [int(r["correct"]) for r in records]
            partial_conf = [float(r["confidence"]) for r in records]
            summary = summarize_partial(records, partial_corr, partial_conf)
            tqdm.write(
                f"[RD Eval NQ Open] {len(records)}/{n} | acc_so_far={summary['accuracy']:.4f} | "
                f"ece_so_far={summary['ece']:.4f} | "
                f"auroc_so_far={summary['auroc']:.4f} | "
                f"unparsed={int(sum(1.0 - r['format_ok'] for r in records))}/{len(records)}"
            )

            judge_q, judge_a, judge_p = [], [], []
            pending_records = []

    if pending_records:
        judged = api_yesno_judge(judge_q, judge_a, judge_p)
        for rec_j, lab in zip(pending_records, judged):
            rec_j["correct"] = int(lab)
        pending_write.extend(pending_records)

        partial_corr = [int(r["correct"]) for r in records]
        partial_conf = [float(r["confidence"]) for r in records]
        summary = summarize_partial(records, partial_corr, partial_conf)
        tqdm.write(
            f"[RD Eval NQ Open] {len(records)}/{n} | acc_so_far={summary['accuracy']:.4f} | "
            f"ece_so_far={summary['ece']:.4f} | "
            f"auroc_so_far={summary['auroc']:.4f} | "
            f"unparsed={int(sum(1.0 - r['format_ok'] for r in records))}/{len(records)}"
        )

    if jsonl_path is not None and pending_write:
        append_cases_jsonl(jsonl_path, pending_write)

    correctness = [int(r["correct"]) for r in records]
    confidences = [float(r["confidence"]) for r in records]
    metrics = {
        "method": "rd_verbal",
        "dataset": "nq_open",
        "sample_size": len(records),
        **summarize_partial(records, correctness, confidences),
        "avg_completion_length": float(np.mean([len(r["prediction_text"]) for r in records])) if records else float("nan"),
    }
    return metrics, records


# ============================================================
# 11) Save outputs
# ============================================================
csv_header = [
    "method",
    "dataset",
    "sample_size",
    "accuracy",
    "avg_confidence",
    "confidence_format_adherence",
    "brier_score",
    "ece",
    "auroc",
    "avg_completion_length",
    "checkpoint_path",
]


def save_outputs(csv_path: str, metrics: dict, records: list[dict], jsonl_path: str | None = None):
    append_row_csv(csv_path, csv_header, metrics)
    if jsonl_path is None:
        jsonl_path = csv_path[:-4] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 12) Run eval
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", dest="lora_path", default=None,
                        help="Path to LoRA adapter checkpoint. Omit for base-model eval.")
    args = parser.parse_args()
    LORA_ROOT_PATH = args.lora_path

    mode = "ours" if LORA_ROOT_PATH else "BASE"
    output_csv = str(ROOT / f"nq_open_verbal_eval_{mode}.csv")
    jsonl_path = output_csv[:-4] + ".jsonl"

    model, tokenizer = unsloth_load_checkpoint(LORA_ROOT_PATH, BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH)
    metrics, records = evaluate_model_on_dataset(eval_ds, model, tokenizer, jsonl_path=jsonl_path, jsonl_flush_every=20)

    metrics["checkpoint_path"] = LORA_ROOT_PATH or "__BASE__"
    checkpoint_stem = Path((LORA_ROOT_PATH or "__BASE__").rstrip("/")).name
    ckpt_name = checkpoint_stem if LORA_ROOT_PATH else "BASE"

    save_outputs(output_csv, metrics, records, jsonl_path=jsonl_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

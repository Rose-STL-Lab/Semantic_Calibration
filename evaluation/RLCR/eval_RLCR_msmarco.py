#!/usr/bin/env python
# coding: utf-8

# ============================================================
# MSMARCO verbal-confidence evaluation for RLCR
# - ONE generation per question
# - Confidence from <confidence>...</confidence>
# - Standalone script aligned with other RLCR eval methods
# ============================================================

import os
import argparse

# --- HF cache / vLLM env ---
HF_CACHE = "./hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
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
from vllm.lora.request import LoRARequest

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
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
SAMPLE_SIZE = 1000
MAX_PROMPT_LENGTH = 3072
EVAL_MAX_NEW_TOKENS = 1536
ECE_BINS = 10
FLUSH_EVERY = 20


# ============================================================
# 1) Policy Model (Unsloth LoRA / merged checkpoint)
# ============================================================
def unsloth_load_checkpoint(checkpoint_path: Optional[str], base_model: str = BASE_MODEL, max_seq_length: int = MAX_SEQ_LENGTH):
    model_name = base_model
    adapter_path = None
    lora_request = None

    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path

    if checkpoint_path and os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        adapter_path = checkpoint_path
    elif checkpoint_path and os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "config.json")):
        model_name = checkpoint_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if adapter_path is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        model.load_adapter(adapter_path, adapter_name="verbal_eval")
        model.set_adapter("verbal_eval")
        lora_request = LoRARequest("verbal_eval", 1, adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer, lora_request


# ============================================================
# 2) Judge / Correctness
# ============================================================
JUDGE_SYSTEM = "You are a helpful assistant that helps people find information."
JUDGE_PROMPT = """Given a question and two potential answers, you need to analyze whether the two answers convey the same meaning. Respond with "yes" or "no".
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

TRIVIAQA_CORRECTNESS_JUDGE_PROMPT = """Given a question, one generated answer, and a list of candidate correct answers, you need to analyze whether the generated answer matches any candidate correct answer. Respond with "yes" or "no".
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

TRUTHFULQA_CORRECTNESS_JUDGE_PROMPT = """Given a question, one generated answer, and a list of candidate correct answers, you need to analyze whether the generated answer matches any candidate correct answer. Respond with "yes" or "no".
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

MSMARCO_MULTI_CORRECTNESS_JUDGE_PROMPT = """Given a question, one generated answer, and a list of candidate correct answers, you need to analyze whether the generated answer matches any candidate correct answer. Respond with "yes" or "no".
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


def api_yesno_judge(questions: List[str], golds: List[List[str]], preds: List[Optional[str]], prompt_template: str):
    client = make_openai_client()
    model_name = "gpt-4.1-nano"
    labels = []
    for question, gold_answers, pred in zip(questions, golds, preds):
        pred_text = _clean_api_text(pred or "I don't know")
        gold_text = _clean_api_text("; ".join(str(x) for x in (gold_answers or [])))
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
    tag_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if tag_matches:
        return tag_matches[-1].strip()
    line_match = re.search(r"Answer\s*:\s*(.*?)(?:,\s*Confidence\s*:|\n\s*Confidence\s*:|$)", text, flags=re.I | re.S)
    if line_match:
        return line_match.group(1).strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    if "Confidence:" in text:
        text = re.split(r"Confidence\s*:", text, maxsplit=1, flags=re.I)[0].strip().rstrip(',').strip()
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
    raw_text = str(text or "")
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw_text, flags=re.I | re.S)
    ans = canonicalize_answer(matches[-1]) if matches else ""
    return ans if ans else None


def count_text_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return 0
    return int(len(tokenizer(str(text or ""), add_special_tokens=False)["input_ids"]))


_CONFIDENCE_TAG_RE = re.compile(r"<confidence>\s*(.*?)\s*</confidence>", flags=re.I | re.S)

def parse_verbal_confidence(text: str):
    raw_text = str(text or "")
    matches = _CONFIDENCE_TAG_RE.findall(raw_text)
    raw = matches[-1].strip() if matches else None
    if raw is None:
        return None, None
    try:
        value = float(raw)
    except Exception:
        return None, raw
    if not (0.0 <= value <= 1.0):
        return None, raw
    return float(value), raw


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
RLCR_TABC_LONG_PROMPT = (
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
SYSTEM_PROMPT = RLCR_TABC_LONG_PROMPT


def make_prompt_for_truth(q: str, tokenizer=None, system_prompt: str | None = None):
    sys = system_prompt or SYSTEM_PROMPT
    user = f"\n\nPROBLEM: {q}\n\n"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return sys + "\n\nPROBLEM: " + q + "\n\n"


# ============================================================
# 6) Dataset
# ============================================================
raw = load_dataset("microsoft/ms_marco", "v1.1")
if "validation" in raw:
    eval_raw = raw["validation"]
elif "dev" in raw:
    eval_raw = raw["dev"]
elif "test" in raw:
    eval_raw = raw["test"]
else:
    first_split = next(iter(raw.keys()))
    eval_raw = raw[first_split]


def convert_to_item(ex):
    question = str(ex.get("query", "")).strip()
    well_formed = ex.get("wellFormedAnswers") or []
    answers = ex.get("answers") or []
    raw_answers = well_formed if len(well_formed) > 0 else answers
    aliases = [str(a).strip() for a in raw_answers if str(a).strip()]
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "answers": aliases,
        "question": question,
    }


def has_valid_gold_answer(ex):
    ans = ex["answers"]
    return isinstance(ans, list) and len(ans) > 0


eval_ds = eval_raw.map(convert_to_item).filter(has_valid_gold_answer).shuffle(seed=SEED)
eval_ds = eval_ds.select(range(0, min(SAMPLE_SIZE, len(eval_ds))))
print(f"validation size={len(eval_ds)}")


# ============================================================
# 7) Utils: CSV + plotting
# ============================================================
def append_row_csv(csv_path: str, header: list[str], row: dict):
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if need_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# ============================================================
# 8) Metrics
# ============================================================
def expected_calibration_error(confidences, correctness, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    if len(confidences) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        conf_bin = float(confidences[mask].mean())
        acc_bin = float(correctness[mask].mean())
        ece += (n_k / len(confidences)) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_metrics(confidences, correctness):
    metrics = {
        "avg_confidence": float("nan"),
        "brier_score": float("nan"),
        "ece": float("nan"),
        "auroc": float("nan"),
        "accuracy": float("nan"),
        "count": int(len(correctness)),
    }
    if not correctness:
        return metrics
    y_true = np.asarray(correctness, dtype=float)
    y_confs = np.asarray(confidences, dtype=float)
    metrics["avg_confidence"] = float(np.mean(y_confs))
    metrics["brier_score"] = float(np.mean((y_confs - y_true) ** 2))
    metrics["ece"] = expected_calibration_error(y_confs, y_true, n_bins=ECE_BINS)
    metrics["accuracy"] = float(np.mean(y_true))
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_confs))
    return metrics


# ============================================================
# 9) Generation helpers
# ============================================================
@torch.inference_mode()
def generate_one_answer(model, tokenizer, prompt_text: str, lora_request=None, max_new_tokens: int = EVAL_MAX_NEW_TOKENS):
    model.eval()
    def _hf_generate(text: str) -> str:
        model_device = next(model.parameters()).device
        enc = tokenizer(text, return_tensors="pt").to(model_device)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    if hasattr(model, "fast_generate"):
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(max_new_tokens),
                seed=SEED,
            )
            prompt_use = prompt_text
            if tokenizer is not None:
                toks = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                if len(toks) > MAX_PROMPT_LENGTH:
                    prompt_use = tokenizer.decode(toks[-MAX_PROMPT_LENGTH:], skip_special_tokens=False)
            outs = model.fast_generate([prompt_use], sampling_params=sampling_params, lora_request=lora_request)
            raw_text = str(outs[0].outputs[0].text)
        except Exception as e:
            msg = str(e)
            if ("sampling_params" in msg) or ("fast_inference=True" in msg):
                raw_text = _hf_generate(prompt_text)
            else:
                raise
    else:
        raw_text = _hf_generate(prompt_text)
    answer = extract_answer(raw_text)
    confidence, confidence_raw = parse_verbal_confidence(raw_text)
    return {
        "raw_generation": raw_text,
        "prediction": answer,
        "raw_confidence": confidence_raw,
        "conf_hat": float(confidence) if confidence is not None else 0.0,
        "format_ok": int(answer is not None and confidence is not None),
    }


# ============================================================
# 10) Evaluation
# ============================================================
def score_prediction(prediction: Optional[str], aliases: List[str], question: str) -> float:
    if prediction is None:
        return 0.0
    return float(api_yesno_judge([question], [aliases], [prediction], MSMARCO_MULTI_CORRECTNESS_JUDGE_PROMPT)[0])

def load_existing_records(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        return []
    loaded = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                loaded.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return loaded


def append_jsonl_records(jsonl_path: str, new_records: list[dict]):
    if not new_records:
        return
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def evaluate_model_on_dataset(ds, model, tokenizer, lora_request=None, resume_jsonl_path: Optional[str] = None):
    records = load_existing_records(resume_jsonl_path) if resume_jsonl_path else []
    done = min(len(records), len(ds))
    if done > 0:
        tqdm.write(f"[Resume] loaded {done} records from {resume_jsonl_path}")

    pending = []
    progress = tqdm(ds, desc="rlcr:rlcr_verbal:msmarco", total=len(ds))
    for idx, ex in enumerate(progress, start=1):
        if idx <= done:
            continue
        prompt_text = make_prompt_for_truth(ex["question"], tokenizer=tokenizer, system_prompt=SYSTEM_PROMPT)
        gen = generate_one_answer(model, tokenizer, prompt_text, lora_request=lora_request, max_new_tokens=EVAL_MAX_NEW_TOKENS)
        correctness = score_prediction(prediction=gen["prediction"], aliases=ex["answers"], question=ex["question"])
        rec = {
            "question": ex["question"],
            "gold_answers": ex["answers"],
            "raw_generation": gen["raw_generation"],
            "prediction": gen["prediction"],
            "raw_confidence": gen["raw_confidence"],
            "format_ok": int(gen["format_ok"]),
            "correct": float(correctness),
            "conf_hat": float(gen["conf_hat"]),
            "prompt_tokens": count_text_tokens(tokenizer, prompt_text),
            "completion_tokens": count_text_tokens(tokenizer, gen["raw_generation"]),
            "answer_tokens": count_text_tokens(tokenizer, gen["prediction"] or ""),
        }
        rec["total_tokens"] = int(rec["prompt_tokens"] + rec["completion_tokens"])
        records.append(rec)
        pending.append(rec)
        if resume_jsonl_path and (len(pending) >= FLUSH_EVERY):
            append_jsonl_records(resume_jsonl_path, pending)
            pending = []
        if idx % 5 == 0:
            part = records
            metrics_so_far = compute_metrics([r["conf_hat"] for r in part], [r["correct"] for r in part])
            tqdm.write(
                f"[RLCR Eval MSMARCO] {len(part)}/{len(ds)} | acc_so_far={metrics_so_far['accuracy']:.4f} | "
                f"ece_so_far={metrics_so_far['ece']:.4f} | "
                f"auroc_so_far={metrics_so_far['auroc']:.4f} | "
                f"unparsed={sum(1 - r['format_ok'] for r in part)}/{len(part)}"
            )
    if resume_jsonl_path and pending:
        append_jsonl_records(resume_jsonl_path, pending)
    metrics = {
        "method": "rlcr_verbal",
        "dataset": "msmarco",
        "sample_size": len(records),
        **compute_metrics([r["conf_hat"] for r in records], [r["correct"] for r in records]),
        "avg_completion_length": float(np.mean([len(r["raw_generation"]) for r in records])) if records else float("nan"),
    }
    return metrics, records


# ============================================================
# 11) Save outputs
# ============================================================
csv_header = [
    "method",
    "dataset",
    "sample_size",
    "avg_completion_length",
    "avg_confidence",
    "brier_score",
    "ece",
    "auroc",
    "accuracy",
    "count",
    "checkpoint_path",
]


def save_outputs(csv_path: str, metrics: dict, records: list[dict], write_jsonl: bool = True):
    append_row_csv(csv_path, csv_header, metrics)
    if write_jsonl:
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
    output_csv = str(ROOT / f"msmarco_verbal_eval_{mode}.csv")
    output_jsonl = output_csv[:-4] + ".jsonl"

    dataset = eval_ds
    model, tokenizer, lora_request = unsloth_load_checkpoint(LORA_ROOT_PATH, BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH)
    metrics, records = evaluate_model_on_dataset(
        dataset,
        model,
        tokenizer,
        lora_request=lora_request,
        resume_jsonl_path=output_jsonl,
    )

    ckpt_name = Path((LORA_ROOT_PATH or "BASE").rstrip("/")).name if LORA_ROOT_PATH else "BASE"
    metrics["checkpoint_path"] = LORA_ROOT_PATH or "__BASE__"
    save_outputs(output_csv, metrics, records, write_jsonl=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

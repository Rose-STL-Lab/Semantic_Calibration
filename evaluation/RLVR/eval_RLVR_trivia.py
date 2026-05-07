#!/usr/bin/env python
# coding: utf-8

# ============================================================
# TriviaQA evaluation for BASE + all LoRA checkpoints
# Judge: GPT API (OpenAI-compatible chat completions)
# - ONE judge prompt (used for BOTH correctness + SAME)
# - One-line output: "Correct: YES" or "Correct: NO"
# - Canonicalize + shortcut (A_use == B_use => YES)
# - LEFT padding for decoder-only models
# - CPU-friendly LRU cache (hash keys) for SAME + correctness
# - Minimal changes to your original eval logic: metrics/plot/ckpt enumeration preserved
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
import random
import numpy as np
import torch
import re, time, csv, hashlib, json, string
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm
from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau


# ===== 0. Seed / Device =====
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
# 1) Policy Model (Unsloth LoRA)
# ============================================================
max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)

# --- FIX: decoder-only should use LEFT padding for correct generation ---
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing=False,
    random_state=3407,
)


# ============================================================
# 2) Judge: GPT API (ONE prompt for SAME + one TriviaQA correctness prompt)
# ============================================================
JUDGE_MODEL_ID = "gpt-4.1-nano"
JUDGE_API_MAX_RETRIES = 2
JUDGE_API_SLEEP_SEC = 1.5
JUDGE_API_MAX_WORKERS = 6

def _make_judge_client():
    try:
        return AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
        )
    except KeyError:
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

judge_client = _make_judge_client()

# Same prompt as your train/correctness/calibration version
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

JUDGE_SYSTEM = "You are a helpful assistant that helps people find information."

# --- stats ---
JUDGE_STATS = {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []}
SAME_STATS  = {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []}


# ============================================================
# 3) Text extraction / canonicalization (same as current train.py variant)
# ============================================================
_CANON_PREFIX_RE = re.compile(
    r"^\s*(answer\s*:\s*|final\s*answer\s*:\s*|the\s*answer\s*is\s*|answer\s*is\s*)",
    flags=re.I,
)
_CANON_TAIL_RE = re.compile(r"\s*(explanation\s*:.*)$", flags=re.I | re.S)

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
    m = re.search(r"(?is)final\s*answer\s*:\s*(.+)$", text)
    if m:
        return m.group(1).strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text

def normalize_for_equiv(text: str, max_chars: int = 2000) -> str:
    t = extract_completion_text(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]

def normalize_answer(text: str) -> str:
    t = str(text or "").lower().replace("_", " ")
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(rf"[{re.escape(string.punctuation)}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonicalize_answer(text: str) -> str:
    t = normalize_for_equiv(text)
    t = _CANON_TAIL_RE.sub("", t).strip()
    t = _CANON_PREFIX_RE.sub("", t).strip()
    t = re.sub(r"[ \t]*\.\s*$", "", t).strip()
    return t


# ============================================================
# 4) LRU cache (hash keys) for judge calls
# ============================================================
class LRUCache(OrderedDict):
    def __init__(self, max_size=200_000):
        super().__init__()
        self.max_size = int(max_size)

    def get_item(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def set_item(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def clear_all(self):
        super().clear()

# cache size for eval (SAME dominates)
JUDGE_CACHE_MAX = 200_000
JUDGE_CACHE = LRUCache(max_size=JUDGE_CACHE_MAX)

def _hash_key(q: str, a_use: str, b_use: str) -> bytes:
    s = (q or "") + "\n<|A|>\n" + (a_use or "") + "\n<|B|>\n" + (b_use or "")
    return hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16).digest()

def _parse_yesno_one_line(txt: str):
    if not txt:
        return None
    m = re.search(r"(?im)^\s*CORRECT\s*:\s*(YES|NO)\b", txt)
    if m:
        return 1 if m.group(1).upper() == "YES" else 0
    m = re.search(r"\b(YES|NO)\b", (txt or "").upper())
    if m:
        return 1 if m.group(1).upper() == "YES" else 0
    return None

# Defense against prompt / format injection in user-supplied content (model
# outputs and dataset text) that flows into the judge's user message.
_CHAT_TEMPLATE_TOKEN_RE = re.compile(r"<\|[^|>\s]{1,80}\|>")
_JUDGE_OUTPUT_MIMIC_RE = re.compile(r"\bcorrect\s*:\s*(?:yes|no)\b", flags=re.I)

def _sanitize_for_judge(text, max_chars: int = 1000) -> str:
    t = "" if text is None else str(text)
    t = _CHAT_TEMPLATE_TOKEN_RE.sub(" ", t)
    t = _JUDGE_OUTPUT_MIMIC_RE.sub("[redacted]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def _build_judge_input(q: str, a: str, b: str, prompt_template: str = JUDGE_PROMPT):
    user = prompt_template.format(
        q=_sanitize_for_judge(q),
        a=_sanitize_for_judge(a),
        b=_sanitize_for_judge(b),
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _judge_once_api(messages, max_new_tokens: int = 16) -> str:
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL_ID,
        messages=messages,
        temperature=0.0,
        max_tokens=int(max_new_tokens),
        top_p=1.0,
        n=1,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


@torch.no_grad()
def judge_yesno_batched(
    preds,
    golds,
    questions,
    prompt_template: str = JUDGE_PROMPT,
    max_new_tokens: int = 16,
    batch_size: int = 16,
    max_retries: int = 2,
    use_cache: bool = True,
    shortcut: bool = True,
    stats: dict | None = None,
):
    """
    Batched judge for pairwise similarity:
      - ONE judge prompt
      - canonicalize + shortcut
      - LRU cache on (q, a_use, b_use)
      - retry only unparsed
    Returns: labels(list[int]), unparsed(int)
    """
    assert len(preds) == len(golds) == len(questions)
    n = len(preds)
    labels = [0] * n
    need_idx = []
    inputs = [None] * n
    keys = [None] * n
    a_uses = [None] * n
    b_uses = [None] * n

    if stats is None:
        stats = {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []}

    for i, (q, a, b) in enumerate(zip(questions, preds, golds)):
        q = "" if q is None else str(q)

        a_use = canonicalize_answer(a)
        b_use = canonicalize_answer(b)
        a_uses[i] = a_use
        b_uses[i] = b_use

        if shortcut and a_use != "" and (a_use == b_use):
            labels[i] = 1
            continue

        key = _hash_key(q, a_use, b_use)
        keys[i] = key

        if use_cache:
            cached = JUDGE_CACHE.get_item(key)
            if cached is not None:
                labels[i] = int(cached)
                continue

        # Build chat-templated judge input
        inputs[i] = _build_judge_input(q=q, a=a_use, b=b_use, prompt_template=prompt_template)
        need_idx.append(i)

    if not need_idx:
        return labels, 0

    for _ in range(int(max_retries)):
        if not need_idx:
            break

        new_need = []
        for s in range(0, len(need_idx), int(batch_size)):
            batch_ids = need_idx[s:s + int(batch_size)]

            def _judge_one(idx):
                txt = ""
                for attempt in range(int(max_retries) + 1):
                    try:
                        txt = _judge_once_api(inputs[idx], max_new_tokens=max_new_tokens)
                        break
                    except Exception:
                        if attempt >= int(max_retries):
                            txt = ""
                            break
                        time.sleep(JUDGE_API_SLEEP_SEC)
                return idx, _parse_yesno_one_line(txt)

            max_workers = min(int(JUDGE_API_MAX_WORKERS), len(batch_ids)) if batch_ids else 1
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_judge_one, idx): idx for idx in batch_ids}
                for fut in as_completed(futures):
                    idx, y = fut.result()
                    stats["total"] += 1

                    if y is None:
                        new_need.append(idx)
                    else:
                        labels[idx] = int(y)
                        if y == 1:
                            stats["parsed_yes"] += 1
                        else:
                            stats["parsed_no"] += 1
                        if use_cache:
                            JUDGE_CACHE.set_item(keys[idx], int(y))

        need_idx = new_need

    unparsed = 0
    if need_idx:
        unparsed = len(need_idx)
        stats["unparsed"] += unparsed
        # keep a few examples
        for idx in need_idx[: max(0, 10 - len(stats["examples"]))]:
            stats["examples"].append({
                "q": (questions[idx] or "")[:120],
                "a_use": (a_uses[idx] or "")[:200],
                "b_use": (b_uses[idx] or "")[:200],
                "judge_out": "<UNPARSED>",
            })
        for idx in need_idx:
            labels[idx] = 0
            if use_cache:
                JUDGE_CACHE.set_item(keys[idx], 0)

    return labels, unparsed


# ============================================================
# 5) Semantic Entropy via judge clustering (ONE prompt)
# ============================================================
def get_semantic_ids_batched_llmjudge(
    strings_list,
    question: str,
    strict_bidirectional: bool = True,
    max_new_tokens=16,
    batch_size=16,
):
    """
    Cluster strings by judge on similarity:
      - compare i->j and j->i (bidirectional)
      - strict: both YES => same cluster
    """
    n = len(strings_list)
    if n <= 1:
        return list(range(n))

    a_list, b_list, pair_map = [], [], {}
    for i in range(n):
        for j in range(i + 1, n):
            idx_f = len(a_list)
            a_list.append(strings_list[i])
            b_list.append(strings_list[j])

            idx_r = len(a_list)
            a_list.append(strings_list[j])
            b_list.append(strings_list[i])

            pair_map[(i, j)] = (idx_f, idx_r)

    q_list = [question] * len(a_list)
    same01, _unp = judge_yesno_batched(
        preds=a_list,
        golds=b_list,
        questions=q_list,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_retries=2,
        use_cache=True,
        shortcut=True,
        stats=SAME_STATS,
    )

    semantic_set_ids = [-1] * n
    next_id = 0
    for i in range(n):
        if semantic_set_ids[i] != -1:
            continue
        semantic_set_ids[i] = next_id
        for j in range(i + 1, n):
            if semantic_set_ids[j] != -1:
                continue
            idx_f, idx_r = pair_map[(i, j)]
            s_f = int(same01[idx_f])
            s_r = int(same01[idx_r])
            eq = (s_f == 1 and s_r == 1) if strict_bidirectional else (s_f == 1 or s_r == 1)
            if eq:
                semantic_set_ids[j] = next_id
        next_id += 1

    return semantic_set_ids


def cluster_assignment_entropy(semantic_ids):
    n = len(semantic_ids)
    if n <= 1:
        return 0.0
    counts = np.bincount(semantic_ids)
    p = counts / n
    ent = -(p * np.log(p + 1e-12)).sum()
    return float(ent)


def compute_semantic_entropy(samples, question: str):
    if not samples:
        return 0.0
    semantic_ids = get_semantic_ids_batched_llmjudge(
        samples,
        question=question,
        strict_bidirectional=True,
        max_new_tokens=16,
        batch_size=16,
    )
    return cluster_assignment_entropy(semantic_ids)


# ============================================================
# 6) Prompting
# ============================================================
SYSTEM_PROMPT = (
    "Provide a concise and direct answer to the question. "
    "Avoid unnecessary explanations or additional text."
)

_DEBUG_QUESTION = "Who wrote Hamlet?"
_DEBUG_MSGS = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": _DEBUG_QUESTION},
]
print("[PromptDebug] has_apply_chat_template=", hasattr(tokenizer, "apply_chat_template"))
print("[PromptDebug] has_chat_template=", bool(getattr(tokenizer, "chat_template", None)))
if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
    _debug_prompt = tokenizer.apply_chat_template(_DEBUG_MSGS, tokenize=False, add_generation_prompt=True)
else:
    _debug_prompt = SYSTEM_PROMPT + "\n\n" + _DEBUG_QUESTION + "\n\nAnswer:"
print("[PromptDebug] prompt_preview=", _debug_prompt[:500])
print("[PromptDebug] contains_Answer=", "Answer:" in _debug_prompt)

def render_generation_prompt(prompt_messages, tokenizer):
    if isinstance(prompt_messages, list) and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt_messages, list):
        chunks = []
        for msg in prompt_messages:
            if isinstance(msg, dict):
                chunks.append(str(msg.get("content", "")))
            else:
                chunks.append(str(msg))
        return "\n\n".join(chunks).strip() + "\n\nAnswer:"
    return str(prompt_messages)


# ============================================================
# 7) Dataset (TriviaQA)
# ============================================================
raw = load_dataset("mandarjoshi/trivia_qa", "rc.web.nocontext")
train_raw = raw["train"]
validation_raw = raw["validation"]

def convert_to_item(ex):
    aliases = list(ex["answer"].get("normalized_aliases", []))
    aliases.extend([normalize_answer(a) for a in ex["answer"].get("human_answers", [])])
    aliases = [a for a in aliases if a]
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "answer": aliases,
        "question": ex["question"],
        "meta": {
            "question_id": ex.get("question_id", ""),
            "question_source": ex.get("question_source", ""),
        },
    }

validation_ds = validation_raw.map(convert_to_item).shuffle(seed=SEED)

validation_ds = validation_ds.select(range(0, min(1000, len(validation_ds))))
print(f"validation size={len(validation_ds)}")


# ============================================================
# 8) Utils: CSV + plotting
# ============================================================
def append_row_csv(csv_path: str, header: list[str], row: dict):
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


def save_cases_jsonl(jsonl_path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
# 9) Metrics
# ============================================================
EVAL_MAX_NEW_TOKENS = 256
ECE_BINS = 10

def expected_calibration_error(confidences, correctness, n_bins=ECE_BINS):
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    N = len(confidences)
    if N == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        n_k = mask.sum()
        if n_k == 0:
            continue
        conf_bin = float(confidences[mask].mean())
        acc_bin = float(correctness[mask].mean())
        ece += (n_k / N) * abs(acc_bin - conf_bin)
    return float(ece)

def brier_soft(p_hats, confs):
    p = np.asarray(p_hats, dtype=float)
    c = np.asarray(confs, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(np.mean((c - p) ** 2))

def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = spearmanr(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)

def safe_kendall(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = kendalltau(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)

def safe_auc(y_true, y_score, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=float)
    if not np.all((y_true == 0) | (y_true == 1)):
        y_true = (y_true >= threshold).astype(int)
    else:
        y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return None
    try:
        return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    except Exception:
        return None


# ============================================================
# 10) Generation helpers
# ============================================================
@torch.inference_mode()
def generate_k_and_semantic_confidence_for_eval(
    model, tokenizer, prompt: str,
    lora_request=None,
    k: int = 5,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.95,
    gold_answer: list[str] | None = None,
    question: str | None = None,
):
    model.eval()

    samples = []
    if hasattr(model, "fast_generate") and lora_request is not None:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=int(k),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            seed=SEED,
        )
        outs = model.fast_generate([prompt], sampling_params=sampling_params, lora_request=lora_request)
        samples = [extract_completion_text(o.text) for o in outs[0].outputs]
    else:
        model_dev = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt").to(model_dev)
        in_len = enc["input_ids"].shape[-1]
        for _ in range(k):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            gen_ids = out[0, in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            samples.append(extract_completion_text(text))

    # semantic uncertainty -> confidence
    q_for_entropy = question or ""
    H = compute_semantic_entropy(samples, question=q_for_entropy)
    conf_hat = float(np.exp(-H))

    # correctness for each sample (compare generated answer to candidate correct answers)
    labels_k = None
    if gold_answer is not None:
        gold_clean = "; ".join(normalize_for_equiv(g) for g in gold_answer if g)
        samples_clean = [normalize_for_equiv(s) for s in samples]
        q_list = [question or ""] * len(samples_clean)

        labels_k, _unp = judge_yesno_batched(
            preds=samples_clean,
            golds=[gold_clean] * len(samples_clean),
            questions=q_list,
            prompt_template=TRIVIAQA_CORRECTNESS_JUDGE_PROMPT,
            max_new_tokens=16,
            batch_size=16,
            max_retries=2,
            use_cache=True,
            shortcut=True,
            stats=JUDGE_STATS,
        )

    return {"samples": samples, "conf_hat": conf_hat, "labels_k": labels_k}


@torch.inference_mode()
def evaluate_model_on_dataset(
    ds, model, tokenizer,
    lora_request=None,
    k: int = 5,
    limit: int | None = None,
    show_n: int = 5,
    jsonl_path: str | None = None,
    jsonl_flush_every: int = 50,
):
    total = 0
    correct_sum = 0.0
    confs, p_hats = [], []
    cases = []
    pending_cases = []

    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()
    start_idx = 0

    if jsonl_path is not None:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        existing_cases = load_jsonl_rows(jsonl_path)
        if existing_cases:
            existing_cases = existing_cases[:n]
            cases.extend(existing_cases)
            start_idx = len(existing_cases)
            for row in existing_cases:
                p_hat = float(row.get("p_hat", 0.0))
                conf_hat = float(row.get("conf_hat", 0.0))
                total += 1
                correct_sum += p_hat
                p_hats.append(p_hat)
                confs.append(conf_hat)
            print(f"[Resume] loaded {start_idx} existing cases from {jsonl_path}")

    for i in tqdm(range(start_idx, n), desc=f"CSR Eval TriviaQA (mean@{k})", total=n):
        ex = ds[i]
        q = ex["question"]
        gold = ex["answer"]
        prompt = render_generation_prompt(ex["prompt"], tokenizer)

        out = generate_k_and_semantic_confidence_for_eval(
            model, tokenizer, prompt,
            lora_request=lora_request,
            k=k,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            gold_answer=gold,
            question=q,
        )

        samples = out["samples"]
        conf_hat = out["conf_hat"]
        labels_k = out["labels_k"] or [0] * len(samples)

        p_hat = float(sum(labels_k) / max(1, len(labels_k)))

        total += 1
        correct_sum += p_hat
        confs.append(conf_hat)
        p_hats.append(p_hat)

        case_row = {
            "question": q,
            "gold_answers": gold if isinstance(gold, list) else [gold],
            "samples": samples,
            "labels_k": labels_k,
            "p_hat": p_hat,
            "conf_hat": conf_hat,
        }
        cases.append(case_row)
        pending_cases.append(case_row)

        if jsonl_path is not None and len(pending_cases) >= jsonl_flush_every:
            append_cases_jsonl(jsonl_path, pending_cases)
            pending_cases = []

        if total % 5 == 0:
            acc_so_far = correct_sum / max(1, total)
            ece_so_far = expected_calibration_error(confs, p_hats, n_bins=ECE_BINS)
            auroc_so_far = safe_auc(p_hats, confs)
            tqdm.write(
                f"[CSR Eval TriviaQA] {total}/{n} | acc_so_far={acc_so_far:.4f} | "
                f"ece_so_far={ece_so_far:.4f} | "
                f"auroc_so_far={auroc_so_far if auroc_so_far is not None else 'NA'}"
            )

    acc = correct_sum / max(1, total)
    ece = expected_calibration_error(confs, p_hats, n_bins=ECE_BINS)
    auc_conf = safe_auc(p_hats, confs)
    brier = brier_soft(p_hats, confs)
    spearman = safe_spearman(confs, p_hats)
    kendall = safe_kendall(confs, p_hats)

    print(
        f"[CSR Eval TriviaQA] N={total}  Acc_mean@{k}={acc:.4f}  "
        f"ECE_soft({ECE_BINS}-bin)={ece:.4f}  "
        f"Brier_soft(MSE)={brier:.4f}  "
        f"Spearman={spearman if spearman is not None else 'NA'}  "
        f"Kendall={kendall if kendall is not None else 'NA'}  "
        f"AUROC(conf)={auc_conf if auc_conf is not None else 'NA'}\n"
    )

    if jsonl_path is not None and pending_cases:
        append_cases_jsonl(jsonl_path, pending_cases)

    return {
        "n": total,
        "accuracy_mean": acc,
        "ece_soft": ece,
        "brier_soft": brier,
        "spearman": spearman,
        "kendall": kendall,
        "auroc_conf": auc_conf,
        "confs": confs,
        "p_hats": p_hats,
        "cases": cases,
    }


# ============================================================
# 11) Checkpoint enumeration
# ============================================================
def list_checkpoints_sorted(root_or_ckpt_path: str):
    p = (root_or_ckpt_path or "").rstrip("/")
    if re.search(r"/checkpoint-\d+$", p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"[list_checkpoints_sorted] Not a directory: {p}")

    ckpts = []
    for name in os.listdir(p):
        m = re.match(r"^checkpoint-(\d+)$", name)
        if not m:
            continue
        step = int(m.group(1))
        full = os.path.join(p, name)
        if os.path.isdir(full):
            ckpts.append((step, full))

    ckpts.sort(key=lambda x: x[0])
    return [path for _, path in ckpts]


# ============================================================
# 12) Run eval for BASE + all checkpoints
# ============================================================
num_rollout = 8
mode = "RLVR"  # distinguish filename

# Path to a single LoRA checkpoint, or to a directory containing CHECKPOINT_NAMES.
_parser = argparse.ArgumentParser()
_parser.add_argument("--lora-path", dest="lora_path", default=None,
                     help="Path to LoRA adapter checkpoint. Omit for base-model eval.")
_args, _ = _parser.parse_known_args()
LORA_ROOT_PATH = _args.lora_path  # None => base eval

chosen_ds = validation_ds  # aligned with training-time validation

OUT_CSV = os.path.join(os.path.dirname(__file__), f"triviaqa_eval_{mode}.csv")
ECE_BINS = 10

csv_header = [
    "timestamp",
    "split",
    "ckpt_path",
    "step",
    "k",
    "n",
    "accuracy_mean",
    "ece_soft",
    "brier_soft",
    "spearman",
    "kendall",
    "auroc_conf",
    "judge_total",
    "judge_unparsed",
    "judge_unparsed_rate",
    "same_total",
    "same_unparsed",
    "same_unparsed_rate",
    "judge_cache_size",
]

ckpt_paths = list_checkpoints_sorted(LORA_ROOT_PATH) if LORA_ROOT_PATH else []

if LORA_ROOT_PATH:
    run_items = [(p, p) for p in ckpt_paths]
else:
    run_items = [("BASE", None)]


print("[EVAL] will run:", len(run_items), "models")
print("[EVAL] BASE + checkpoints:", [os.path.basename(x[0]) for x in run_items[:5]], "...")

for run_tag, lora_dir in run_items:
    print("\n" + "=" * 110)

    # reset stats per model
    JUDGE_STATS.update({"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []})
    SAME_STATS.update({"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0, "examples": []})

    # keep cache across checkpoints for speed (recommended)
    # If you want fully independent per-ckpt stats, uncomment:
    # JUDGE_CACHE.clear_all()

    if lora_dir is None:
        print("[EVAL] Running BASE (no LoRA)")
        lora_request = None
        ckpt_name = "BASE"
        step = 0
        ckpt_path_for_csv = "__BASE__"
    else:
        print("[EVAL] Loading LoRA:", lora_dir)
        lora_request = model.load_lora(lora_dir)
        ckpt_name = os.path.basename(lora_dir.rstrip("/"))
        m = re.search(r"checkpoint-(\d+)", ckpt_name)
        step = int(m.group(1)) if m else -1
        ckpt_path_for_csv = lora_dir

    model = model.eval()

    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", ckpt_name)
    jsonl_path = os.path.join(os.path.dirname(__file__), f"triviaqa_{safe_tag}.jsonl")

    ft_metrics = evaluate_model_on_dataset(
        ds=chosen_ds,
        model=model,
        tokenizer=tokenizer,
        k=num_rollout,
        lora_request=lora_request,
        limit=len(chosen_ds),
        show_n=2,
        jsonl_path=jsonl_path,
        jsonl_flush_every=20,
    )

    tot = int(JUDGE_STATS["total"])
    unp = int(JUDGE_STATS["unparsed"])
    unp_rate = float(unp / max(1, tot))

    stot = int(SAME_STATS["total"])
    sunp = int(SAME_STATS["unparsed"])
    sunp_rate = float(sunp / max(1, stot))

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    append_row_csv(OUT_CSV, csv_header, {
        "timestamp": ts,
        "split": "validation",
        "ckpt_path": ckpt_path_for_csv,
        "step": step,
        "k": num_rollout,
        "n": ft_metrics["n"],
        "accuracy_mean": ft_metrics["accuracy_mean"],
        "ece_soft": ft_metrics["ece_soft"],
        "brier_soft": ft_metrics["brier_soft"],
        "spearman": "" if ft_metrics["spearman"] is None else ft_metrics["spearman"],
        "kendall": "" if ft_metrics["kendall"] is None else ft_metrics["kendall"],
        "auroc_conf": "" if ft_metrics["auroc_conf"] is None else ft_metrics["auroc_conf"],
        "judge_total": tot,
        "judge_unparsed": unp,
        "judge_unparsed_rate": unp_rate,
        "same_total": stot,
        "same_unparsed": sunp,
        "same_unparsed_rate": sunp_rate,
        "judge_cache_size": len(JUDGE_CACHE),
    })

    print(f"[Saved] metrics -> {OUT_CSV}")
    print(f"[Saved] jsonl   -> {jsonl_path}")

    if unp > 0:
        ex0 = JUDGE_STATS["examples"][0] if JUDGE_STATS["examples"] else None
        print(f"[Judge correctness] unparsed={unp}/{tot} ({unp_rate:.4f}) example0={ex0}")
    if sunp > 0:
        ex0 = SAME_STATS["examples"][0] if SAME_STATS["examples"] else None
        print(f"[Judge SAME] unparsed={sunp}/{stot} ({sunp_rate:.4f}) example0={ex0}")

    print(f"[JUDGE_CACHE] size={len(JUDGE_CACHE)}/{JUDGE_CACHE.max_size}")
    print("=" * 110)

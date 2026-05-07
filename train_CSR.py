#!/usr/bin/env python
# coding: utf-8

# ============================================================
# GRPO training (HotpotQA) - calibration reward and correctness reward
# - Judge = GPT API (OpenAI-compatible chat completions)
# - Single judge format: one-line "Correct: <YES or NO>"
# - Canonicalize + shortcut (A_use == B_use => YES)
# - CPU-friendly LRU cache (hash keys) + periodic stats print + optional clear
# ============================================================

# ===== 0. Imports & Env =====
import argparse
import os, re, time, csv, math, numbers, random, hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)

from unsloth import FastLanguageModel
import unsloth.models.llama as u_llama
from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# HF / vLLM cache env
# -----------------------------
HF_CACHE = "./hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# -----------------------------
# Seed / device
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--cali-w", type=float, default=0.2, dest="cali_w")
ARGS = parser.parse_args()

print("device_count =", torch.cuda.device_count())
print("current_device =", torch.cuda.current_device())
print("name0 =", torch.cuda.get_device_name(0))

# ============================================================
# (NEW) Reward weights + linear ramp schedule
# ============================================================
CALI_W_START = 0.1                  # start calibration weight
CALI_W_END   = float(ARGS.cali_w)  # end calibration weight
CORR_W       = 1.0                  # correctness fixed weight

CALI_W = CALI_W_START  # will be updated during training
CALI_W_END_TAG = format(CALI_W_END, "g")
experiment = f"CSR_{CALI_W_END_TAG}"
print(f"Experiment name = {experiment}")
print(f"CALI_W_START = {CALI_W_START:.4f}, CALI_W_END = {CALI_W_END:.4f}")

def cali_weight_schedule(progress: float) -> float:
    """
    Linear ramp:
      progress in [0,1]
      CALI_W = start + progress*(end-start)
    """
    p = float(min(max(progress, 0.0), 1.0))
    return float(CALI_W_START + p * (CALI_W_END - CALI_W_START))


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
    use_gradient_checkpointing="unsloth",
    random_state=3407,
).to(DEVICE)

# ============================================================
# 2) Judge API client + prompt (single format)
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

JUDGE_PROMPT = """Given a question and two potential answers, you need to analyze whether the two answers convey the same meaning. Respond with "yes" or "no".
Your response should be completely based on the similarity between the two answers.
DO NOT use your own knowledge base when do the comparison.

Question:
{q}
Response 1:
{a}
Response 2:
{b}

Your response should use the following format:
Correct: <yes or no>
""".rstrip()

# --- Judge system prompt ---
JUDGE_SYSTEM = "You are a helpful assistant that helps people find information."

# ============================================================
# 3) CPU-friendly LRU cache (hash keys) + stats
# ============================================================
class LRUCache(OrderedDict):
    """
    CPU-friendly LRU with bounded size.
    Keys are small hashes (bytes) to reduce RAM.
    """
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


# ---- tune these ----
JUDGE_CACHE_MAX = 120_000
CACHE_PRINT_EVERY = 25       # print hit-rate every N steps
CACHE_CLEAR_EVERY = 0        # set >0 if you really want periodic clear (0 = never)
CUDA_EMPTY_CACHE_EVERY = 0   # set >0 if you want torch.cuda.empty_cache() periodically (0 = never)

JUDGE_CACHE = LRUCache(max_size=JUDGE_CACHE_MAX)

CACHE_STATS = {
    "hit": 0,
    "miss": 0,
    "parsed": 0,
    "unparsed": 0,
}

def _hash_key(q: str, a_use: str, b_use: str) -> bytes:
    s = (q or "") + "\n<|A|>\n" + (a_use or "") + "\n<|B|>\n" + (b_use or "")
    return hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16).digest()

def _parse_yesno_one_line(txt: str):
    """
    Prefer 'Correct: YES/NO' line; fallback to any YES/NO token.
    Return 1/0 or None if unparsed.
    """
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


def _build_judge_input(q: str, a: str, b: str):
    user = JUDGE_PROMPT.format(
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


# ============================================================
# 4) Text extraction / canonicalization (same as current train.py variant)
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
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text

def normalize_for_equiv(text: str, max_chars: int = 2000) -> str:
    t = extract_completion_text(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]

def canonicalize_answer(text: str) -> str:
    t = normalize_for_equiv(text)
    t = _CANON_TAIL_RE.sub("", t).strip()
    t = _CANON_PREFIX_RE.sub("", t).strip()
    t = re.sub(r"[ \t]*\.\s*$", "", t).strip()
    return t

def _extract_user_question_from_prompt(p):
    if isinstance(p, list):
        for msg in reversed(p):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
    return str(p)

# ============================================================
# 5) API judge with LRU cache + shortcut
# ============================================================
@torch.no_grad()
def judge_yesno_batched(
    preds,
    golds,
    questions,
    max_new_tokens: int = 16,
    batch_size: int = 16,
    max_retries: int = 2,
    use_cache: bool = True,
    shortcut: bool = True,
):
    """
    preds/golds/questions: list[str]
    Return: labels(list[int]), unparsed(int)
    """
    assert len(preds) == len(golds) == len(questions)
    n = len(preds)
    labels = [0] * n
    need_idx = []
    prompts = [None] * n
    keys = [None] * n

    for i, (q, a, b) in enumerate(zip(questions, preds, golds)):
        q = "" if q is None else str(q)
        a_use = canonicalize_answer(a)
        b_use = canonicalize_answer(b)

        if shortcut and a_use != "" and (a_use == b_use):
            labels[i] = 1
            continue

        key = _hash_key(q, a_use, b_use)
        keys[i] = key

        if use_cache:
            cached = JUDGE_CACHE.get_item(key)
            if cached is not None:
                CACHE_STATS["hit"] += 1
                labels[i] = int(cached)
                continue

        CACHE_STATS["miss"] += 1
        prompts[i] = _build_judge_input(q, a_use, b_use)
        need_idx.append(i)

    if not need_idx:
        return labels, 0

    # retry only for unparsed
    for _ in range(int(max_retries)):
        if not need_idx:
            break

        new_need = []
        for s in range(0, len(need_idx), int(batch_size)):
            batch_ids = need_idx[s:s+int(batch_size)]

            def _judge_one(idx):
                txt = ""
                for attempt in range(int(max_retries) + 1):
                    try:
                        txt = _judge_once_api(prompts[idx], max_new_tokens=max_new_tokens)
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

                    if y is None:
                        new_need.append(idx)
                    else:
                        CACHE_STATS["parsed"] += 1
                        labels[idx] = int(y)
                        if use_cache:
                            JUDGE_CACHE.set_item(keys[idx], int(y))

        need_idx = new_need

    # remaining unparsed => 0
    unparsed = 0
    if need_idx:
        unparsed = len(need_idx)
        CACHE_STATS["unparsed"] += unparsed
        for idx in need_idx:
            labels[idx] = 0
            if use_cache:
                JUDGE_CACHE.set_item(keys[idx], 0)

    return labels, unparsed


# ============================================================
# 6) Dataset (HotpotQA)
# ============================================================
SYSTEM_PROMPT = (
    "Provide a concise and direct answer to the question. "
    "Avoid unnecessary explanations or additional text."
)


raw = load_dataset("hotpot_qa", "distractor")
train_raw = raw["train"]
validation_raw  = raw["validation"]

def convert_to_item(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "answer": ex.get("answer", ""),
        "question": ex["question"],
        "meta": {
            "id": ex.get("id", ""),
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
        },
    }

train_ds = train_raw.map(convert_to_item).shuffle(seed=SEED)
validation_ds  = validation_raw.map(convert_to_item).shuffle(seed=SEED)

train_ds = train_ds.select(range(0, min(10000, len(train_ds))))
validation_ds  = validation_ds.select(range(0, min(1000, len(validation_ds))))
print(f"train size={len(train_ds)}, validation size={len(validation_ds)}")


# ============================================================
# 7) Reward utils
# ============================================================
def _group_by_prompt(prompts):
    groups = OrderedDict()
    for i, p in enumerate(prompts):
        groups.setdefault(str(p), []).append(i)
    return groups

def _gold_for_row(answer, i, B, N):
    if isinstance(answer, list):
        if len(answer) == B or len(answer) == N:
            return answer[i]
        raise ValueError(f"Unexpected answer length: len(answer)={len(answer)} (B={B}, N={N})")
    return answer


_REWARD_BATCH_Q_CACHE = {}

def _reward_batch_cache_key(prompts, completions, answer) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    N = len(completions)
    B = len(prompts)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        q = _extract_user_question_from_prompt(p)
        a = normalize_for_equiv(extract_completion_text(c))
        g = normalize_for_equiv(_gold_for_row(answer, i, B=B, N=N))
        h.update((q or "").encode("utf-8", errors="ignore"))
        h.update(b"\n<|C|>\n")
        h.update((a or "").encode("utf-8", errors="ignore"))
        h.update(b"\n<|G|>\n")
        h.update((g or "").encode("utf-8", errors="ignore"))
        h.update(b"\n<|ROW|>\n")
    return h.digest()


# ============================================================
# 8) calibration reward only
# ============================================================
@torch.no_grad()
def calibration_reward(prompts, completions, answer, **kwargs):
    """
    Combined reward computed in one judge pass per prompt-group.

    For each sample j in a prompt-group:
      1. Correctness part: CORR_W * q_j, where q_j = Judge(C_j vs gold) in {0,1}
      2. Calibration part: CALI_W * r_j, where r_j depends on pairwise consistency
         through BCE(q_j, s_{j,i}) and s_{j,i} = Judge(C_j vs C_i)

    Final reward: CORR_W * q_j + CALI_W * r_j
    """
    eps = 1e-6

    N = len(completions)
    B = len(prompts)
    if N != B:
        raise ValueError(f"[calibration_reward] Expected N == B in prompt-dup GRPO, got N={N}, B={B}")

    groups = _group_by_prompt(prompts)

    comp_texts = [extract_completion_text(c) for c in completions]
    rewards = torch.zeros(N, device=DEVICE, dtype=torch.float32)
    batch_q_labels = [0] * N
    batch_key = _reward_batch_cache_key(prompts, completions, answer)

    for _, idxs in groups.items():
        group = [normalize_for_equiv(comp_texts[i]) for i in idxs]
        gold  = normalize_for_equiv(_gold_for_row(answer, idxs[0], B=B, N=N))
        K = len(group)

        q_str = _extract_user_question_from_prompt(prompts[idxs[0]])

        # ---- q_j: judge(C_j vs gold) ----
        q01, _unp_q = judge_yesno_batched(
            preds=group,
            golds=[gold] * K,
            questions=[q_str] * K,
            max_new_tokens=16,
            batch_size=16,
            max_retries=2,
            use_cache=True,
            shortcut=True,
        )
        q = torch.tensor(q01, dtype=torch.float32, device=DEVICE)  # [K] in {0,1}
        for j, row_idx in enumerate(idxs):
            batch_q_labels[row_idx] = int(q01[j])

        if K == 1:
            ce = torch.zeros(1, device=DEVICE, dtype=torch.float32)
        else:
            # ordered pairs j->i (i!=j)
            a_list, b_list, owner_j = [], [], []
            for j in range(K):
                for i2 in range(K):
                    if i2 == j:
                        continue
                    a_list.append(group[j])
                    b_list.append(group[i2])
                    owner_j.append(j)

            # ---- s_{j,i}: judge(C_j vs C_i) ----
            s01, _unp_s = judge_yesno_batched(
                preds=a_list,
                golds=b_list,
                questions=[q_str] * len(a_list),
                max_new_tokens=16,
                batch_size=16,
                max_retries=2,
                use_cache=True,
                shortcut=True,
            )

            s = torch.tensor(s01, dtype=torch.float32, device=DEVICE).clamp(eps, 1.0 - eps)  # [K*(K-1)]
            owner = torch.tensor(owner_j, dtype=torch.long, device=DEVICE)
            q_rep = q[owner]  # [K*(K-1)]

            # BCE(q, s): q is "label", s is "prob"
            ce_pairs = -(q_rep * torch.log(s) + (1.0 - q_rep) * torch.log(1.0 - s))  # [K*(K-1)]

            ce = torch.zeros(K, device=DEVICE, dtype=torch.float32)
            cnt = torch.zeros(K, device=DEVICE, dtype=torch.float32)
            for idx, j in enumerate(owner_j):
                ce[j] += ce_pairs[idx]
                cnt[j] += 1.0
            ce = ce / cnt.clamp_min(1.0)

        r = (5.0 - ce).clamp(0.0, 5.0)
        for j, row_idx in enumerate(idxs):
            rewards[row_idx] = r[j]

    _REWARD_BATCH_Q_CACHE[batch_key] = batch_q_labels

    # (NEW) apply dynamic calibration weight
    return (rewards * float(CALI_W)).tolist()




@torch.no_grad()
def correctness_reward(prompts, completions, answer, weight: float = 1.0, **kwargs):
    """
    Reward = weight if judge says YES (completion vs gold), else 0.
    Uses the SAME cleaning/canonicalization + cache + retry.
    """
    N = len(completions)
    B = len(prompts)
    if N != B:
        raise ValueError(f"[correctness_reward] Expected N == B in prompt-dup GRPO, got N={N}, B={B}")

    batch_key = _reward_batch_cache_key(prompts, completions, answer)
    cached_labels = _REWARD_BATCH_Q_CACHE.pop(batch_key, None)
    if cached_labels is not None:
        return [float(weight) if int(y) == 1 else 0.0 for y in cached_labels]

    responses = [extract_completion_text(c) for c in completions]
    responses = [normalize_for_equiv(r) for r in responses]

    if isinstance(answer, list):
        golds = [normalize_for_equiv(a) for a in answer]
    else:
        golds = [normalize_for_equiv(answer) for _ in range(N)]

    questions = [_extract_user_question_from_prompt(p) for p in prompts]

    labels, unparsed = judge_yesno_batched(
        preds=responses,
        golds=golds,
        questions=questions,
        max_new_tokens=16,
        batch_size=16,
        max_retries=2,
        use_cache=True,
        shortcut=True,
    )

    if unparsed > 0:
        print(f"[Judge] unparsed={unparsed}/{len(labels)}")

    return [float(weight) if y == 1 else 0.0 for y in labels]


def correctness_reward_fixed(prompts, completions, answer, **kwargs):
    return correctness_reward(prompts, completions, answer, weight=CORR_W, **kwargs)


# ============================================================
# 9) Validation eval helpers
# ============================================================
EVAL_NUM_ROLLOUT = 8
EVAL_MAX_NEW_TOKENS = 256
EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.95
EVAL_BINS = 10

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

def get_semantic_ids_batched_llmjudge(strings_list, question: str, strict_bidirectional: bool = True, max_new_tokens: int = 16, batch_size: int = 16):
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
    same01, _ = judge_yesno_batched(
        preds=a_list,
        golds=b_list,
        questions=q_list,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_retries=2,
        use_cache=True,
        shortcut=True,
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
    ent_max = np.log(n)
    return float(np.clip(ent / (ent_max + 1e-12), 0.0, 1.0))

def compute_semantic_entropy(samples, question: str):
    if not samples:
        return 0.0
    semantic_ids = get_semantic_ids_batched_llmjudge(samples, question=question, strict_bidirectional=True, max_new_tokens=16, batch_size=16)
    return cluster_assignment_entropy(semantic_ids)

def expected_calibration_error(confidences, correctness, n_bins=EVAL_BINS):
    confidences = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
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
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = spearmanr(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)

def safe_kendall(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = kendalltau(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)

def safe_auc(y_true, y_score, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=float)
    if not np.all((y_true == 0) | (y_true == 1)):
        y_true = (y_true >= threshold).astype(int)
    else:
        y_true = y_true.astype(int)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    try:
        return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    except Exception:
        return None

@torch.inference_mode()
def generate_k_and_semantic_confidence_for_eval(model, tokenizer, prompt, lora_request=None, k: int = EVAL_NUM_ROLLOUT, max_new_tokens: int = EVAL_MAX_NEW_TOKENS, temperature: float = EVAL_TEMPERATURE, top_p: float = EVAL_TOP_P, gold_answer: str | None = None, question: str | None = None):
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
        for _ in range(int(k)):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            gen_ids = out[0, in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            samples.append(extract_completion_text(text))

    H = compute_semantic_entropy(samples, question=question or "")
    conf_hat = float(1.0 - H)

    labels_k = None
    if gold_answer is not None:
        gold_clean = normalize_for_equiv(gold_answer)
        samples_clean = [normalize_for_equiv(s) for s in samples]
        q_list = [question or ""] * len(samples_clean)
        labels_k, _ = judge_yesno_batched(
            preds=samples_clean,
            golds=[gold_clean] * len(samples_clean),
            questions=q_list,
            max_new_tokens=16,
            batch_size=16,
            max_retries=2,
            use_cache=True,
            shortcut=True,
        )

    return {"samples": samples, "conf_hat": conf_hat, "labels_k": labels_k}

@torch.inference_mode()
def evaluate_model_on_validation(ds, model, tokenizer, lora_request=None, k: int = EVAL_NUM_ROLLOUT, limit: int | None = None):
    total = 0
    correct_sum = 0.0
    confs, p_hats = [], []

    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        ex = ds[i]
        q = ex["question"]
        gold = ex["answer"]
        prompt = render_generation_prompt(ex["prompt"], tokenizer)

        out = generate_k_and_semantic_confidence_for_eval(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            lora_request=lora_request,
            k=k,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            gold_answer=gold,
            question=q,
        )

        labels_k = out["labels_k"] or [0] * len(out["samples"])
        p_hat = float(sum(labels_k) / max(1, len(labels_k)))
        conf_hat = float(out["conf_hat"])

        total += 1
        correct_sum += p_hat
        confs.append(conf_hat)
        p_hats.append(p_hat)

    acc = correct_sum / max(1, total)
    ece = expected_calibration_error(confs, p_hats, n_bins=EVAL_BINS)
    brier = brier_soft(p_hats, confs)
    spearman = safe_spearman(confs, p_hats)
    kendall = safe_kendall(confs, p_hats)
    auc_conf = safe_auc(p_hats, confs)

    return {
        "n": total,
        "accuracy_mean": acc,
        "ece_soft": ece,
        "brier_soft": brier,
        "spearman": spearman,
        "kendall": kendall,
        "auroc_conf": auc_conf,
    }

# ============================================================
# 9) Trainer configs
# ============================================================
max_prompt_length = 256
training_args = GRPOConfig(
    beta=0.01,
    loss_type="grpo",
    shuffle_dataset=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=1,
    max_steps=-1,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"outputs_{experiment}",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[calibration_reward, correctness_reward_fixed],  # separate calibration + correctness rewards
    args=training_args,
    train_dataset=train_ds,
)

# ============================================================
# 10) Logging + cache callback
# ============================================================
class MetricCSVCallback(TrainerCallback):
    def __init__(
        self,
        prefer_keys=("reward",),
        also_log=("kl", "reward_std", "loss"),
        outdir="outputs",
        csv_name="metrics_log.csv",
        keep_hist=False,
    ):
        self.prefer_keys = list(prefer_keys)
        self.also_log = list(also_log)
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        self.main_key = None
        self.keep_hist = bool(keep_hist)
        self.hist = {} if self.keep_hist else None

        self.csv_path = os.path.join(outdir, csv_name)
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._csv_header = None
        print(f"[MetricCSVCallback] CSV -> {self.csv_path}")

    def _pick_main_key(self, logs):
        for k in self.prefer_keys:
            if k in logs and isinstance(logs[k], numbers.Number):
                return k
        for k, v in logs.items():
            if isinstance(v, numbers.Number):
                return k
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if self.main_key is None:
            self.main_key = self._pick_main_key(logs)
            print(f"[MetricCSVCallback] available keys: {list(logs.keys())}")
            print(f"[MetricCSVCallback] chosen main key: {self.main_key!r}")

        step = int(state.global_step)
        row = {"step": step}

        for k, v in logs.items():
            if isinstance(v, numbers.Number):
                row[k] = float(v)
        for k in self.also_log:
            if k in logs and isinstance(logs[k], numbers.Number):
                row[k] = float(logs[k])

        # (NEW) always write current CALI_W into CSV
        row["cali_w"] = float(CALI_W)

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
            print(f"[MetricCSVCallback] CSV saved: {self.csv_path}")
        except Exception:
            pass

class JudgeCacheCallback(TrainerCallback):
    """
    Prints cache stats periodically; optional clear.
    Only affects speed (judge calls), not reward definition.
    """
    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0:
            return

        if CACHE_PRINT_EVERY and (step % int(CACHE_PRINT_EVERY) == 0):
            hit = CACHE_STATS["hit"]
            miss = CACHE_STATS["miss"]
            tot = hit + miss
            hr = (hit / tot) if tot > 0 else 0.0
            print(
                f"[JudgeCache] step={step} size={len(JUDGE_CACHE)}/{JUDGE_CACHE.max_size} "
                f"hit={hit} miss={miss} hit_rate={hr:.3f} parsed={CACHE_STATS['parsed']} unparsed={CACHE_STATS['unparsed']}"
            )

        if CACHE_CLEAR_EVERY and (step % int(CACHE_CLEAR_EVERY) == 0):
            JUDGE_CACHE.clear_all()
            print(f"[JudgeCache] step={step} cache CLEARED. size={len(JUDGE_CACHE)}")

        if CUDA_EMPTY_CACHE_EVERY and torch.cuda.is_available() and (step % int(CUDA_EMPTY_CACHE_EVERY) == 0):
            torch.cuda.empty_cache()
            print(f"[CUDA] step={step} torch.cuda.empty_cache()")


# (NEW) ramp-up callback to update CALI_W every step
class EpochEvalCallback(TrainerCallback):
    def __init__(self, validation_ds, tokenizer, outdir: str, csv_name: str = "validation_metrics.csv", eval_every_steps: int = 250, eval_limit: int | None = None, split_name: str | None = None):
        self.validation_ds = validation_ds
        self.tokenizer = tokenizer
        self.outdir = outdir
        self.eval_every_steps = max(1, int(eval_every_steps))
        self.eval_limit = eval_limit
        if split_name is not None:
            self.split_name = split_name
        elif validation_ds is train_ds:
            self.split_name = "train"
        elif validation_ds is globals().get("validation_ds"):
            self.split_name = "validation"
        else:
            self.split_name = "unknown"
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, csv_name)
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._csv_header = None
        print(f"[EpochEvalCallback] CSV -> {self.csv_path}")

    def _write_row(self, row):
        if self._csv_writer is None:
            self._csv_header = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header, extrasaction="ignore")
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_f.flush()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step <= 0 or (step % self.eval_every_steps) != 0:
            return

        epoch_value = state.epoch if state.epoch is not None else 0.0
        epoch_idx = int(round(float(epoch_value))) if epoch_value is not None else 0

        eval_model = model or kwargs.get("model")
        if eval_model is None:
            print("[EpochEvalCallback] model is None, skip validation")
            return

        ckpt_dir = os.path.join(self.outdir, f"step_eval_checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        lora_request = None
        try:
            eval_model.save_pretrained(ckpt_dir)
            if hasattr(eval_model, "load_lora"):
                lora_request = eval_model.load_lora(ckpt_dir)
            print(f"[EpochEvalCallback] saved epoch checkpoint -> {ckpt_dir}")
        except Exception as e:
            print(f"[EpochEvalCallback] warning: save/load epoch checkpoint failed: {e}")
            lora_request = None

        t0 = time.time()
        metrics = evaluate_model_on_validation(
            ds=self.validation_ds,
            model=eval_model,
            tokenizer=self.tokenizer,
            lora_request=lora_request,
            k=EVAL_NUM_ROLLOUT,
            limit=self.eval_limit,
        )
        elapsed_sec = time.time() - t0

        row = {
            "epoch": epoch_idx,
            "step": step,
            "split": self.split_name,
            "ckpt_path": ckpt_dir,
            "cali_w": float(CALI_W),
            "n": int(metrics["n"]),
            "accuracy_mean": float(metrics["accuracy_mean"]),
            "ece_soft": float(metrics["ece_soft"]),
            "brier_soft": float(metrics["brier_soft"]),
            "spearman": "" if metrics["spearman"] is None else float(metrics["spearman"]),
            "kendall": "" if metrics["kendall"] is None else float(metrics["kendall"]),
            "auroc_conf": "" if metrics["auroc_conf"] is None else float(metrics["auroc_conf"]),
            "elapsed_sec": float(elapsed_sec),
            "judge_cache_size": int(len(JUDGE_CACHE)),
            "judge_cache_hit": int(CACHE_STATS["hit"]),
            "judge_cache_miss": int(CACHE_STATS["miss"]),
            "judge_parsed": int(CACHE_STATS["parsed"]),
            "judge_unparsed": int(CACHE_STATS["unparsed"]),
        }
        self._write_row(row)
        print(
            f"[EpochEval] epoch={epoch_idx} step={state.global_step} "
            f"acc={metrics['accuracy_mean']:.4f} ece={metrics['ece_soft']:.4f} "
            f"brier={metrics['brier_soft']:.4f} auroc={metrics['auroc_conf'] if metrics['auroc_conf'] is not None else 'NA'} "
            f"elapsed={elapsed_sec/60:.1f} min"
        )

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._csv_f.flush()
            self._csv_f.close()
            print(f"[EpochEvalCallback] CSV saved: {self.csv_path}")
        except Exception:
            pass

class CalibWeightRampCallback(TrainerCallback):
    """
    Update global CALI_W during training (linear ramp).
    Uses state.epoch (float) if available; fallback to step/max_steps.
    """
    def __init__(self, num_epochs: int, print_every_steps: int = 25):
        self.num_epochs = float(num_epochs)
        self.print_every_steps = int(print_every_steps)

    def on_step_end(self, args, state, control, **kwargs):
        global CALI_W

        if state.max_steps and state.max_steps > 0:
            progress = float(state.global_step) / float(state.max_steps)
        else:
            progress = 0.0

        CALI_W = float(cali_weight_schedule(progress))

        if self.print_every_steps and (state.global_step % self.print_every_steps == 0):
            # state.epoch can be None early in some setups
            ep = state.epoch if state.epoch is not None else -1.0
            print(f"[CALI_W] step={state.global_step} epoch={ep:.3f} progress={progress:.3f} CALI_W={CALI_W:.4f}")


trainer.add_callback(MetricCSVCallback(
    prefer_keys=("reward",),
    also_log=("kl", "reward_std", "loss"),
    outdir=f"outputs_{experiment}",
))
trainer.add_callback(JudgeCacheCallback())
trainer.add_callback(CalibWeightRampCallback(
    num_epochs=training_args.num_train_epochs,
    print_every_steps=25,
))
trainer.add_callback(EpochEvalCallback(
    validation_ds=validation_ds,
    tokenizer=tokenizer,
    outdir=f"outputs_{experiment}",
    eval_every_steps=250,
    eval_limit=len(validation_ds),
))

# ============================================================
# 11) Train + save
# ============================================================
trainer.train()

model.save_pretrained(f"grpo_saved_lora_qa_{experiment}")
tokenizer.save_pretrained(f"grpo_saved_lora_qa_{experiment}")
print("[Saved] LoRA + tokenizer")

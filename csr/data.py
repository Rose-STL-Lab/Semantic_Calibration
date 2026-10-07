"""Dataset loading. Every dataset is mapped onto one record schema:
``{"prompt": [system, user], "question": str, "answer": str | list (alternatives), "meta": dict}``.
"""

from __future__ import annotations

from functools import partial

from datasets import load_dataset

from .text import clean_unicode_text, normalize_answer


def _messages(system_prompt: str, question: str, user_template: str | None = None) -> list[dict]:
    user = user_template.format(question=question) if user_template else question
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# per-dataset field mapping
def _hotpot(example, system_prompt, user_template):
    question = example["question"]
    return {
        "prompt": _messages(system_prompt, question, user_template),
        "question": question,
        "answer": example.get("answer", ""),
        "meta": {
            "id": example.get("id", ""),
            "type": example.get("type", ""),
            "level": example.get("level", ""),
        },
    }


def _format_hotpot_context(example) -> str:
    """Distractor paragraphs as "Title: <t>\\n<sentences>" blocks -- byte-identical to the
    original context-grounded runs, so reused RLVR/RD/RLCR checkpoints see their training prompt.
    """
    ctx = example.get("context", {}) or {}
    parts = []
    for title, sentences in zip(ctx.get("title", []) or [], ctx.get("sentences", []) or []):
        body = " ".join(s.strip() for s in (sentences or []) if s)
        if title or body:
            parts.append(f"Title: {title}\n{body}")
    return "\n\n".join(parts).strip()


# RLCR numbers the passages and repeats the strict-output reminder inside the problem.
# Its paragraph *removal* is deliberately NOT reproduced: every method must see the same evidence.
_RLCR_REMINDER = (
    "Your answer will be verified with exact match score. To ensure correct "
    "verification, only provide the answer within the <answer> </answer> tags. "
    "Do not put any sentences or reasoning process within the <answer> </answer> tags."
)


def _format_hotpot_context_rlcr(example) -> str:
    """The passages in RLCR's own layout: "Paragraph <i>" blocks under rules."""
    ctx = example.get("context", {}) or {}
    sections = []
    for idx, sentences in enumerate(ctx.get("sentences", []) or []):
        body = "\n".join(sentences or [])
        sections.append(f"Paragraph {idx} \n\n {body} \n\n")
        sections.append("--------------------------------\n\n")
    return "".join(sections)


def _hotpot_context(example, system_prompt, user_template, method: str = "CSR"):
    """HotpotQA with the passages supplied, laid out in each method's own phrasing (as trained)."""
    question = example["question"]
    record = _hotpot(example, system_prompt, user_template)

    if method == "RLCR":
        problem = (f"Question: {question} \n\n"
                   f"{_RLCR_REMINDER} \n\n"
                   f"Supporting Information: {_format_hotpot_context_rlcr(example)} \n\n")
        record["prompt"] = _messages(system_prompt, problem, user_template)
        return record

    context = _format_hotpot_context(example)
    if not context:
        return record
    if method == "RD":
        # RD's own context phrasing (`_normalize_squad_dataset_map_function`).
        record["prompt"] = _messages(
            system_prompt, f"Context: {context}, Question: {question}", user_template)
    else:
        record["prompt"] = _messages(
            system_prompt, f"Context:\n{context}\n\nQuestion: {question}")
    return record


def _trivia(example, system_prompt, user_template):
    question = example["question"]
    answer = example.get("answer", {}) or {}
    aliases = list(answer.get("normalized_aliases", []))
    aliases.extend(normalize_answer(a) for a in answer.get("human_answers", []))
    aliases = [a for a in aliases if a]
    return {
        "prompt": _messages(system_prompt, question, user_template),
        "question": question,
        "answer": aliases,
        "meta": {
            "question_id": example.get("question_id", ""),
            "question_source": example.get("question_source", ""),
        },
    }


def _msmarco(example, system_prompt, user_template):
    question = clean_unicode_text(example["query"])
    well_formed = example.get("wellFormedAnswers") or []
    answers = example.get("answers") or []
    raw_answers = well_formed if len(well_formed) > 0 else answers
    gold = [clean_unicode_text(a) for a in raw_answers if clean_unicode_text(a)]
    return {
        "prompt": _messages(system_prompt, question, user_template),
        "question": question,
        "answer": gold,
        "meta": {
            "query_id": example.get("query_id"),
            "query_type": example.get("query_type", ""),
        },
    }


def _nq_open(example, system_prompt, user_template):
    question = clean_unicode_text(example["question"])
    raw_answer = example["answer"]
    if isinstance(raw_answer, list):
        gold = [clean_unicode_text(a) for a in raw_answer]
    else:
        gold = clean_unicode_text(raw_answer)
    return {
        "prompt": _messages(system_prompt, question, user_template),
        "question": question,
        "answer": gold,
        "meta": {"metadata": example.get("metadata", "")},
    }


BUILDERS = {
    "hotpot": _hotpot,
    "hotpot_context": _hotpot_context,
    "trivia": _trivia,
    "msmarco": _msmarco,
    "nq_open": _nq_open,
}

# MSMARCO has rows with no usable gold answer; they cannot be scored.
REQUIRES_NONEMPTY_GOLD = {"msmarco"}


def _resolve_split(raw, requested: str) -> str:
    if requested in raw:
        return requested
    for fallback in ("validation", "dev", "test", "train"):
        if fallback in raw:
            return fallback
    return next(iter(raw.keys()))


def load_split(cfg, split: str = "eval", system_prompt: str | None = None,
               size: int | None = None):
    """Load one split of the configured dataset in the unified schema.

    ``split`` is "train" or "eval"; sizes default to the config's
    ``train_size`` / ``eval_size``.
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name not in BUILDERS:
        raise ValueError(f"no field mapping for dataset {name!r}")

    method_cfg = cfg.get("method", {})
    system_prompt = system_prompt if system_prompt is not None \
        else method_cfg.get("system_prompt", "")
    user_template = method_cfg.get("user_template")

    # dataset may override the method's system prompt; RD/RLCR keep their own format instructions
    override = (ds_cfg.get("system_prompt") or {})
    if isinstance(override, str):
        override = {"default": override}
    method_name = str(method_cfg.get("name", ""))
    if method_name in override:
        system_prompt = override[method_name]
    elif "default" in override and method_name not in ("RD", "RLCR"):
        system_prompt = override["default"]

    hf = ds_cfg["hf"]
    args = [hf["path"]] + ([hf["config"]] if hf.get("config") else [])
    raw = load_dataset(*args)

    requested = ds_cfg["train_split"] if split == "train" else ds_cfg["eval_split"]
    split_name = _resolve_split(raw, requested)
    if split_name != requested:
        print(f"[data] split {requested!r} unavailable, using {split_name!r}")

    builder = BUILDERS[name]
    if name == "hotpot_context":
        builder = partial(_hotpot_context, method=method_name)
    dataset = raw[split_name].map(
        lambda ex: builder(ex, system_prompt, user_template),
        remove_columns=raw[split_name].column_names,
    )

    if name in REQUIRES_NONEMPTY_GOLD:
        dataset = dataset.filter(lambda ex: isinstance(ex["answer"], list) and len(ex["answer"]) > 0)

    dataset = dataset.shuffle(seed=int(cfg.get("seed", 42)))

    if size is None:
        size = ds_cfg["train_size"] if split == "train" else ds_cfg["eval_size"]
    if size:
        dataset = dataset.select(range(0, min(int(size), len(dataset))))

    print(f"[data] {name}/{split_name}: {len(dataset)} examples")
    return dataset


def correctness_prompt_kind(cfg) -> str:
    """Judge prompt for correctness: ``judge`` reuses the equivalence prompt, ``gold`` picks the
    single- or multi-gold correctness prompt by gold count.
    """
    return cfg["dataset"].get("correctness_prompt", "judge")

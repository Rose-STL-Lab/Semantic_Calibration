# Dataset loading. Every dataset is mapped onto one record schema:
# {"prompt": [system, user], "question": str,
#  "answer": str | list (alternatives), "meta": dict}.

from __future__ import annotations

import json
from functools import partial

from datasets import load_dataset

from .parsing import clean_unicode_text, normalize_answer


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
    # "Title: <t>\n<sentences>" blocks, the layout the context checkpoints train on
    ctx = example.get("context", {}) or {}
    parts = []
    for title, sentences in zip(ctx.get("title", []) or [], ctx.get("sentences", []) or []):
        body = " ".join(s.strip() for s in (sentences or []) if s)
        if title or body:
            parts.append(f"Title: {title}\n{body}")
    return "\n\n".join(parts).strip()


# RLCR numbers the passages and repeats the strict-output reminder inside the problem.
# Its paragraph *removal* is deliberately not reproduced: same evidence for every method.
_RLCR_REMINDER = (
    "Your answer will be verified with exact match score. To ensure correct "
    "verification, only provide the answer within the <answer> </answer> tags. "
    "Do not put any sentences or reasoning process within the <answer> </answer> tags."
)


def _rlcr_passages(bodies) -> str:
    # RLCR's passage layout: numbered "Paragraph <i>" blocks, no titles, separator rules
    sections = []
    for idx, body in enumerate(bodies):
        sections.append(f"Paragraph {idx} \n\n {body} \n\n")
        sections.append("--------------------------------\n\n")
    return "".join(sections)


def _rlcr_problem(system_prompt, user_template, bodies, question):
    # RLCR trains on "Question / reminder / Supporting Information"; any context
    # dataset must use this layout or the checkpoint sees an unseen prompt format
    problem = (f"Question: {question} \n\n"
               f"{_RLCR_REMINDER} \n\n"
               f"Supporting Information: {_rlcr_passages(bodies)} \n\n")
    return _messages(system_prompt, problem, user_template)


def _format_hotpot_context_rlcr(example) -> str:
    ctx = example.get("context", {}) or {}
    return _rlcr_passages("\n".join(s or [])
                          for s in (ctx.get("sentences", []) or []))


def _hotpot_context(example, system_prompt, user_template, method: str = "CSR"):
    # passages supplied, laid out in each method's own phrasing (as trained)
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
        # RD's own context phrasing (_normalize_squad_dataset_map_function)
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


def _simpleqa(example, system_prompt, user_template):
    question = example["problem"]
    meta = example.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta.replace("'", '"'))
        except json.JSONDecodeError:
            meta = {}
    return {
        "prompt": _messages(system_prompt, question, user_template),
        "question": question,
        "answer": str(example.get("answer", "")),
        "meta": {
            "topic": meta.get("topic", ""),
            "answer_type": meta.get("answer_type", ""),
        },
    }


def _context_prompt(system_prompt, user_template, context: str, question: str):
    # the layout hotpot_context trains on, so an OOD prompt differs only in content
    if not context:
        return _messages(system_prompt, question, user_template)
    return _messages(system_prompt, f"Context:\n{context}\n\nQuestion: {question}")


def _musique(example, system_prompt, user_template, method: str = "CSR"):
    question = example["question"]
    blocks, bodies = [], []
    for para in example.get("paragraphs") or []:
        title = (para.get("title") or "").strip()
        body = (para.get("paragraph_text") or "").strip()
        if title or body:
            blocks.append(f"Title: {title}\n{body}")
            bodies.append(body)
    aliases = [a for a in (example.get("answer_aliases") or []) if str(a).strip()]
    gold = [str(example.get("answer", ""))] + [str(a) for a in aliases]
    prompt = (_rlcr_problem(system_prompt, user_template, bodies, question)
              if method == "RLCR" else
              _context_prompt(system_prompt, user_template, "\n\n".join(blocks), question))
    return {
        "prompt": prompt,
        "question": question,
        "answer": [g for g in gold if g.strip()],
        "meta": {
            "id": example.get("id", ""),
            # always True in the answerable-only release used here
            "answerable": example.get("answerable"),
            "hops": str(example.get("id", "")).split("hop")[0],
        },
    }


def _msmarco_context(example, system_prompt, user_template, method: str = "CSR"):
    # the same queries and golds as _msmarco, with the passages MSMARCO ships
    record = _msmarco(example, system_prompt, user_template)
    passages = example.get("passages") or {}
    texts = passages.get("passage_text") or []
    blocks, bodies = [], []
    for i, text in enumerate(texts):
        body = clean_unicode_text(text).strip()
        if body:
            blocks.append(f"Title: Passage {i + 1}\n{body}")
            bodies.append(body)
    record["prompt"] = (
        _rlcr_problem(system_prompt, user_template, bodies, record["question"])
        if method == "RLCR" else
        _context_prompt(system_prompt, user_template, "\n\n".join(blocks), record["question"]))
    record["meta"]["num_passages"] = len(blocks)
    return record


def _twowiki(example, system_prompt, user_template, method: str = "CSR"):
    # HotpotQA's layout with `content` in place of `sentences`
    question = example["question"]
    ctx = example.get("context", {}) or {}
    blocks, bodies = [], []
    for title, sentences in zip(ctx.get("title", []) or [], ctx.get("content", []) or []):
        body = " ".join(str(x).strip() for x in (sentences or []) if x)
        if title or body:
            blocks.append(f"Title: {title}\n{body}")
            bodies.append(body)
    prompt = (_rlcr_problem(system_prompt, user_template, bodies, question)
              if method == "RLCR" else
              _context_prompt(system_prompt, user_template, "\n\n".join(blocks), question))
    return {
        "prompt": prompt,
        "question": question,
        "answer": str(example.get("answer", "")),
        "meta": {
            "id": example.get("_id", ""),
            # compositional | bridge_comparison | comparison | inference
            "type": example.get("type", ""),
        },
    }


BUILDERS = {
    "hotpot": _hotpot,
    "hotpot_context": _hotpot_context,
    "trivia": _trivia,
    "msmarco": _msmarco,
    "nq_open": _nq_open,
    "simpleqa": _simpleqa,
    "musique": _musique,
    "msmarco_context": _msmarco_context,
    "2wiki": _twowiki,
    "2wiki_comp": _twowiki,
    "2wiki_inf": _twowiki,
}

# builders whose prompt depends on the method (RD/RLCR keep their own layouts)
METHOD_AWARE_BUILDERS = {"hotpot_context", "musique", "msmarco_context",
                         "2wiki", "2wiki_comp", "2wiki_inf"}

# MSMARCO has rows with no usable gold; unscorable
REQUIRES_NONEMPTY_GOLD = {"msmarco", "msmarco_context"}


def _resolve_split(raw, requested: str) -> str:
    if requested in raw:
        return requested
    for fallback in ("validation", "dev", "test", "train"):
        if fallback in raw:
            return fallback
    return next(iter(raw.keys()))


def load_split(cfg, split: str = "eval", size: int | None = None):
    # split is "train" or "eval"; sizes default to train_size / eval_size
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name not in BUILDERS:
        raise ValueError(f"no field mapping for dataset {name!r}")

    method_cfg = cfg.get("method", {})
    system_prompt = method_cfg.get("system_prompt", "")
    user_template = method_cfg.get("user_template")

    # dataset may override the method's system prompt; RD/RLCR keep their own format
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
    # RLCR gets its own layout on every context dataset; RD only on hotpot_context
    if name in METHOD_AWARE_BUILDERS:
        builder = partial(builder, method=method_name)
    dataset = raw[split_name].map(
        lambda ex: builder(ex, system_prompt, user_template),
        remove_columns=raw[split_name].column_names,
    )

    if name in REQUIRES_NONEMPTY_GOLD:
        dataset = dataset.filter(lambda ex: isinstance(ex["answer"], list) and len(ex["answer"]) > 0)

    # comparison-type 2Wiki questions are answerable without the passages; keep_types drops them
    keep = ds_cfg.get("keep_types")
    if keep:
        keep = set(keep)
        before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["meta"].get("type") in keep)
        print(f"[data] {name}: kept {len(dataset)}/{before} rows of types {sorted(keep)}")

    dataset = dataset.shuffle(seed=int(cfg.get("seed", 42)))

    # single-split datasets: eval takes the head, train the tail, so they stay disjoint
    if ds_cfg.get("single_split"):
        head = int(ds_cfg["eval_size"])
        if split == "train":
            dataset = dataset.select(range(min(head, len(dataset)), len(dataset)))
        else:
            dataset = dataset.select(range(0, min(head, len(dataset))))

    if size is None:
        size = ds_cfg["train_size"] if split == "train" else ds_cfg["eval_size"]
    if size:
        dataset = dataset.select(range(0, min(int(size), len(dataset))))

    print(f"[data] {name}/{split_name}: {len(dataset)} examples")
    return dataset


def correctness_prompt_kind(cfg) -> str:
    # judge: reuse the equivalence prompt. gold: single/multi correctness by gold count
    return cfg["dataset"].get("correctness_prompt", "judge")

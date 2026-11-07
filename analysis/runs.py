# Run directories read by the appendix scripts; README Section 5 writes these names.

from __future__ import annotations

import json

MODELS = ["llama8b", "qwen7b", "mistral7b"]

# (directory name, display label)
DATASETS = [
    ("hotpot", "HotpotQA"),
    ("trivia", "TriviaQA"),
    ("msmarco", "MSMARCO"),
    ("nq_open", "NQ-Open"),
]

MODEL_LABELS = {"llama8b": "Llama", "qwen7b": "Qwen", "mistral7b": "Mistral"}

# K=16 where confidence comes from the rollout spread; greedy (k1) for the two
# that verbalize one.
RUN_TEMPLATES = {
    "Base": "Base_{model}_{dataset}_base_nano_k16",
    "RLVR": "RLVR_{model}_{dataset}_step2500_nano_k16",
    "RD": "RD_{model}_{dataset}_b8_step1250_nano_k1",
    "RLCR": "RLCR_{model}_{dataset}_step2500_nano_k1",
    "CSR": "CSR_{model}_{dataset}_z10_step2500_nano_k16",
}

# confidence from semantic dispersion, not read off a verbalized scalar
SEMANTIC_METHODS = {"Base", "RLVR", "CSR"}

# the sampling methods, whose runs store K rollouts per question; the post-hoc
# estimators and aggregation rules iterate this, not RUN_TEMPLATES
SEMANTIC_TEMPLATES = {m: t for m, t in RUN_TEMPLATES.items() if m in SEMANTIC_METHODS}


def is_final(path) -> bool:
    # results.json written by a finished evaluation
    try:
        return json.loads(path.read_text()).get("status") == "final"
    except (OSError, ValueError):
        return False


def missing_runs(results_dir, templates: dict, model: str) -> list[str]:
    # run names among templates x DATASETS without a finished results.json
    names = [t.format(model=model, dataset=ds) for t in templates.values() for ds, _ in DATASETS]
    return [n for n in names if not is_final(results_dir / n / "results.json")]

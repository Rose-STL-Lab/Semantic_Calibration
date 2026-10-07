"""Run directories for the appendix scripts: <METHOD>_<backbone>_<dataset>[_<arm>]_<step>_<judge>_k<K>."""

from __future__ import annotations

MODELS = ["llama8b", "qwen7b", "mistral7b"]

# (directory name, label used in the paper's tables)
DATASETS = [
    ("hotpot", "HotpotQA"),
    ("trivia", "TriviaQA"),
    ("msmarco", "MSMARCO"),
    ("nq_open", "NQ-Open"),
]

MODEL_LABELS = {"llama8b": "Llama", "qwen7b": "Qwen", "mistral7b": "Mistral"}

# K=16 for the methods whose confidence comes from the rollout spread, greedy
# decoding (k1) for the two that verbalize a confidence.
RUN_TEMPLATES = {
    "Base": "Base_{model}_{dataset}_base_oss_k16",
    "RLVR": "RLVR_{model}_{dataset}_step2500_oss_k16",
    "RD": "RD_{model}_{dataset}_step2500_oss_k1",
    "RLCR": "RLCR_{model}_{dataset}_step2500_oss_k1",
    "CSR": "CSR_{model}_{dataset}_z10_step2500_oss_k16",
}

# Methods whose confidence is derived from semantic dispersion rather than read
# off a verbalized scalar.
SEMANTIC_METHODS = {"Base", "RLVR", "CSR"}

DATASET_NAMES = [name for name, _ in DATASETS]


def run_name(method: str, model: str, dataset: str) -> str:
    """The directory under ``results/`` holding one method/model/dataset run."""
    return RUN_TEMPLATES[method].format(model=model, dataset=dataset)

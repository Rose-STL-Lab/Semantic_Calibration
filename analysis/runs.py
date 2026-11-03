# Run directories for the appendix scripts:
# <METHOD>_<backbone>_<dataset>[_<arm>]_<step>_<judge>_k<K>.

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

# K=16 where confidence comes from the rollout spread; greedy (k1) for the two
# that verbalize one.
RUN_TEMPLATES = {
    "Base": "Base_{model}_{dataset}_base_oss_k16",
    "RLVR": "RLVR_{model}_{dataset}_step2500_oss_k16",
    "RD": "RD_{model}_{dataset}_b8_step1250_oss_k1",
    "RLCR": "RLCR_{model}_{dataset}_step2500_oss_k1",
    "CSR": "CSR_{model}_{dataset}_z10_step2500_oss_k16",
}

# confidence from semantic dispersion, not read off a verbalized scalar
SEMANTIC_METHODS = {"Base", "RLVR", "CSR"}

# SemDens and the SIMBA rules need per-rollout clusters, which only the sampling
# methods record -- they iterate this, not RUN_TEMPLATES.
SEMANTIC_TEMPLATES = {m: t for m, t in RUN_TEMPLATES.items() if m in SEMANTIC_METHODS}



def run_name(method: str, model: str, dataset: str) -> str:
    return RUN_TEMPLATES[method].format(model=model, dataset=dataset)

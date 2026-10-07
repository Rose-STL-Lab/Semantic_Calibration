"""Table 1 recomputed with majority-vote accuracy, next to the published one.

Confidence is untouched (still exp(-SE) over the K rollouts); only the label
changes, so ECE and AUROC are recomputed against it.

    python analysis/majority_vote.py            # markdown
    python analysis/majority_vote.py --latex    # Table 1 body, both blocks
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.runs import (  # noqa: E402
    DATASETS, MODELS, MODEL_LABELS, RUN_TEMPLATES, SEMANTIC_METHODS, run_name,
)

from csr.metrics import summarize                      # noqa: E402

RESULTS = ROOT / "results"
ECE_BINS = 10

# Semantic methods: K rollouts per question, so the label can be redefined.
SEMANTIC = {
    "Base": "Base_{m}_{d}_base_oss_k16",
    "RLVR": "RLVR_{m}_{d}_step2500_oss_k16",
    "CSR": "CSR_{m}_{d}_z10_step2500_oss_k16",
}
# Verbalized methods: one answer, one confidence -- nothing to re-aggregate.
VERBALIZED = {
    "RD": "RD_{m}_{d}_step2500_oss_k1",
    "RLCR": "RLCR_{m}_{d}_step2500_oss_k1",
}
ORDER = ["Base", "RLVR", "RD", "RLCR", "CSR"]


def load(exp: str) -> dict:
    return json.loads((RESULTS / exp / "results.json").read_text())


def token_cost(records: list[dict]) -> float:
    """Prompt counted once plus every completion, averaged over questions."""
    total = 0.0
    for record in records:
        rollouts = record.get("rollouts")
        if rollouts is None:                     # verbalized: a single answer
            total += record["prompt_tokens"] + record["completion_tokens"]
        else:
            total += record["prompt_tokens"] + sum(r["completion_tokens"] for r in rollouts)
    return total / len(records)


def majority_label(record: dict) -> float:
    """1 when at least half the rollouts in the largest semantic cluster are correct."""
    counts = collections.Counter(r["semantic_id"] for r in record["rollouts"])
    top = counts.most_common(1)[0][0]
    correct = [r["correct"] for r in record["rollouts"] if r["semantic_id"] == top]
    return 1.0 if sum(correct) / len(correct) >= 0.5 else 0.0


def scores(exp: str, vote: bool) -> dict:
    payload = load(exp)
    records = payload["records"]
    confs = [r["confidence"] for r in records]
    if vote and records and records[0].get("rollouts") is not None:
        labels = [majority_label(r) for r in records]
    else:
        labels = [float(r["accuracy"]) for r in records]
    out = summarize(confs, labels, ECE_BINS)
    out["tok"] = token_cost(records)
    return out


def table(model: str, vote: bool) -> dict[str, dict]:
    """{method: {dataset: metrics, 'avg': metrics}} for one model block."""
    block = {}
    for method in ORDER:
        pattern = SEMANTIC.get(method) or VERBALIZED[method]
        per_ds = {}
        for ds in DATASETS:
            exp = pattern.format(model=model, dataset=ds)
            if not (RESULTS / exp / "results.json").exists():
                per_ds[ds] = None
                continue
            per_ds[ds] = scores(exp, vote)
        done = [v for v in per_ds.values() if v]
        per_ds["avg"] = ({k: sum(v[k] for v in done) / len(done)
                          for k in ("accuracy", "ece", "auroc", "tok")}
                         if len(done) == len(DATASETS) else None)
        block[method] = per_ds
    return block


def fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def print_markdown() -> None:
    for model in MODELS:
        now = table(model, vote=False)
        vote = table(model, vote=True)
        print(f"\n### {model}  (per cell: current -> majority vote)")
        header = "| Method | " + " | ".join(f"{d} acc / ECE" for d in DATASETS) + " | Avg acc / ECE / AUROC |"
        print(header)
        print("|" + "---|" * (len(DATASETS) + 2))
        for method in ORDER:
            cells = []
            for ds in DATASETS:
                a, b = now[method][ds], vote[method][ds]
                if a is None:
                    cells.append("--")
                elif method in VERBALIZED:
                    cells.append(f"{fmt(a['accuracy'])} / {fmt(a['ece'])}")
                else:
                    cells.append(f"{fmt(a['accuracy'])}→{fmt(b['accuracy'])} / "
                                 f"{fmt(a['ece'])}→{fmt(b['ece'])}")
            a, b = now[method]["avg"], vote[method]["avg"]
            if a is None:
                cells.append("--")
            elif method in VERBALIZED:
                cells.append(f"{fmt(a['accuracy'])} / {fmt(a['ece'])} / {fmt(a['auroc'])}")
            else:
                cells.append(f"{fmt(a['accuracy'])}→{fmt(b['accuracy'])} / "
                             f"{fmt(a['ece'])}→{fmt(b['ece'])} / "
                             f"{fmt(a['auroc'])}→{fmt(b['auroc'])}")
            print(f"| {method} | " + " | ".join(cells) + " |")


def print_latex() -> None:
    """The 15 numeric columns of Table 1, majority-vote version, per model."""
    for model in MODELS:
        vote = table(model, vote=True)
        print(f"% ---- {model} (majority vote) ----")
        for method in ORDER:
            cells = []
            for ds in DATASETS:
                m = vote[method][ds]
                cells += ["--"] * 3 if m is None else [fmt(m["accuracy"]), fmt(m["ece"]), fmt(m["auroc"])]
            avg = vote[method]["avg"]
            cells += ["--"] * 3 if avg is None else [f"{avg['tok']:.2f}", fmt(avg["ece"]), fmt(avg["auroc"])]
            print(f"& {method:5} & " + " & ".join(cells) + "\\\\")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latex", action="store_true", help="print LaTeX rows instead of markdown")
    args = parser.parse_args()
    print_latex() if args.latex else print_markdown()


if __name__ == "__main__":
    main()

"""Table 6: two post-hoc estimators (SemDens, bayes-beta) recomputed from the stored runs.

    python analysis/estimators.py                 # markdown, K=16
    python analysis/estimators.py --k 8           # first 8 rollouts
    python analysis/estimators.py --latex         # Table 6 body
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.runs import DATASETS, MODELS, SEMANTIC_TEMPLATES  # noqa: E402

from csr.metrics import expected_calibration_error      # noqa: E402

RESULTS = ROOT / "results"
EPS = 1e-3            # Jaccard puts mass on exactly 0 and 1; Beta cannot
N_FOLDS = 5
ECE_BINS = 10

_TOKEN = re.compile(r"[^a-z0-9]+")


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.split(str(text).lower()) if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fit_beta(x: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Beta fit, floored so a degenerate fold stays finite."""
    x = np.clip(np.asarray(x, float), EPS, 1 - EPS)
    mean, var = float(x.mean()), max(float(x.var()), 1e-6)
    conc = max(mean * (1 - mean) / var - 1.0, 1e-3)
    return max(mean * conc, 1e-2), max((1 - mean) * conc, 1e-2)


def load_run(exp: str, k: int) -> list[dict]:
    """Records of one finished run, truncated to the first k rollouts."""
    payload = json.loads((RESULTS / exp / "results.json").read_text())
    if payload.get("status") != "final":
        raise ValueError(f"{exp}: status={payload.get('status')!r}, not final")
    out = []
    for record in payload["records"]:
        rollouts = record["rollouts"][:k]
        out.append({
            "idx": str(record["idx"]),
            "answers": [r["extracted_answer"] for r in rollouts],
            "correct": [int(r["correct"]) for r in rollouts],
            "cluster": [int(r["semantic_id"]) for r in rollouts],
        })
    return out


def semantic_density(records: list[dict]) -> np.ndarray:
    """Semantic density per question, references weighted by exp(mean token log-prob).

    Deviates from Qiu et al.: the kernel is the 0/1 same-semantic-cluster indicator,
    not DeBERTa-MNLI 1-(p_contra + 0.5 p_neutral), and the references are the K
    temperature rollouts, not diverse beam search (Appendix D.1).  With no
    log-probability file the weights go uniform -- the unweighted variant, whereas
    the published table is the weighted one.
    """
    out = []
    for rec in records:
        ids = rec["cluster"]
        answers = rec["answers"]
        lp = rec.get("logprob")
        k = len(ids)

        seen, ref = set(), []
        for i in range(k):                      # deduplicate by exact string
            if answers[i] not in seen:
                seen.add(answers[i])
                ref.append(i)

        w = {i: (float(np.exp(lp[i])) if lp is not None else 1.0) for i in ref}
        total = sum(w.values())
        if total <= 0:
            out.append(1.0)
            continue
        # Each rollout in turn is the scored response; the reference set includes it.
        out.append(float(np.mean([
            sum(w[i] for i in ref if ids[i] == ids[t]) / total for t in range(k)
        ])))
    return np.asarray(out)


def bayes_beta(records: list[dict]) -> np.ndarray:
    """Out-of-fold SIMBA-UQ posterior, averaged over each question's rollouts."""
    sims, labels, groups = [], [], []
    for qi, rec in enumerate(records):
        tsets = [tokens(a) for a in rec["answers"]]
        k = len(tsets)
        for i in range(k):
            row = [jaccard(tsets[i], tsets[j]) for j in range(k) if j != i]
            sims.append(np.clip(row, EPS, 1 - EPS))
            labels.append(rec["correct"][i])
            groups.append(qi)
    sims = np.asarray(sims, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)

    conf = np.zeros(len(labels), dtype=float)
    if len(set(labels.tolist())) < 2:
        # A dataset where every rollout is correct (or none) carries no signal.
        conf[:] = float(labels.mean())
    else:
        for train, test in GroupKFold(n_splits=N_FOLDS).split(sims, labels, groups):
            p0 = float(labels[train].mean())
            a1, b1 = fit_beta(sims[train][labels[train] == 1].ravel())
            a0, b0 = fit_beta(sims[train][labels[train] == 0].ravel())
            ll1 = beta_dist.logpdf(sims[test], a1, b1).sum(axis=1) + np.log(max(p0, 1e-6))
            ll0 = beta_dist.logpdf(sims[test], a0, b0).sum(axis=1) + np.log(max(1 - p0, 1e-6))
            hi = np.maximum(ll1, ll0)          # log-sum-exp, the posteriors underflow otherwise
            conf[test] = np.exp(ll1 - hi) / (np.exp(ll1 - hi) + np.exp(ll0 - hi))

    per_question = np.zeros(len(records), dtype=float)
    counts = np.zeros(len(records), dtype=float)
    np.add.at(per_question, groups, conf)
    np.add.at(counts, groups, 1.0)
    return per_question / np.maximum(counts, 1.0)


LOGPROBS_NAME = "logprobs_hf.json"   # the transformers scorer; see aggregation


def attach_logprobs(exp: str, records: list[dict], k: int) -> bool:
    """Add each rollout's mean token log-probability, which SemDens weights by."""
    path = RESULTS / exp / LOGPROBS_NAME
    if not path.exists():
        print(f"[warn] {exp}: no {LOGPROBS_NAME}, SemDens falls back to uniform weights",
              file=sys.stderr)
        return False
    stored = json.loads(path.read_text())
    for rec in records:
        rec["logprob"] = [float(x) for x in stored[rec["idx"]][:k]]
    return True


ESTIMATORS = {"SemDens": semantic_density, "bayes-beta": bayes_beta}


def score(exp: str, k: int) -> dict[str, tuple[float, float]]:
    """{estimator: (ECE, AUROC)} for one run."""
    records = load_run(exp, k)
    attach_logprobs(exp, records, k)
    acc = np.asarray([np.mean(r["correct"]) for r in records], dtype=float)
    labels = (acc >= 0.5).astype(int)
    out = {}
    for name, fn in ESTIMATORS.items():
        conf = fn(records)
        auroc = float(roc_auc_score(labels, conf)) if 0 < labels.mean() < 1 else float("nan")
        out[name] = (expected_calibration_error(conf, acc, ECE_BINS), auroc)
    return out


def collect(k: int) -> dict:
    """{(estimator, model, method): {dataset: (ece, auroc), 'avg': ...}}"""
    table = {}
    for model in MODELS:
        for method, pattern in SEMANTIC_TEMPLATES.items():
            per_ds = {}
            for ds, _ in DATASETS:
                exp = pattern.format(model=model, dataset=ds)
                if not (RESULTS / exp / "results.json").exists():
                    print(f"[skip] missing run: {exp}")
                    continue
                per_ds[ds] = score(exp, k)
                print(f"[done] {exp:44} " + "  ".join(
                    f"{n}: ECE {v[0]:.4f} AUROC {v[1]:.4f}" for n, v in per_ds[ds].items()), flush=True)
            for name in ESTIMATORS:
                cells = {ds: per_ds[ds][name] for ds in per_ds}
                if len(cells) == len(DATASETS):
                    cells["avg"] = (float(np.mean([v[0] for v in cells.values()])),
                                    float(np.mean([v[1] for v in cells.values()])))
                table[(name, model, method)] = cells
    return table


def print_markdown(table: dict, k: int) -> None:
    for name in ESTIMATORS:
        print(f"\n### {name}  (K={k}, ECE / AUROC per cell)")
        print("| Model | Method | " + " | ".join(t for _, t in DATASETS) + " | Avg |")
        print("|" + "---|" * (len(DATASETS) + 3))
        for model in MODELS:
            for method in SEMANTIC_TEMPLATES:
                cells = table.get((name, model, method), {})
                row = []
                for ds, _ in DATASETS + [("avg", "Avg")]:
                    v = cells.get(ds)
                    row.append("--" if v is None else f"{v[0]:.4f} / {v[1]:.4f}")
                print(f"| {model} | {method} | " + " | ".join(row) + " |")


def print_latex(table: dict, k: int) -> None:
    print(f"% Table 6 body, K={k}, gpt-oss-20b judge")
    for name in ESTIMATORS:
        print(f"% ---- {name} ----")
        for model in MODELS:
            for method in SEMANTIC_TEMPLATES:
                cells = table.get((name, model, method), {})
                row = []
                for ds, _ in DATASETS + [("avg", "Avg")]:
                    v = cells.get(ds)
                    row.append("-- / --" if v is None else f"{v[0]:.4f} / {v[1]:.4f}")
                print(f"& {method:5} & " + " & ".join(row) + "\\\\   % " + model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=16, help="rollouts per question (16 = the full run)")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    table = collect(args.k)
    print_latex(table, args.k) if args.latex else print_markdown(table, args.k)


if __name__ == "__main__":
    main()

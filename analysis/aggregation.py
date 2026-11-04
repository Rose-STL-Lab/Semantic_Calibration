"""Two SIMBA-UQ aggregation rules (arithmetic, clf-pairs+gen) over the stored rollouts.

    python analysis/aggregation.py [--k 16] [--latex]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.runs import DATASETS, SEMANTIC_TEMPLATES  # noqa: E402

from csr.metrics import expected_calibration_error          # noqa: E402
from analysis.estimators import jaccard, tokens, load_run  # noqa: E402

RESULTS = ROOT / "results"
MODEL = "qwen7b"                # default; override with --model (llama8b, mistral7b)
SEED = 0
N_FOLDS = 5
ECE_BINS = 10


def sim_matrix(answers: list[str]) -> np.ndarray:
    tsets = [tokens(a) for a in answers]
    k = len(tsets)
    m = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            m[i, j] = m[j, i] = jaccard(tsets[i], tsets[j])
    return m


def arithmetic(records: list[dict]) -> np.ndarray:
    # group mean pairwise similarity, one per question
    out = []
    for rec in records:
        m = sim_matrix(rec["answers"])
        k = len(m)
        per_rollout = m.sum(axis=1) / max(k - 1, 1)     # diagonal is zero
        out.append(float(per_rollout.mean()))
    return np.asarray(out)


# departs from SIMBA-UQ: sorted sims, min_samples_leaf=5 forest, question-grouped
# 5-fold, not depth 4 + one random 50/50 split
def clf_pairs_gen(records: list[dict]) -> np.ndarray:
    # out-of-fold random-forest posterior, averaged per question
    feats, labels, groups = [], [], []
    for qi, rec in enumerate(records):
        m = sim_matrix(rec["answers"])
        k = len(m)
        lp = rec.get("logprob")
        for i in range(k):
            # sorted: rollouts are exchangeable, index order is noise
            sims = sorted(float(m[i, j]) for j in range(k) if j != i)
            gen = [float(np.exp(lp[i]))] if lp is not None else []
            feats.append(sims + gen)
            labels.append(rec["correct"][i])
            groups.append(qi)
    X = np.asarray(feats, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)

    conf = np.zeros(len(y), dtype=float)
    if len(set(y.tolist())) < 2:
        conf[:] = float(y.mean())
    else:
        for train, test in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
            rf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                        n_jobs=8, min_samples_leaf=5)
            rf.fit(X[train], y[train])
            proba = rf.predict_proba(X[test])
            conf[test] = proba[:, list(rf.classes_).index(1)] if 1 in rf.classes_ else 0.0

    per_q = np.zeros(len(records)); counts = np.zeros(len(records))
    np.add.at(per_q, groups, conf)
    np.add.at(counts, groups, 1.0)
    return per_q / np.maximum(counts, 1.0)


RULES = {"arithmetic": arithmetic, "clf-pairs+gen": clf_pairs_gen}


NO_LOGPROBS = False          # set by --no-logprobs
LOGPROBS_NAME = "logprobs_hf.json"


def attach_logprobs(exp: str, records: list[dict], k: int) -> bool:
    # the "+gen" log-prob feature, from the run's LOGPROBS_NAME file if present
    if NO_LOGPROBS:
        return False
    path = RESULTS / exp / LOGPROBS_NAME
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    for rec in records:
        rec["logprob"] = [float(x) for x in stored[rec["idx"]][:k]]
    return True


def score(exp: str, k: int) -> dict[str, tuple[float, float]]:
    records = load_run(exp, k)
    if not attach_logprobs(exp, records, k):
        why = ("--no-logprobs" if NO_LOGPROBS
               else f"no {LOGPROBS_NAME} (write it with analysis/logprobs_hf.py)")
        print(f"[warn] {exp}: {why}, so clf-pairs+gen drops the '+gen' feature "
              f"and fits the similarities alone", file=sys.stderr)
    acc = np.asarray([np.mean(r["correct"]) for r in records], dtype=float)
    labels = (acc >= 0.5).astype(int)
    out = {}
    for name, fn in RULES.items():
        conf = fn(records)
        auroc = float(roc_auc_score(labels, conf)) if 0 < labels.mean() < 1 else float("nan")
        out[name] = (expected_calibration_error(conf, acc, ECE_BINS), auroc)
    return out


def collect(k: int, model: str = MODEL) -> dict:
    table = {}
    for method, pattern in SEMANTIC_TEMPLATES.items():
        per_ds = {}
        for ds, _ in DATASETS:
            exp = pattern.format(model=model, dataset=ds)
            per_ds[ds] = score(exp, k)
            print(f"[done] {exp:40} " + "  ".join(
                f"{n}: ECE {v[0]:.4f} AUROC {v[1]:.4f}" for n, v in per_ds[ds].items()),
                file=sys.stderr, flush=True)
        for rule in RULES:
            cells = {ds: per_ds[ds][rule] for ds, _ in DATASETS}
            cells["avg"] = (float(np.mean([v[0] for v in cells.values()])),
                            float(np.mean([v[1] for v in cells.values()])))
            table[(rule, method)] = cells
    return table


def render(table: dict, latex: bool) -> None:
    cols = [d for d, _ in DATASETS] + ["avg"]
    for rule in RULES:
        best_e = {c: min(table[(rule, m)][c][0] for m in SEMANTIC_TEMPLATES) for c in cols}
        best_a = {c: max(table[(rule, m)][c][1] for m in SEMANTIC_TEMPLATES) for c in cols}
        if not latex:
            print(f"\n### {rule}")
            print("| Method | " + " | ".join(t for _, t in DATASETS) + " | Avg |")
            print("|" + "---|" * (len(cols) + 1))
        for method in SEMANTIC_TEMPLATES:
            cells = []
            for c in cols:
                e, a = table[(rule, method)][c]
                if latex:
                    se = f"\\textbf{{{e:.4f}}}" if abs(e - best_e[c]) < 1e-12 else f"{e:.4f}"
                    sa = f"\\textbf{{{a:.4f}}}" if abs(a - best_a[c]) < 1e-12 else f"{a:.4f}"
                    cells.append(f"{se} / {sa}")
                else:
                    cells.append(f"{e:.4f} / {a:.4f}")
            if latex:
                label = f"\\textbf{{{method}}}" if method == "CSR" else method
                print(f"{label} & " + " & ".join(cells) + "\\\\")
            else:
                print(f"| {method} | " + " | ".join(cells) + " |")


def main() -> None:
    global NO_LOGPROBS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--no-logprobs", action="store_true",
                        help="drop the '+gen' log-probability feature and fit the "
                             "pairwise similarities alone, so every backbone is "
                             "scored by the same recipe")
    parser.add_argument("--model", default=MODEL,
                        help="which backbone's runs to score "
                             "(qwen7b, llama8b, mistral7b)")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    NO_LOGPROBS = args.no_logprobs
    render(collect(args.k, args.model), args.latex)


if __name__ == "__main__":
    main()

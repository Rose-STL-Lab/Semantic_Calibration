"""Table 7's clf-pairs+gen re-fit under SIMBA-UQ's own protocol, as a robustness check.

Depth-4 forest, a random 50/50 question split repeated 5 times, mean over the repeats
with min/max as the spread; aggregation.py fits the same features more strictly.

    python analysis/simba.py --model llama8b
    python analysis/simba.py --all --latex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.aggregation import attach_logprobs, sim_matrix  # noqa: E402
from analysis.runs import DATASETS, MODELS, RUN_TEMPLATES, run_name  # noqa: E402
from analysis.estimators import load_run  # noqa: E402
from csr.metrics import expected_calibration_error  # noqa: E402

ECE_BINS = 10
N_REPEATS = 5          # "repeat the experiment 5 times"
MAX_DEPTH = 4          # "a random forest with a maximum depth of 4"
N_TREES = 100          # sklearn's default; the paper does not state one


def features(records: list[dict], sort_sims: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-rollout rows: the K-1 pairwise similarities plus exp(avg token logit)."""
    feats, labels, groups = [], [], []
    for qi, rec in enumerate(records):
        m = sim_matrix(rec["answers"])
        k = len(m)
        lp = rec.get("logprob")
        for i in range(k):
            sims = [float(m[i, j]) for j in range(k) if j != i]
            if sort_sims:
                sims = sorted(sims)
            row = sims + ([float(np.exp(lp[i]))] if lp is not None else [])
            feats.append(row)
            labels.append(rec["correct"][i])
            groups.append(qi)
    return (np.asarray(feats, dtype=float),
            np.asarray(labels, dtype=int),
            np.asarray(groups, dtype=int))


def simba_scores(records: list[dict], sort_sims: bool, seed: int) -> np.ndarray:
    """Question-level confidence on the held-out half, or NaN for the train half."""
    X, y, groups = features(records, sort_sims)
    n_q = len(records)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_q)
    test_q = set(order[: n_q // 2].tolist())          # a random half of the questions
    test = np.fromiter((g in test_q for g in groups), dtype=bool, count=len(groups))

    conf = np.full(len(y), np.nan)
    if len(set(y[~test].tolist())) < 2:
        conf[test] = float(y[~test].mean()) if (~test).any() else 0.0
    else:
        rf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                    random_state=seed, n_jobs=8)
        rf.fit(X[~test], y[~test])
        proba = rf.predict_proba(X[test])
        conf[test] = (proba[:, list(rf.classes_).index(1)]
                      if 1 in rf.classes_ else 0.0)

    per_q = np.full(n_q, np.nan)
    for q in test_q:
        per_q[q] = float(np.nanmean(conf[groups == q]))
    return per_q


def score(exp: str, k: int, sort_sims: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """(mean ECE, mean AUROC) and their (spread) over the five repeats."""
    records = load_run(exp, k)
    attach_logprobs(exp, records, k)
    acc = np.asarray([np.mean(r["correct"]) for r in records], dtype=float)
    labels = (acc >= 0.5).astype(int)

    eces, aurocs = [], []
    for rep in range(N_REPEATS):
        conf = simba_scores(records, sort_sims, seed=rep)
        m = ~np.isnan(conf)
        a, c, l = acc[m], conf[m], labels[m]
        eces.append(expected_calibration_error(c, a, ECE_BINS))
        aurocs.append(float(roc_auc_score(l, c)) if 0 < l.mean() < 1 else float("nan"))
    return ((float(np.mean(eces)), float(np.mean(aurocs))),
            (float(np.max(eces) - np.min(eces)), float(np.max(aurocs) - np.min(aurocs))))


def run(model: str, k: int, sort_sims: bool, latex: bool) -> None:
    cols = [d for d, _ in DATASETS] + ["avg"]
    table, spread = {}, {}
    for method, pattern in RUN_TEMPLATES.items():
        per_ds = {ds: score(pattern.format(model=model, dataset=ds), k, sort_sims)
                  for ds, _ in DATASETS}
        cells = {ds: per_ds[ds][0] for ds, _ in DATASETS}
        cells["avg"] = (float(np.mean([v[0] for v in cells.values()])),
                        float(np.mean([v[1] for v in cells.values()])))
        table[method] = cells
        spread[method] = {ds: per_ds[ds][1] for ds, _ in DATASETS}

    best_e = {c: min(table[m][c][0] for m in RUN_TEMPLATES) for c in cols}
    best_a = {c: max(table[m][c][1] for m in RUN_TEMPLATES) for c in cols}
    print(f"\n### {model}  (SIMBA setup, sims {'sorted' if sort_sims else 'unsorted'})")
    print("| Method | " + " | ".join(t for _, t in DATASETS) + " | Avg |")
    print("|" + "---|" * (len(cols) + 1))
    for method in RUN_TEMPLATES:
        cells = []
        for c in cols:
            e, a = table[method][c]
            se = f"**{e:.4f}**" if e == best_e[c] else f"{e:.4f}"
            sa = f"**{a:.4f}**" if a == best_a[c] else f"{a:.4f}"
            cells.append(f"{se} / {sa}")
        print(f"| {method} | " + " | ".join(cells) + " |")
    mx = max(max(v) for d in spread.values() for v in d.values())
    print(f"  (largest spread over the {N_REPEATS} repeats, any cell: {mx:.4f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--model", default="qwen7b", help="qwen7b, llama8b or mistral7b")
    p.add_argument("--all", action="store_true", help="every backbone in MODELS")
    p.add_argument("--sort-sims", action="store_true",
                   help="sort the similarity vector; the paper does not say to, "
                        "and rollout order is arbitrary, so this is the ambiguity")
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    for m in (MODELS if a.all else [a.model]):
        run(m, a.k, a.sort_sims, a.latex)


if __name__ == "__main__":
    main()

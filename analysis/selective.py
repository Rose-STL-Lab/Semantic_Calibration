"""Tables 5, 8 and 12, recomputed from this repo's evaluation runs.

Re-summarises records that already exist -- no GPU, no judge.

    python analysis/selective.py --table 5|8|12|all [--latex]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.runs import (  # noqa: E402
    DATASETS, MODELS, MODEL_LABELS, RUN_TEMPLATES, SEMANTIC_METHODS, run_name,
)

RESULTS = ROOT / "results"


def load(exp: str) -> dict:
    payload = json.loads((RESULTS / exp / "results.json").read_text())
    if payload.get("status") != "final":
        raise ValueError(f"{exp}: not final")
    return payload


def conf_acc(exp: str) -> tuple[np.ndarray, np.ndarray]:
    recs = load(exp)["records"]
    return (np.asarray([r["confidence"] for r in recs], dtype=float),
            np.asarray([r["accuracy"] for r in recs], dtype=float))


def selective(conf: np.ndarray, acc: np.ndarray) -> tuple[float, float]:
    """AURAC and AUGRC over every cut-off, with ties resolved in expectation.

    The label is the binarised 1[acc >= 0.5], not the rollout fraction the rest of
    analysis/ uses.
    """
    label = (acc >= 0.5).astype(float)
    n = len(label)
    counts = np.arange(1, n + 1)

    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    sorted_label = label[order]

    expected = np.empty(n)          # expected correct answers among the first i
    prefix = 0.0
    i = 0
    # ties in expectation: t accepted from a tied group of m holding k correct -> t*k/m
    while i < n:
        j = i
        while j < n and sorted_conf[j] == sorted_conf[i]:
            j += 1
        size = j - i
        correct = sorted_label[i:j].sum()
        expected[i:j] = prefix + np.arange(1, size + 1) * (correct / size)
        prefix += correct
        i = j

    aurac = float(np.mean(expected / counts))        # accuracy among accepted
    augrc = float(np.mean((counts - expected) / n))  # accepted AND wrong, over all
    return aurac, augrc


def table5(latex: bool) -> None:
    """Table 5: questions accepted in decreasing order of the method's own confidence."""
    for model in MODELS:
        rows = {}
        for method, pattern in RUN_TEMPLATES.items():
            cells = {}
            for ds, _ in DATASETS:
                cells[ds] = selective(*conf_acc(pattern.format(model=model, dataset=ds)))
            cells["avg"] = (float(np.mean([v[0] for v in cells.values()])),
                            float(np.mean([v[1] for v in cells.values()])))
            rows[method] = cells
        cols = [d for d, _ in DATASETS] + ["avg"]
        best_r = {c: max(rows[m][c][0] for m in RUN_TEMPLATES) for c in cols}
        best_g = {c: min(rows[m][c][1] for m in RUN_TEMPLATES) for c in cols}
        if latex:
            print(f"\\multirow{{5}}{{*}}{{{MODEL_LABELS[model]}}}")
        else:
            print(f"\n### {model}  AURAC↑ / AUGRC↓")
            print("| Method | " + " | ".join(t for _, t in DATASETS) + " | Avg |")
            print("|" + "---|" * (len(cols) + 1))
        for method in RUN_TEMPLATES:
            out = []
            for c in cols:
                r, g = rows[method][c]
                if latex:
                    # The table gives AURAC and AUGRC their own columns.
                    sr = f"\\textbf{{{r:.4f}}}" if abs(r - best_r[c]) < 1e-12 else f"{r:.4f}"
                    sg = f"\\textbf{{{g:.4f}}}" if abs(g - best_g[c]) < 1e-12 else f"{g:.4f}"
                    out.extend((sr, sg))
                else:
                    out.append(f"{r:.4f} / {g:.4f}")
            if latex:
                print(f"& {method} & " + " & ".join(out) + "\\\\")
            else:
                print(f"| {method} | " + " | ".join(out) + " |")
        if latex:
            print("\\midrule")


def table8(latex: bool) -> None:
    """Relative change vs Base, averaged over model families.

    (M-B)/B for accuracy and AUROC but (B-M)/B for ECE, so positive is always better.
    """
    def metrics(exp):
        s = load(exp)["summary"]
        return s["accuracy"], s["ece"], s["auroc"]

    acc_id, ece_id, auc_id, acc_ood, ece_ood, auc_ood = ({} for _ in range(6))
    for method in RUN_TEMPLATES:
        if method == "Base":
            continue
        rel = {k: [] for k in ("acc_id", "ece_id", "auc_id", "acc_ood", "ece_ood", "auc_ood")}
        for model in MODELS:
            b_id = metrics(run_name("Base", model, "hotpot"))
            m_id = metrics(run_name(method, model, "hotpot"))
            rel["acc_id"].append((m_id[0] - b_id[0]) / b_id[0] * 100)
            rel["ece_id"].append((b_id[1] - m_id[1]) / b_id[1] * 100)
            rel["auc_id"].append((m_id[2] - b_id[2]) / b_id[2] * 100)
            ood = [d for d, _ in DATASETS if d != "hotpot"]
            b = np.mean([metrics(run_name("Base", model, d)) for d in ood], axis=0)
            mm = np.mean([metrics(run_name(method, model, d)) for d in ood], axis=0)
            rel["acc_ood"].append((mm[0] - b[0]) / b[0] * 100)
            rel["ece_ood"].append((b[1] - mm[1]) / b[1] * 100)
            rel["auc_ood"].append((mm[2] - b[2]) / b[2] * 100)
        acc_id[method], ece_id[method], auc_id[method] = (np.mean(rel["acc_id"]),
                                                          np.mean(rel["ece_id"]),
                                                          np.mean(rel["auc_id"]))
        acc_ood[method], ece_ood[method], auc_ood[method] = (np.mean(rel["acc_ood"]),
                                                             np.mean(rel["ece_ood"]),
                                                             np.mean(rel["auc_ood"]))
    order = [m for m in RUN_TEMPLATES if m != "Base"]
    best = {name: max(d.values()) for name, d in
            (("acc_id", acc_id), ("ece_id", ece_id), ("auc_id", auc_id),
             ("acc_ood", acc_ood), ("ece_ood", ece_ood), ("auc_ood", auc_ood))}
    if not latex:
        print("\n### Table 8  change against Base (%), averaged over the three model families")
        print("| Method | ID Acc | ID ECE drop | ID AUROC | OOD Acc | OOD ECE drop | OOD AUROC |")
        print("|---|---|---|---|---|---|---|")
    for method in order:
        vals = [(acc_id[method], "acc_id"), (ece_id[method], "ece_id"), (auc_id[method], "auc_id"),
                (acc_ood[method], "acc_ood"), (ece_ood[method], "ece_ood"), (auc_ood[method], "auc_ood")]
        if latex:
            cells = []
            for v, key in vals:
                s = f"{v:+.1f}"
                cells.append(f"$\\mathbf{{{s}}}$" if abs(v - best[key]) < 1e-9 else f"${s}$")
            label = f"\\textbf{{{method}}}" if method == "CSR" else method
            print(f"{label:6} & " + " & ".join(cells) + "\\\\")
        else:
            print(f"| {method} | " + " | ".join(f"{v:+.1f}" for v, _ in vals) + " |")


def table12(latex: bool) -> None:
    """Table 12: fraction of questions whose verbalized confidence parsed and was in range."""
    for model in MODELS:
        if latex:
            print(f"\\multirow{{5}}{{*}}{{{model.capitalize()}}}")
        else:
            print(f"\n### {model}  parsing rate")
            print("| Method | " + " | ".join(t for _, t in DATASETS) + " |")
            print("|" + "---|" * (len(DATASETS) + 1))
        for method, pattern in RUN_TEMPLATES.items():
            cells = []
            for ds, _ in DATASETS:
                if method in SEMANTIC_METHODS:
                    cells.append("1.00")   # no confidence token to parse
                else:
                    recs = load(pattern.format(model=model, dataset=ds))["records"]
                    rate = float(np.mean([bool(r.get("format_ok", True)) for r in recs]))
                    cells.append(f"{rate:.3f}")
            if latex:
                print(f"& {method:5} & " + " & ".join(cells) + "\\\\")
            else:
                print(f"| {method} | " + " | ".join(cells) + " |")
        if latex:
            print("\\midrule")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", choices=["5", "8", "12", "all"], default="all")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    if args.table in ("5", "all"):
        table5(args.latex)
    if args.table in ("8", "all"):
        table8(args.latex)
    if args.table in ("12", "all"):
        table12(args.latex)


if __name__ == "__main__":
    main()

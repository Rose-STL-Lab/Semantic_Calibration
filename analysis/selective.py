"""Selective prediction (AURAC / AUGRC) and verbalized-confidence parse rates.

Recomputed from existing evaluation records; needs no GPU or judge.

    python analysis/selective.py --table selective|parse|all [--latex]
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
    # a question counts as correct when acc >= 0.5
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


def selective_table(latex: bool) -> None:
    # accepted in decreasing order of the method's own confidence
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


def parse_rate_table(latex: bool) -> None:
    # fraction whose verbalized confidence parsed and was in range
    for model in MODELS:
        if latex:
            print(f"\\multirow{{5}}{{*}}{{{MODEL_LABELS[model]}}}")
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
    parser.add_argument("--table", choices=["selective", "parse", "all"], default="all")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    if args.table in ("selective", "all"):
        selective_table(args.latex)
    if args.table in ("parse", "all"):
        parse_rate_table(args.latex)


if __name__ == "__main__":
    main()

"""Table 8 derived from the paper's Table 1, not from the raw runs.

Table 8 is defined as a transformation of Table 1, and some Table 1 cells are
not reproducible from this repo's ``results/``, so the Table 1 body is parsed
out of ``sections/experiments.tex`` instead of being recomputed.

    python analysis/tradeoff.py --tex <experiments.tex> [--latex]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Path to the LaTeX source that holds Table 1; pass it with --tex.
DEFAULT_TEX = None
MODELS = ["Llama", "Qwen", "Mistral"]
METHODS = ["Base", "RLVR", "RD", "RLCR", "CSR"]
DATASETS = ["hotpot", "trivia", "msmarco", "nq_open"]      # column order in Table 1
CLEAN = re.compile(r"\\textbf\{|\\underline\{|\}|\$|\\mathbf\{")


def parse_table1(path: Path) -> dict[tuple[str, str], dict[str, tuple[float, float, float]]]:
    """{(model, method): {dataset: (acc, ece, auroc)}} from the Table 1 body."""
    lines = path.read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if "label{tab:main_results}" in l)
    end = next(i for i in range(start, len(lines)) if "\\bottomrule" in lines[i])
    table, model = {}, None
    for line in lines[start:end]:
        stripped = line.strip()
        if "multirow" in stripped:
            model = next(m for m in MODELS if m in stripped)
            continue
        if not stripped.startswith("&") or model is None:
            continue
        cells = [CLEAN.sub("", c).strip() for c in stripped.rstrip("\\").split("&")[1:]]
        method = cells[0].strip()
        if method not in METHODS:
            continue
        values = [float(c) for c in cells[1:13]]           # 4 datasets x (acc, ece, auroc)
        table[(model, method)] = {
            ds: tuple(values[3 * i: 3 * i + 3]) for i, ds in enumerate(DATASETS)
        }
    missing = [(m, meth) for m in MODELS for meth in METHODS if (m, meth) not in table]
    if missing:
        raise ValueError(f"Table 1 rows not parsed: {missing}")
    return table


def relative(table, method: str) -> list[float]:
    """[acc, ece-red, auroc] for HotpotQA then for the OOD mean, in percent."""
    out = []
    for scope in ("hotpot", "ood"):
        acc, ece, auroc = [], [], []
        for model in MODELS:
            base, mine = table[(model, "Base")], table[(model, method)]
            if scope == "hotpot":
                b, m = np.array(base["hotpot"]), np.array(mine["hotpot"])
            else:
                ood = [d for d in DATASETS if d != "hotpot"]
                b = np.mean([base[d] for d in ood], axis=0)
                m = np.mean([mine[d] for d in ood], axis=0)
            acc.append((m[0] - b[0]) / b[0] * 100)
            ece.append((b[1] - m[1]) / b[1] * 100)          # reduction: positive is better
            auroc.append((m[2] - b[2]) / b[2] * 100)
        out += [float(np.mean(acc)), float(np.mean(ece)), float(np.mean(auroc))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, required=True)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    table = parse_table1(args.tex)
    rows = {m: relative(table, m) for m in METHODS if m != "Base"}
    best = [max(v[i] for v in rows.values()) for i in range(6)]

    if not args.latex:
        print("### Table 8 (from the Table 1 cells), change against Base in %")
        print("| Method | ID Acc | ID ECE drop | ID AUROC | OOD Acc | OOD ECE drop | OOD AUROC |")
        print("|---|---|---|---|---|---|---|")
    for method, vals in rows.items():
        if args.latex:
            cells = [f"$\\mathbf{{{v:+.1f}}}$" if abs(v - best[i]) < 1e-9 else f"${v:+.1f}$"
                     for i, v in enumerate(vals)]
            label = f"\\textbf{{{method}}}" if method == "CSR" else method
            print(f"{label:6} & " + " & ".join(cells) + "\\\\")
        else:
            print(f"| {method} | " + " | ".join(f"{v:+.1f}" for v in vals) + " |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Collect every ``results/*/results.json`` into one table.

    python scripts/summarize.py                        # markdown to stdout
    python scripts/summarize.py --csv results/all.csv  # also write a CSV
    python scripts/summarize.py --model qwen --metric ece
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import DATASETS, METHODS, MODELS  # noqa: E402

COLUMNS = ["accuracy", "ece", "auroc", "brier", "avg_confidence", "avg_total_tokens"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None, help="results directory (default: ./results)")
    parser.add_argument("--model", action="append", default=None, choices=MODELS)
    parser.add_argument("--method", action="append", default=None, choices=METHODS)
    parser.add_argument("--dataset", action="append", default=None, choices=DATASETS)
    parser.add_argument("--metric", action="append", default=None, choices=COLUMNS,
                        help="restrict the reported columns (repeatable)")
    parser.add_argument("--csv", default=None, help="also write the table to this CSV path")
    return parser.parse_args()


def load_rows(results_dir, wanted):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "results.json"),
                                 recursive=True)):
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue

        summary = payload.get("summary", {})
        row = {
            "model": payload.get("model", "?"),
            "method": payload.get("method", "?"),
            "dataset": payload.get("dataset", "?"),
            "checkpoint": os.path.basename(str(payload.get("checkpoint", "base"))),
            "n": summary.get("n"),
        }
        for key in wanted:
            row[key] = summary.get(key)
        rows.append(row)

    # paper main-table order: model, method, dataset (in-distribution first)
    def sort_key(r):
        def rank(value, order):
            return order.index(value) if value in order else len(order)
        return (rank(r["model"], MODELS), rank(r["method"], METHODS),
                rank(r["dataset"], DATASETS), r["checkpoint"])

    return sorted(rows, key=sort_key)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def main():
    args = parse_args()
    results_dir = args.results or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    wanted = args.metric or COLUMNS

    rows = load_rows(results_dir, wanted)
    for key, allowed in (("model", args.model), ("method", args.method),
                         ("dataset", args.dataset)):
        if allowed:
            rows = [r for r in rows if r[key] in allowed]

    if not rows:
        print(f"no results.json found under {results_dir}", file=sys.stderr)
        return 1

    header = ["model", "method", "dataset", "checkpoint", "n"] + wanted
    widths = [max(len(h), *(len(fmt(r.get(h))) for r in rows)) for h in header]

    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(h)).ljust(w)
                                for h, w in zip(header, widths)) + " |")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[saved] {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

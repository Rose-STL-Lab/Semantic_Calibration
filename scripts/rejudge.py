#!/usr/bin/env python
"""Re-score an existing RD / RLCR evaluation with a different judge, without regenerating.

Only ``correct``/``accuracy`` are recomputed (same rule as eval_verbalized.py:
no extracted answer -> 0, no judge call); answers, confidences and format flags
are copied from the source. The judge comes entirely from ``--set judge.*``.

    python scripts/rejudge.py --method RD --model qwen --dataset hotpot --src <results/ dir> \
        --exp-name <new run dir> --set judge.base_url=http://127.0.0.1:8310/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import add_common_args, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--src", required=True, help="source results/<exp> directory name")
    parser.add_argument("--chunk", type=int, default=200, help="questions per judge batch")
    args = parser.parse_args()
    if not args.exp_name:
        raise SystemExit("--exp-name is required (never overwrite the source run)")
    if args.exp_name == args.src:
        raise SystemExit("--exp-name must differ from --src")

    cfg = load_config(args.method, args.model, args.dataset, args.overrides)
    from csr.data import correctness_prompt_kind
    from csr.judge import make_judge
    from csr.metrics import summarize
    from csr.results import RunDir, print_summary

    src_path = os.path.join(cfg["paths"]["results"], args.src, "results.json")
    src = json.load(open(src_path))
    if src.get("status") != "final":
        raise SystemExit(f"{src_path} is not final")
    markers = cfg["model"].get("chat_end_markers", [])
    kind = correctness_prompt_kind(cfg)
    judge = make_judge(cfg, chat_end_markers=markers)

    records = [dict(r) for r in src["records"]]
    old = [int(r["correct"]) for r in records]
    todo = [r for r in records if r.get("extracted_answer")]
    for r in records:
        if not r.get("extracted_answer"):
            r["correct"], r["accuracy"] = 0, 0.0
    for start in range(0, len(todo), args.chunk):
        block = todo[start:start + args.chunk]
        golds = [r["gold_answers"] if len(r["gold_answers"]) > 1 else r["gold_answers"][0] for r in block]
        labels = judge.score_against_gold_batch([[r["extracted_answer"]] for r in block], golds,
                                                [r["question"] for r in block], kind)
        for r, lab in zip(block, labels):
            r["correct"], r["accuracy"] = int(lab[0]), float(lab[0])
        print(f"  judged {min(start + args.chunk, len(todo))}/{len(todo)}", flush=True)

    new = [int(r["correct"]) for r in records]
    summary = summarize([r["confidence"] for r in records], [r["accuracy"] for r in records],
                        int(cfg["method"]["eval"]["ece_bins"]))
    summary.update({
        "num_rollouts": 1,
        "checkpoint_step": src.get("summary", {}).get("checkpoint_step"),
        "confidence_format_adherence": float(sum(bool(r.get("format_ok")) for r in records) / max(1, len(records))),
        "judge": judge.stats_summary(),
        "rejudged_from": args.src,
        # fraction of labels unchanged relative to the source run's judge
        "agreement_with_source_labels": float(sum(a == b for a, b in zip(old, new)) / max(1, len(new))),
    })
    run = RunDir(cfg["paths"]["results"], args.exp_name)
    run.save_config(cfg, checkpoint=None)
    path = run.save_results(cfg, summary, records, checkpoint=None)
    print_summary(summary, title=args.exp_name)
    print(f"  source acc {sum(old)/len(old):.4f} -> rejudged acc {sum(new)/len(new):.4f}; "
          f"label agreement {summary['agreement_with_source_labels']:.4f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()

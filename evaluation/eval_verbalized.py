#!/usr/bin/env python
"""Verbalized-confidence evaluation for RD and RLCR.

One greedy generation per question; confidence is parsed from the model's own
output (RD: ``Answer: …, Confidence: <0-10>``; RLCR: ``<confidence>…</confidence>``)
and correctness comes from the same LLM judge the answer-only methods use.

    python evaluation/eval_verbalized.py --method RD --model qwen --dataset trivia \
        --ckpt ckpt/RD/<exp-name>/model_finetuned_step_1250   # omit --ckpt for the frozen base
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import add_common_args, default_exp_name, load_config, seed_everything
from csr.parsing import UNPARSED_CONFIDENCE

VERBALIZED_METHODS = {"RD", "RLCR"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint dir (LoRA adapter). "
                             "Omit to evaluate the frozen base model.")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing results.json and start over")
    return parser.parse_args()


# mean over the records that carry `key`
def _avg(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return float(sum(values) / len(values)) if values else float("nan")


def main():
    args = parse_args()
    if args.method not in VERBALIZED_METHODS:
        raise SystemExit(f"--method {args.method} is answer-only; "
                         f"use evaluation/eval_semantic.py instead")

    cfg = load_config(args.method, args.model, args.dataset, args.overrides)
    seed_everything(int(cfg["seed"]))

    from tqdm.auto import tqdm

    from csr.data import correctness_prompt_kind, load_split
    from csr.generation import count_tokens, generate
    from csr.judge import make_judge
    from csr.metrics import summarize
    from csr.modeling import checkpoint_step, load_for_eval, resolve_checkpoints
    from csr.results import RunDir, print_summary
    from csr.generation import render_generation_prompt
    from csr.parsing import parse_verbalized_output

    eval_cfg = cfg["method"]["eval"]
    conf_cfg = cfg["method"]["confidence"]
    tmpl_kwargs = cfg["model"].get("chat_template_kwargs") or {}
    markers = cfg["model"].get("chat_end_markers", [])
    prompt_kind = correctness_prompt_kind(cfg)
    separators = cfg["model"].get("verbalized_separators", [","])
    seed = int(cfg["seed"])

    checkpoints = resolve_checkpoints(args.ckpt)
    dataset = load_split(cfg, split="eval")

    for checkpoint in checkpoints:
        # reloaded per checkpoint; the LoRA adapter comes back as lora_request
        model, tokenizer, lora_request = load_for_eval(
            cfg, int(eval_cfg["max_seq_length"]), checkpoint=checkpoint)

        tag = os.path.basename(checkpoint.rstrip("/")) if checkpoint else "BASE"
        exp_name = args.exp_name or default_exp_name(cfg, checkpoint)

        run = RunDir(cfg["paths"]["results"], exp_name)
        if not args.fresh:
            run.check_resume(checkpoint)
        run.save_config(cfg, checkpoint=checkpoint)
        if args.fresh:
            run.reset_records()

        judge = make_judge(cfg, chat_end_markers=markers)
        n = len(dataset)
        records = run.load_records(limit=n)
        if records:
            print(f"[resume] {len(records)} questions already done in {run.records_path}")

        pending: list[dict] = []
        print(f"\n{'=' * 78}\n[eval] {exp_name}  ({tag}, verbalized)\n{'=' * 78}")

        for i in tqdm(range(len(records), n), initial=len(records), total=n,
                      desc=f"{cfg['method']['name']} {cfg['dataset']['name']}"):
            example = dataset[i]
            question = example["question"]
            gold = example["answer"]
            prompt = render_generation_prompt(example["prompt"], tokenizer, tmpl_kwargs)

            raw = generate(model, tokenizer, prompt, n=1,
                           temperature=float(eval_cfg["temperature"]),
                           top_p=float(eval_cfg["top_p"]),
                           max_new_tokens=int(eval_cfg["max_new_tokens"]),
                           seed=seed, lora_request=lora_request)[0]

            answer, confidence, confidence_raw = parse_verbalized_output(
                raw, style=conf_cfg["style"], separators=separators,
                chat_end_markers=markers, scale=float(conf_cfg["scale"]),
            )

            correct = 0
            if answer:
                correct = int(judge.score_against_gold([answer], gold, question, prompt_kind)[0])

            prompt_tokens = count_tokens(tokenizer, prompt)
            completion_tokens = count_tokens(tokenizer, raw)

            records.append({
                "idx": i,
                "question": question,
                "gold_answers": gold if isinstance(gold, list) else [gold],
                "full_answer": raw,
                "extracted_answer": answer,
                "semantic_id": None,          # verbalized methods do not cluster
                "confidence": float(confidence) if confidence is not None
                              else UNPARSED_CONFIDENCE,
                "confidence_raw": confidence_raw,
                "format_ok": bool(answer is not None and confidence is not None),
                "correct": correct,
                "accuracy": float(correct),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })
            pending.append(records[-1])

            if len(pending) >= 20:
                run.append_records(pending)
                pending = []

            if (i + 1) % 25 == 0:
                partial = summarize([r["confidence"] for r in records],
                                    [r["accuracy"] for r in records],
                                    int(eval_cfg["ece_bins"]))
                tqdm.write(f"  [{i + 1}/{n}] acc={partial['accuracy']:.4f} "
                           f"ece={partial['ece']:.4f} "
                           f"auroc={partial['auroc'] if partial['auroc'] is not None else 'NA'}")

        run.append_records(pending)

        summary = summarize([r["confidence"] for r in records],
                            [r["accuracy"] for r in records],
                            int(eval_cfg["ece_bins"]))
        summary.update({
            "num_rollouts": 1,
            "checkpoint_step": checkpoint_step(checkpoint),
            "confidence_format_adherence": float(
                sum(bool(r["format_ok"]) for r in records) / max(1, len(records))),
            "avg_prompt_tokens": _avg(records, "prompt_tokens"),
            "avg_completion_tokens": _avg(records, "completion_tokens"),
            "avg_total_tokens": _avg(records, "total_tokens"),
            "judge": judge.stats_summary(),
        })

        judge.assert_parsed()
        path = run.save_results(cfg, summary, records, checkpoint=checkpoint)
        print_summary(summary, title=exp_name)
        print(f"[saved] {path}")

        del model


if __name__ == "__main__":
    main()

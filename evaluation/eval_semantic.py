#!/usr/bin/env python
"""Answer-only evaluation for Base, RLVR and CSR.

K rollouts per question; confidence is ``exp(-SE)`` over the judge's semantic
clusters, question accuracy the fraction of rollouts judged correct.

    python evaluation/eval_semantic.py --method CSR --model qwen --dataset trivia \
        --ckpt ckpt/CSR/qwen7b/checkpoint-2500   # omit --ckpt for the frozen base
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import add_common_args, default_exp_name, load_config, seed_everything, setup_env

SEMANTIC_METHODS = {"Base", "RLVR", "CSR"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--ckpt", default=None,
                        help="LoRA checkpoint dir, or a parent dir holding several. "
                             "Omit to evaluate the frozen base model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions (debugging)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing results.json and start over")
    parser.add_argument("--question-batch", type=int, default=1,
                        help="questions generated and judged together (default 1 = one at a "
                             "time). Larger values keep a self-hosted judge busy; each "
                             "question's rollouts, clustering and scoring are unchanged.")
    parser.add_argument("--phase", choices=("both", "generate", "judge"), default="both",
                        help="'both' (default) generates and judges in one alternating pass. "
                             "'generate' writes the rollouts and stops, holding the GPU but "
                             "never calling the judge; 'judge' reads those rollouts back, "
                             "scores them and writes the final results, needing no GPU. "
                             "Splitting is for large K, where a run would otherwise leave "
                             "the GPU idle during O(K^2) pairwise judging and the judge idle "
                             "during generation.")
    return parser.parse_args()


# mean over the records that carry `key`: records seeded from a training-time
# validation pass have no token accounting, so a plain sum() would raise on None
def _avg(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return float(sum(values) / len(values)) if values else float("nan")


def main():
    args = parse_args()
    if args.method not in SEMANTIC_METHODS:
        raise SystemExit(f"--method {args.method} is verbalized; "
                         f"use evaluation/eval_verbalized.py instead")
    if args.method == "Base" and args.ckpt:
        raise SystemExit("--method Base evaluates the frozen model and takes no --ckpt")

    cfg = load_config(args.method, args.model, args.dataset, args.overrides)
    setup_env(cfg)
    seed_everything(int(cfg["seed"]))

    # imported after setup_env so the HF cache env is in place
    from tqdm.auto import tqdm

    from csr.data import correctness_prompt_kind, load_split
    from csr.generation import rollouts_with_tokens
    from csr.judge import make_judge
    from csr.metrics import summarize
    from csr.results import RunDir, checkpoint_step, print_summary, resolve_checkpoints
    from csr.semantic import semantic_confidence
    from csr.text import extract_completion_text, render_generation_prompt

    eval_cfg = cfg["method"]["eval"]
    markers = cfg["model"].get("chat_end_markers", [])
    seed = int(cfg["seed"])
    k = int(eval_cfg["num_rollouts"])

    prompt_kind = correctness_prompt_kind(cfg)
    checkpoints = resolve_checkpoints(args.ckpt)
    dataset = load_split(cfg, split="eval", size=args.limit)

    # judge phase re-reads rollouts from disk, so it must not take a GPU --
    # csr.modeling imports unsloth, which refuses to load without one
    if args.phase == "judge":
        model = tokenizer = None
    else:
        from csr.modeling import load_for_eval
        model, tokenizer, _ = load_for_eval(cfg, int(eval_cfg["max_seq_length"]),
                                            checkpoint=None)

    for checkpoint in checkpoints:
        lora_request = (model.load_lora(os.path.abspath(checkpoint))
                        if checkpoint and model is not None else None)
        tag = os.path.basename(checkpoint.rstrip("/")) if checkpoint else "BASE"
        exp_name = args.exp_name or default_exp_name(cfg, checkpoint)

        run = RunDir(cfg["paths"]["results"], exp_name)
        run.save_config(cfg, checkpoint=checkpoint)
        if args.fresh:
            run.reset_records()

        judge = make_judge(cfg, chat_end_markers=markers)
        n = len(dataset)
        records = run.load_records(limit=n)
        if records:
            print(f"[resume] {len(records)} questions already done in {run.records_path}")
        unjudged = sum(1 for r in records if "confidence" not in r)
        if unjudged and args.phase == "both":
            raise SystemExit(
                f"{run.records_path} holds {unjudged} unjudged records from a "
                f"--phase generate run; finish them with --phase judge (the "
                f"alternating pass cannot resume into the middle of a split run)")
        if not records and args.phase == "judge":
            raise SystemExit(f"{run.records_path} has no records to judge; "
                             f"run --phase generate first")

        pending: list[dict] = []
        print(f"\n{'=' * 78}\n[eval] {exp_name}  ({tag}, K={k})\n{'=' * 78}")

        from csr.generation import rollouts_with_tokens_batch
        from csr.semantic import semantic_confidence_batch

        qb = max(1, int(args.question_batch))
        gen_kwargs = dict(
            n=k,
            temperature=float(eval_cfg["temperature"]),
            top_p=float(eval_cfg["top_p"]),
            max_new_tokens=int(eval_cfg["max_new_tokens"]),
            seed=seed,
            lora_request=lora_request,
        )

        # one record builder for all phases: "both" judges inline, the split
        # phases call make_record and judge_record from different processes
        def make_record(i, example, gen, answers):
            gold = example["answer"]
            return {
                "idx": i,
                "question": example["question"],
                "gold_answers": gold if isinstance(gold, list) else [gold],
                "rollouts": [
                    {
                        "full_answer": raw,
                        "extracted_answer": answer,
                        "completion_tokens": int(ctok),
                    }
                    for raw, answer, ctok in zip(gen["raw"], answers,
                                                 gen["completion_tokens"])
                ],
                "prompt_tokens": gen["prompt_tokens"],
                "total_tokens": gen["total_tokens"],
            }

        def judge_record(record, clusters, labels):
            """Fill in everything the judge decides, in place."""
            for rollout, cid, label in zip(record["rollouts"],
                                           clusters["cluster_ids"], labels):
                rollout["semantic_id"] = int(cid)
                rollout["correct"] = int(label)
            record["semantic_ids"] = [int(c) for c in clusters["cluster_ids"]]
            record["num_semantic_clusters"] = clusters["num_clusters"]
            record["semantic_entropy"] = clusters["semantic_entropy"]
            record["confidence"] = clusters["confidence"]
            record["accuracy"] = float(sum(labels) / max(1, len(labels)))
            return record

        def judge_block(recs):
            """Cluster and score a block of already-generated records."""
            answers_list = [[r["extracted_answer"] for r in rec["rollouts"]] for rec in recs]
            questions_b = [rec["question"] for rec in recs]
            golds = [rec["gold_answers"][0] if len(rec["gold_answers"]) == 1
                     else rec["gold_answers"] for rec in recs]
            if len(recs) == 1:
                clusters_list = [semantic_confidence(answers_list[0], questions_b[0], judge)]
                labels_list = [judge.score_against_gold(answers_list[0], golds[0],
                                                        questions_b[0], prompt_kind)]
            else:
                clusters_list = semantic_confidence_batch(answers_list, questions_b, judge)
                labels_list = judge.score_against_gold_batch(
                    answers_list, golds, questions_b, prompt_kind)
            for rec, clusters, labels in zip(recs, clusters_list, labels_list):
                judge_record(rec, clusters, labels)

        if args.phase == "judge":
            # rollouts are on disk; only unjudged records still need the judge
            todo = [r for r in records if "confidence" not in r]
            print(f"[judge] {len(todo)} of {len(records)} records still unjudged")
            bar = tqdm(total=len(records), initial=len(records) - len(todo),
                       desc=f"judge {cfg['dataset']['name']}")
            for pos in range(0, len(todo), qb):
                judge_block(todo[pos:pos + qb])
                bar.update(len(todo[pos:pos + qb]))
                if (pos // qb) % max(1, 20 // qb) == 0:
                    run.rewrite_records(records)
            bar.close()
            run.rewrite_records(records)
        else:
            bar = tqdm(total=n, initial=len(records),
                       desc=f"{cfg['method']['name']} {cfg['dataset']['name']}")
            start = len(records)
            while start < n:
                block = list(range(start, min(start + qb, n)))
                examples = [dataset[i] for i in block]
                prompts = [render_generation_prompt(ex["prompt"], tokenizer) for ex in examples]

                if qb == 1:
                    gens = [rollouts_with_tokens(model, tokenizer, prompts[0], **gen_kwargs)]
                else:
                    gens = rollouts_with_tokens_batch(model, tokenizer, prompts, **gen_kwargs)
                answers_list = [[extract_completion_text(text, markers) for text in g["raw"]]
                                for g in gens]

                new = [make_record(i, example, gen, answers)
                       for i, example, gen, answers in zip(block, examples, gens, answers_list)]
                if args.phase == "both":
                    judge_block(new)

                for i, record in zip(block, new):
                    records.append(record)
                    pending.append(record)

                    if len(pending) >= 20:
                        run.append_records(pending)
                        pending = []

                    if args.phase == "both" and (i + 1) % 25 == 0:
                        partial = summarize([r["confidence"] for r in records],
                                            [r["accuracy"] for r in records],
                                            int(eval_cfg["ece_bins"]))
                        tqdm.write(f"  [{i + 1}/{n}] acc={partial['accuracy']:.4f} "
                                   f"ece={partial['ece']:.4f} "
                                   f"auroc={partial['auroc'] if partial['auroc'] is not None else 'NA'}")

                bar.update(len(block))
                start += len(block)
            bar.close()

            run.append_records(pending)

        if args.phase == "generate":
            print(f"[generate] {len(records)} questions written unjudged to "
                  f"{run.records_path}\n[generate] score them with the same command plus "
                  f"--phase judge (no GPU needed)")
            continue

        summary = summarize([r["confidence"] for r in records],
                            [r["accuracy"] for r in records],
                            int(eval_cfg["ece_bins"]))
        summary.update({
            "num_rollouts": k,
            "checkpoint_step": checkpoint_step(checkpoint),
            "avg_semantic_clusters": float(
                sum(r["num_semantic_clusters"] for r in records) / max(1, len(records))),
            "avg_prompt_tokens": _avg(records, "prompt_tokens"),
            "avg_total_tokens": _avg(records, "total_tokens"),
            "judge": judge.stats_summary(),
        })

        path = run.save_results(cfg, summary, records, checkpoint=checkpoint)
        print_summary(summary, title=exp_name)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()

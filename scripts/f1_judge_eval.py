#!/usr/bin/env python3
"""Build F1-train + F1-eval runs by reclustering saved rollouts with token-F1 equivalence.

Rollouts and correctness labels are reused from the source run; only clusters and
confidence change. Each result goes to a separate *_semf1_k16 directory.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csr.f1_judge import F1EquivalenceJudge
from csr.metrics import summarize
from csr.semantic import semantic_confidence


RESULTS = ROOT / "results"
# per-family F1 thresholds (tau)
MODELS = {
    "llama8b": {"tag": "050", "threshold": 0.50},
    "qwen7b": {"tag": "080", "threshold": 0.80},
    "mistral7b": {"tag": "050", "threshold": 0.50},
}
DATASETS = ("hotpot", "trivia", "msmarco", "nq_open")


def extracted_answers(record: dict) -> list[str]:
    answers = []
    for rollout in record["rollouts"]:
        answer = (
            rollout.get("extracted_answer")
            or rollout.get("answer")
            or rollout.get("full_answer")
            or ""
        )
        answers.append(str(answer))
    return answers


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="model families to rebuild (default: all)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS), help="datasets to rebuild (default: all)")
    args = ap.parse_args()

    manifest = {
        "schema": "csr-f1train-f1eval-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "F1-trained CSR checkpoint; existing K=16 rollouts reclustered with "
            "the same model-family F1 threshold; correctness labels retained "
            "from the source run's judge"
        ),
        "runs": [],
        "macro_averages": {},
    }

    for model, cfg in ((m, MODELS[m]) for m in args.models):
        tag = cfg["tag"]
        threshold = float(cfg["threshold"])
        model_rows = []
        for dataset in args.datasets:
            source_name = f"CSR_{model}_{dataset}_f1eq{tag}_step2500_nano_k16"
            target_name = f"CSR_{model}_{dataset}_f1eq{tag}_step2500_nano_semf1_k16"
            source_path = RESULTS / source_name / "results.json"
            target_path = RESULTS / target_name / "results.json"
            if not source_path.exists():
                raise SystemExit(
                    f"missing {source_path}\n"
                    f"produce it first with:\n"
                    f"  python evaluation/eval_semantic.py --method CSR --model <key> "
                    f"--dataset {dataset} --exp-name {source_name} "
                    f"--ckpt <f1-trained checkpoint> "
                    f"--set judge.equivalence=f1 --set judge.f1_threshold={threshold}")
            source = json.loads(source_path.read_text(encoding="utf-8"))

            if source.get("status") != "final" or source.get("n_records") != 1000:
                raise RuntimeError(f"source is not final n=1000: {source_path}")
            records = copy.deepcopy(source["records"])
            if len(records) != 1000 or sorted(r["idx"] for r in records) != list(range(1000)):
                raise RuntimeError(f"source indices are incomplete: {source_path}")
            if any(len(r.get("rollouts", [])) != 16 for r in records):
                raise RuntimeError(f"source is not uniformly K=16: {source_path}")

            # F1EquivalenceJudge uses >=, matching CSR-F1 training
            judge = F1EquivalenceJudge(
                object(), {"judge": {"f1_threshold": threshold}}, chat_end_markers=()
            )
            for record in records:
                record["source_llm_semantic_ids"] = record.get("semantic_ids")
                record["source_llm_num_semantic_clusters"] = record.get(
                    "num_semantic_clusters"
                )
                record["source_llm_semantic_entropy"] = record.get("semantic_entropy")
                record["source_llm_confidence"] = record.get("confidence")
                block = semantic_confidence(
                    extracted_answers(record), record["question"], judge
                )
                record["semantic_ids"] = block["cluster_ids"]
                record["num_semantic_clusters"] = block["num_clusters"]
                record["semantic_entropy"] = block["semantic_entropy"]
                record["confidence"] = block["confidence"]
                # analysis/ reads clusters from the rollout level
                for rollout, cluster_id in zip(record["rollouts"], block["cluster_ids"]):
                    rollout["semantic_id"] = cluster_id

            summary = summarize(
                [r["confidence"] for r in records],
                [r["accuracy"] for r in records],
            )
            original_summary = source["summary"]
            summary.update(
                {
                    "num_rollouts": 16,
                    "checkpoint_step": original_summary.get("checkpoint_step", 2500),
                    "avg_semantic_clusters": sum(
                        r["num_semantic_clusters"] for r in records
                    )
                    / len(records),
                    "avg_prompt_tokens": original_summary.get("avg_prompt_tokens"),
                    "avg_total_tokens": original_summary.get("avg_total_tokens"),
                    "judge": {
                        "model_id": "gpt-4.1-nano + f1-equivalence",
                        "correctness": original_summary["judge"]["correctness"],
                        "semantic_equivalence": {
                            "backend": "f1",
                            "f1_threshold": threshold,
                            "comparison": ">=",
                            "total": int(judge.f1_same_stats["total"]),
                            "parsed_yes": int(judge.f1_same_stats["parsed_yes"]),
                            "parsed_no": int(judge.f1_same_stats["parsed_no"]),
                            "unparsed": 0,
                            "unparsed_rate": 0.0,
                        },
                        "source_llm_semantic_equivalence": original_summary["judge"][
                            "semantic_equivalence"
                        ],
                    },
                }
            )

            derived = copy.deepcopy(source)
            derived["experiment"] = target_name
            derived["status"] = "final"
            derived["n_records"] = len(records)
            derived["updated"] = datetime.now(timezone.utc).isoformat()
            derived["derivation"] = {
                "schema": "csr-f1train-f1eval-v1",
                "source_experiment": source_name,
                "source_results": str(source_path.relative_to(ROOT)),
                "checkpoint_training_equivalence": "f1",
                "evaluation_equivalence": "f1",
                "f1_threshold": threshold,
                "f1_comparison": ">=",
                "rollouts_regenerated": False,
                "correctness_rejudged": False,
                "correctness_judge": "gpt-4.1-nano",
            }
            derived["summary"] = summary
            derived["records"] = records
            write_json_atomic(target_path, derived)

            row = {
                "model": model,
                "dataset": dataset,
                "threshold": threshold,
                "source_experiment": source_name,
                "experiment": target_name,
                "results_json": str(target_path.relative_to(ROOT)),
                "checkpoint": source["checkpoint"],
                "n": len(records),
                "accuracy": summary["accuracy"],
                "ece": summary["ece"],
                "brier": summary["brier"],
                "auroc": summary["auroc"],
                "avg_confidence": summary["avg_confidence"],
                "avg_semantic_clusters": summary["avg_semantic_clusters"],
            }
            manifest["runs"].append(row)
            model_rows.append(row)

        manifest["macro_averages"][model] = {
            key: sum(row[key] for row in model_rows) / len(model_rows)
            for key in ("accuracy", "ece", "brier", "auroc", "avg_confidence")
        }

    write_json_atomic(RESULTS / "f1train_f1eval_manifest.json", manifest)
    print(json.dumps(manifest["macro_averages"], indent=2))


if __name__ == "__main__":
    main()

# Trainer callbacks: metric logging, the calibration ramp, periodic validation

from __future__ import annotations

import csv
import json
import numbers
import os
import shutil
import time

from transformers import TrainerCallback


def _open_csv(path: str):
    header = None
    if os.path.exists(path) and os.path.getsize(path):
        # a --resume-from-checkpoint run must not wipe the pre-resume rows
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None) or None
    mode = "a" if header else "w"
    return open(path, mode, newline="", encoding="utf-8"), header


class MetricCSVCallback(TrainerCallback):
    # TRL's scalar logs -> <run_dir>/train_metrics.csv

    def __init__(self, out_dir: str, extra: dict | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "train_metrics.csv")
        self.extra = extra or {}
        self._fh, self._header = _open_csv(self.csv_path)
        self._writer = None
        print(f"[metrics] {self.csv_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": int(state.global_step)}
        for key, value in logs.items():
            if isinstance(value, numbers.Number):
                row[key] = float(value)
        for key, getter in self.extra.items():
            row[key] = getter()

        if self._writer is None:
            header = self._header or ["step"] + sorted(k for k in row if k != "step")
            self._writer = csv.DictWriter(self._fh, fieldnames=header, extrasaction="ignore")
            if not self._header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
            print(f"[metrics] saved {self.csv_path}")
        except OSError:
            pass


class CalibrationRampCallback(TrainerCallback):
    # advances cali_w on cali_w_schedule (linear | sigmoid; equal endpoints = constant)

    def __init__(self, reward_computer):
        self.rewards = reward_computer
        self.print_every_steps = 25

    def on_step_end(self, args, state, control, **kwargs):
        progress = 0.0
        if state.max_steps and state.max_steps > 0:
            progress = float(state.global_step) / float(state.max_steps)
        cali_w = self.rewards.set_progress(progress)

        if self.print_every_steps and state.global_step % self.print_every_steps == 0:
            print(f"[cali_w] step={state.global_step} progress={progress:.3f} "
                  f"cali_w={cali_w:.4f}")


class JudgeCacheCallback(TrainerCallback):
    # judge cache hit rate, informational only

    def __init__(self, judge):
        self.judge = judge
        self.print_every_steps = 25

    def on_step_end(self, args, state, control, **kwargs):
        if not self.print_every_steps or state.global_step % self.print_every_steps:
            return
        stats = self.judge.stats_summary()
        print(f"[judge] step={state.global_step} cache={stats['cache_size']} "
              f"correctness={stats['correctness']['total']} "
              f"same={stats['semantic_equivalence']['total']} "
              f"unparsed={stats['correctness']['unparsed'] + stats['semantic_equivalence']['unparsed']}")


class PeriodicEvalCallback(TrainerCallback):
    # scores the policy on the validation split every N steps, skipping the last step.
    # Failures are logged, training continues.

    def __init__(self, cfg, dataset, tokenizer, judge, run_dir: str,
                 correctness_prompt_kind: str = "judge", every_steps: int = 250):
        self.cfg = cfg
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.judge = judge
        self.every_steps = int(every_steps)
        self.run_dir = run_dir
        self.kind = correctness_prompt_kind
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "validation_metrics.csv")
        self._fh, self._header = _open_csv(self.csv_path)
        self._writer = None
        print(f"[validation] {self.csv_path}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if self.every_steps <= 0 or step <= 0 or step % self.every_steps:
            return
        # the final checkpoint is scored by the formal evaluation, not here
        if state.max_steps and step >= int(state.max_steps):
            return
        if model is None:
            return

        try:
            self._evaluate(args, state, model, step)
        except Exception as exc:  # validation must never kill a training run
            print(f"[validation] step {step} skipped: {type(exc).__name__}: {exc}")

    def _evaluate(self, args, state, model, step: int) -> None:
        # always the live weights: on_step_end runs before the Trainer saves this
        # step, so an existing checkpoint-<step> would be left over from an earlier run
        adapter = os.path.join(self.run_dir, f"_eval_adapter_step_{step}")
        try:
            model.save_pretrained(adapter)
            lora_request = model.load_lora(adapter) if hasattr(model, "load_lora") else None
            started = time.time()
            confidences, accuracies, records = self._collect(model, lora_request)
        finally:
            # a full adapter copy per pass (1.1 GB at 32B); the step's checkpoint, if
            # save_steps lands on it, is written by the Trainer right after this
            shutil.rmtree(adapter, ignore_errors=True)
        saved = os.path.join(args.output_dir, f"checkpoint-{step}")
        on_save_step = bool(args.save_steps) and step % int(args.save_steps) == 0
        self._report(state, step, saved if on_save_step else "", started,
                     confidences, accuracies, records)

    def _n(self) -> int:
        return len(self.dataset)

    def _collect(self, model, lora_request):
        from .generation import rollouts_with_tokens
        from .semantic import semantic_confidence
        from .generation import render_generation_prompt
        from .parsing import strip_prompt_prefix

        train_cfg = self.cfg["method"]["train"]
        markers = self.cfg["model"].get("chat_end_markers", [])
        k = int(train_cfg.get("eval_num_rollouts", 16))

        confidences, accuracies = [], []
        records: list[dict] = []

        for i in range(self._n()):
            example = self.dataset[i]
            prompt = render_generation_prompt(example["prompt"], self.tokenizer)
            generated = rollouts_with_tokens(
                model, self.tokenizer, prompt,
                n=k,
                temperature=float(train_cfg.get("eval_temperature", 0.7)),
                top_p=float(train_cfg.get("eval_top_p", 0.95)),
                max_new_tokens=int(train_cfg.get("eval_max_new_tokens", 256)),
                seed=int(self.cfg["seed"]),
                lora_request=lora_request,
            )
            answers = [strip_prompt_prefix(t, markers) for t in generated["raw"]]
            clusters = semantic_confidence(answers, example["question"], self.judge)
            labels = self.judge.score_against_gold(
                answers, example["answer"], example["question"], self.kind)

            confidences.append(clusters["confidence"])
            accuracies.append(float(sum(labels) / max(1, len(labels))))
            records.append({
                "idx": i,
                "question": example["question"],
                "gold_answers": example["answer"] if isinstance(example["answer"], list)
                                else [example["answer"]],
                "rollouts": [
                    {"full_answer": raw, "extracted_answer": ans, "semantic_id": int(cid),
                     "correct": int(lab)}
                    for raw, ans, cid, lab in zip(generated["raw"], answers,
                                                  clusters["cluster_ids"], labels)
                ],
                "num_semantic_clusters": clusters["num_clusters"],
                "semantic_entropy": clusters["semantic_entropy"],
                "confidence": clusters["confidence"],
                "accuracy": accuracies[-1],
            })

        return confidences, accuracies, records

    def _report(self, state, step, adapter, started, confidences, accuracies, records) -> None:
        from .metrics import summarize

        summary = summarize(confidences, accuracies)
        row = {
            "step": step,
            "epoch": float(state.epoch or 0.0),
            "checkpoint": adapter,
            "elapsed_sec": round(time.time() - started, 1),
            **{key: ("" if value is None else value) for key, value in summary.items()},
        }
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=self._header or list(row),
                                          extrasaction="ignore")
            if not self._header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

        # per-question detail, after the CSV row is safe; guarded on its own
        try:
            detail_path = os.path.join(self.run_dir, f"validation_step_{step}.json")
            with open(detail_path, "w", encoding="utf-8") as fh:
                json.dump({"step": step, "checkpoint": adapter, "summary": summary,
                           "records": records}, fh, ensure_ascii=False, indent=2,
                          default=lambda v: v.item() if hasattr(v, "item") else str(v))
            # same flat rows as an eval run's results.csv
            from .results import flatten_records, write_rows_csv
            write_rows_csv(os.path.join(self.run_dir, f"validation_step_{step}.csv"),
                           flatten_records(records))
        except Exception as exc:
            print(f"[validation] step {step}: per-question detail not written: {exc}")

        print(f"[validation] step={step} n={summary['n']} acc={summary['accuracy']:.4f} "
              f"ece={summary['ece']:.4f} "
              f"auroc={summary['auroc'] if summary['auroc'] is not None else 'NA'} "
              f"({row['elapsed_sec'] / 60:.1f} min)")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
        except OSError:
            pass


class VerbalizedEvalCallback(PeriodicEvalCallback):
    # same schedule, for the methods that write their own confidence

    def _collect(self, model, lora_request):
        from .generation import count_tokens, generate
        from .generation import render_generation_prompt
        from .parsing import UNPARSED_CONFIDENCE, parse_verbalized_output

        train_cfg = self.cfg["method"]["train"]
        conf_cfg = self.cfg["method"]["confidence"]
        markers = self.cfg["model"].get("chat_end_markers", [])
        separators = self.cfg["model"].get("verbalized_separators", [","])

        confidences, accuracies = [], []
        records: list[dict] = []

        for i in range(self._n()):
            example = self.dataset[i]
            prompt = render_generation_prompt(example["prompt"], self.tokenizer)
            raw = generate(model, self.tokenizer, prompt, n=1,
                           temperature=float(train_cfg.get("eval_temperature", 0.0)),
                           top_p=float(train_cfg.get("eval_top_p", 1.0)),
                           max_new_tokens=int(train_cfg.get("eval_max_new_tokens", 256)),
                           seed=int(self.cfg["seed"]),
                           lora_request=lora_request)[0]

            answer, confidence, confidence_raw = parse_verbalized_output(
                raw, style=conf_cfg["style"], separators=separators,
                chat_end_markers=markers, scale=float(conf_cfg["scale"]),
            )
            # an answer with an unreadable confidence is still judged
            correct = 0
            if answer:
                correct = int(self.judge.score_against_gold(
                    [answer], example["answer"], example["question"], self.kind)[0])

            confidences.append(float(confidence) if confidence is not None
                               else UNPARSED_CONFIDENCE)
            accuracies.append(float(correct))
            records.append({
                "idx": i,
                "question": example["question"],
                "gold_answers": example["answer"] if isinstance(example["answer"], list)
                                else [example["answer"]],
                "full_answer": raw,
                "extracted_answer": answer,
                "semantic_id": None,
                "confidence": confidences[-1],
                "confidence_raw": confidence_raw,
                "format_ok": bool(answer is not None and confidence is not None),
                "correct": correct,
                "accuracy": float(correct),
                "prompt_tokens": count_tokens(self.tokenizer, prompt),
                "completion_tokens": count_tokens(self.tokenizer, raw),
            })

        return confidences, accuracies, records

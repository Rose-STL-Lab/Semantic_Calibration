# One run directory results/<exp_name>/ holding config.yaml, results.json (the
# artefact to read afterwards) and results.csv (the same data, flat).

from __future__ import annotations

import csv
import json
import os
import re
import time
from typing import Iterable

from .config import default_exp_name, dump_config

# per-question fields repeat on every rollout row, so a row is self-contained
_QUESTION_FIRST = ("idx", "question", "gold_answers")
_ROLLOUT_ORDER = ("rollout", "full_answer", "extracted_answer", "semantic_id",
                  "correct", "completion_tokens")


class RunDir:
    def __init__(self, results_root: str, exp_name: str):
        self.path = os.path.join(results_root, exp_name)
        self.exp_name = exp_name
        os.makedirs(self.path, exist_ok=True)
        self.config_path = os.path.join(self.path, "config.yaml")
        self.results_path = os.path.join(self.path, "results.json")
        self.csv_path = os.path.join(self.path, "results.csv")
        # for callers that print where they resume from
        self.records_path = self.results_path
        self._meta: dict = {}
        self._records: list[dict] = []

    def check_resume(self, checkpoint: str | None) -> None:
        # a run directory holds one checkpoint's records; resuming it with another
        # checkpoint would report the first one's results under the second's name
        try:
            with open(self.results_path, encoding="utf-8") as fh:
                stored = json.load(fh).get("checkpoint")
        except (OSError, ValueError):
            return
        if stored is not None and _same_checkpoint(stored, checkpoint) is False:
            raise SystemExit(
                f"{self.results_path} holds records for checkpoint {stored!r}, not "
                f"{checkpoint or 'base'!r}; pass --fresh or a different --exp-name")

    def save_config(self, cfg, checkpoint: str | None = None) -> None:
        # also remembers method/model/dataset/checkpoint for results.json
        dump_config(cfg, self.config_path)
        self._meta = _describe(self.exp_name, cfg, checkpoint)

    # -- incremental records --------------------------------------------
    def load_records(self, limit: int | None = None) -> list[dict]:
        # what a previous run of this experiment finished, to resume from
        if not os.path.exists(self.results_path) or os.path.getsize(self.results_path) == 0:
            return []
        try:
            with open(self.results_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            return []  # interrupted mid-write; start over
        rows = list(payload.get("records", []))
        if limit is not None:
            rows = rows[:limit]
        # a copy, since callers also append_records() the rows they get back
        self._records = list(rows)
        return rows

    def append_records(self, rows: Iterable[dict]) -> None:
        # append-only; rewrites results.json as a partial
        rows = list(rows)
        if not rows:
            return
        self._records.extend(rows)
        self._write_json(status="partial", summary=None)

    def rewrite_records(self, rows: Iterable[dict]) -> None:
        # the judge phase edits records already on disk; append_records cannot
        self._records = list(rows)
        self._write_json(status="partial", summary=None)

    def reset_records(self) -> None:
        self._records = []
        for path in (self.results_path, self.csv_path):
            if os.path.exists(path):
                os.remove(path)

    # -- final artefact ---------------------------------------------------
    def save_results(self, cfg, summary: dict, records: list[dict],
                     checkpoint: str | None = None) -> str:
        self._records = list(records)
        self._meta = _describe(self.exp_name, cfg, checkpoint)
        self._write_json(status="final", summary=summary)
        self._write_csv()
        return self.results_path

    # -- writers ----------------------------------------------------------
    def _write_json(self, status: str, summary: dict | None) -> None:
        payload = dict(self._meta)
        payload.update({
            "status": status,
            "n_records": len(self._records),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "records": self._records,
        })
        tmp = self.results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=_jsonable)
        os.replace(tmp, self.results_path)   # never leave a half-written file

    def _write_csv(self) -> None:
        write_rows_csv(self.csv_path, flatten_records(self._records))


def write_rows_csv(path: str, rows) -> None:
    # stable column order, shared with the validation callback so both match
    rows = list(rows)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    ordered = [k for k in _QUESTION_FIRST if k in keys]
    ordered += [k for k in _ROLLOUT_ORDER if k in keys and k not in ordered]
    ordered += [k for k in keys if k not in ordered]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csvable(row.get(k)) for k in ordered})
    os.replace(tmp, path)


def _same_checkpoint(stored: str, checkpoint: str | None) -> bool:
    def norm(p):
        return "base" if p in (None, "base") else os.path.realpath(p)
    return norm(stored) == norm(checkpoint)


def run_name(exp_name: str | None, cfg, checkpoint: str | None, n_checkpoints: int) -> str:
    # one results dir per checkpoint: an explicit --exp-name gets the checkpoint
    # name appended when several checkpoints are evaluated in one call
    if not exp_name:
        return default_exp_name(cfg, checkpoint)
    if n_checkpoints > 1 and checkpoint:
        return f"{exp_name}_{os.path.basename(checkpoint.rstrip('/'))}"
    return exp_name


def _describe(exp_name: str, cfg, checkpoint: str | None) -> dict:
    return {
        "experiment": exp_name,
        "method": cfg["method"]["name"],
        "model": cfg["model"]["name"],
        "model_hf_id": cfg["model"]["hf_id"],
        "dataset": cfg["dataset"]["name"],
        "checkpoint": checkpoint or "base",
    }


def flatten_records(records: Iterable[dict]) -> Iterable[dict]:
    # one row per rollout, or per question when there are none; question-level
    # fields (confidence, accuracy, ...) repeat on each rollout row
    for rec in records:
        rollouts = rec.get("rollouts")
        base = {k: v for k, v in rec.items() if k != "rollouts"}
        if not rollouts:
            yield base
            continue
        for j, ro in enumerate(rollouts):
            row = dict(base)
            row["rollout"] = j
            row.update(ro)
            yield row


def _csvable(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=_jsonable)
    if hasattr(value, "item"):
        try:
            return value.item()          # numpy scalar; an array of size > 1 raises
        except ValueError:
            pass
    return value


def _jsonable(value):
    if hasattr(value, "item"):
        try:
            return value.item()          # numpy scalar; an array of size > 1 raises
        except ValueError:
            pass
    return str(value)


def print_summary(summary: dict, title: str = "") -> None:
    def fmt(key, digits=4):
        value = summary.get(key)
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    header = f"[{title}] " if title else ""
    print(
        f"{header}n={summary.get('n')}  acc={fmt('accuracy')}  "
        f"conf={fmt('avg_confidence')}  ECE={fmt('ece')}  Brier={fmt('brier')}  "
        f"AUROC={fmt('auroc')}  Spearman={fmt('spearman')}  Kendall={fmt('kendall')}"
    )


def resolve_checkpoints(path: str | None) -> list[str | None]:
    # one checkpoint-N directory, a parent holding several, or None for a base run
    if not path:
        return [None]

    clean = path.rstrip("/")
    if not os.path.isdir(clean):
        raise FileNotFoundError(f"checkpoint path does not exist: {clean}")

    base = os.path.basename(clean)
    if base.startswith("checkpoint-") or base.startswith("model_finetuned"):
        return [clean]

    children = []
    for name in sorted(os.listdir(clean)):
        full = os.path.join(clean, name)
        if os.path.isdir(full) and (name.startswith("checkpoint-")
                                    or name.startswith("model_finetuned")):
            children.append(full)
    if not children:
        if not os.path.exists(os.path.join(clean, "adapter_config.json")):
            raise FileNotFoundError(f"{clean} holds no checkpoint-*/model_finetuned* "
                                    "directories and is not a LoRA adapter")
        return [clean]

    # RD's model_finetuned and model_finetuned_best copy a stepped checkpoint;
    # pass one of them directly to evaluate it
    stepped = [c for c in children if checkpoint_step(c) >= 0]
    return sorted(stepped or children, key=checkpoint_step)


def checkpoint_step(path: str | None) -> int:
    if not path:
        return 0
    match = re.search(r"(\d+)$", os.path.basename(str(path).rstrip("/")))
    return int(match.group(1)) if match else -1

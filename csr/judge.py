# LLM judge: semantic equivalence and correctness, both as one YES/NO call.
# Configured by configs/judge.yaml; credentials from the environment
# (OPENAI_API_KEY, plus OPENAI_API_VERSION / OPENAI_AZURE_ENDPOINT for Azure).

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from openai import AzureOpenAI, OpenAI

from .parsing import canonicalize_answer, sanitize_for_judge

# not line-anchored: a verdict sharing its line with prose would otherwise
# fall through to _YESNO_ANY_RE, which takes the first YES/NO in the whole text
_YESNO_LINE_RE = re.compile(r"(?i)\bCORRECT\s*:\s*(YES|NO)\b")
_YESNO_ANY_RE = re.compile(r"\b(YES|NO)\b")


def parse_yes_no(text: str) -> int | None:
    # 1 YES, 0 NO, None if unreadable
    if not text:
        return None
    match = _YESNO_LINE_RE.search(text)
    if match:
        return 1 if match.group(1).upper() == "YES" else 0
    match = _YESNO_ANY_RE.search(text.upper())
    if match:
        return 1 if match.group(1) == "YES" else 0
    return None


class LRUCache(OrderedDict):
    # keyed by short digests; memory stays flat over a long run

    def __init__(self, max_size: int = 200_000):
        super().__init__()
        self.max_size = int(max_size)

    def get_item(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def set_item(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.max_size:
            self.popitem(last=False)

def new_stats() -> dict:
    return {"total": 0, "parsed_yes": 0, "parsed_no": 0, "unparsed": 0}


class Judge:
    # batched YES/NO over an OpenAI-compatible chat-completions API

    def __init__(self, cfg, chat_end_markers: Sequence[str] = ()):
        judge_cfg = cfg["judge"] if "judge" in cfg else cfg
        self.model_id = judge_cfg["model_id"]
        self.temperature = float(judge_cfg.get("temperature", 0.0))
        self.max_tokens = int(judge_cfg.get("max_tokens", 16))
        self.max_retries = int(judge_cfg.get("max_retries", 2))
        self.sleep_sec = float(judge_cfg.get("sleep_sec", 1.5))
        self.max_workers = int(judge_cfg.get("max_workers", 6))
        self.batch_size = int(judge_cfg.get("batch_size", 16))
        # short per-request timeout: the SDK default is 600 s, so one stalled
        # connection parks a whole batch. Verdicts are unaffected at temperature 0.
        self.request_timeout_sec = float(judge_cfg.get("request_timeout_sec", 20.0))
        self.client_max_retries = int(judge_cfg.get("client_max_retries", 1))
        # optional self-hosted judge (e.g. local vLLM); unset -> the API client.
        # A reasoning judge spends tokens before its verdict: raise max_tokens.
        self.base_url = judge_cfg.get("base_url") or None
        self.base_api_key = judge_cfg.get("api_key") or "EMPTY"
        self.reasoning_effort = judge_cfg.get("reasoning_effort") or None
        self.system_prompt = judge_cfg["system_prompt"]

        # Rollout-vs-rollout equivalence everywhere, and rollout-vs-gold
        # correctness on datasets whose config asks for it (HotpotQA).
        self.prompt = judge_cfg["judge_prompt"]
        # OOD correctness only, chosen by gold count. Directional: A is the
        # answer judged, B the reference, never swapped.
        self.single_gold_prompt = judge_cfg["single_gold_prompt"]
        self.multi_gold_prompt = judge_cfg["multi_gold_prompt"]

        self.chat_end_markers = list(chat_end_markers or ())
        self.cache = LRUCache(int(judge_cfg.get("cache_max_size", 200_000)))
        self.same_stats = new_stats()
        self.correctness_stats = new_stats()
        self._client = None

    # -- API plumbing ---------------------------------------------------
    @staticmethod
    def _api_key() -> str:
        for name in ("OPENAI_API_KEY", "OPENAI_API_KEY_4O"):
            value = os.environ.get(name)
            if value:
                return value
        raise KeyError(
            "no judge API key: export OPENAI_API_KEY (or OPENAI_API_KEY_4O), or "
            "point judge.base_url at a local server, which needs no key.")

    @property
    def client(self):
        if self._client is None and self.base_url:
            self._client = OpenAI(base_url=self.base_url, api_key=self.base_api_key,
                                  timeout=self.request_timeout_sec,
                                  max_retries=self.client_max_retries)
        if self._client is None:
            key = self._api_key()
            common = {"api_key": key, "timeout": self.request_timeout_sec,
                      "max_retries": self.client_max_retries}
            try:
                self._client = AzureOpenAI(
                    api_version=os.environ["OPENAI_API_VERSION"],
                    azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                    **common,
                )
            except KeyError:
                if os.environ.get("OPENAI_BASE_URL"):
                    common["base_url"] = os.environ["OPENAI_BASE_URL"]
                self._client = OpenAI(**common)
        return self._client

    def _messages(self, template: str, q: str, a: str, b: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": template.format(
                q=sanitize_for_judge(q), a=sanitize_for_judge(a), b=sanitize_for_judge(b))},
        ]

    def _call_once(self, messages) -> str:
        extra = {"extra_body": {"reasoning_effort": self.reasoning_effort}} if self.reasoning_effort else {}
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1.0,
            n=1,
            **extra,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _cache_key(template: str, q: str, a: str, b: str) -> bytes:
        # The template stays in the key so a prompt edit invalidates the cache
        # rather than silently reusing verdicts from the old wording.
        payload = f"{template}\n<|Q|>\n{q}\n<|A|>\n{a}\n<|B|>\n{b}"
        return hashlib.blake2b(payload.encode("utf-8", "ignore"), digest_size=16).digest()

    # -- main entry point -----------------------------------------------
    def yes_no_batched(
        self,
        preds: Sequence[str],
        golds: Sequence[str],
        questions: Sequence[str],
        stats: dict | None = None,
        return_missing: bool = False,
        template: str | None = None,
    ):
        # template defaults to the symmetric prompt; any other is directional (A and B
        # keep their roles). -> (labels, n_unparsed), or (labels, n_unparsed, missing)
        # with return_missing=True; missing indexes the unparsable ones (label 0).
        assert len(preds) == len(golds) == len(questions)
        template = self.prompt if template is None else template
        stats = stats if stats is not None else self.same_stats

        n = len(preds)
        labels = [0] * n
        messages: list = [None] * n
        keys: list = [None] * n
        a_uses: list = [None] * n
        b_uses: list = [None] * n
        pending: list[int] = []

        for i, (q, a, b) in enumerate(zip(questions, preds, golds)):
            q = "" if q is None else str(q)
            a_use = canonicalize_answer(a, self.chat_end_markers)
            b_use = canonicalize_answer(b, self.chat_end_markers)
            # A and B keep the caller's roles; sorting them canonically would halve
            # the calls but put gold in the prediction slot in half of correctness.
            a_uses[i], b_uses[i] = a_use, b_use

            # exact-match shortcut: identical canonical forms need no call
            if a_use != "" and a_use == b_use:
                labels[i] = 1
                continue

            key = self._cache_key(template, q, a_use, b_use)
            keys[i] = key
            cached = self.cache.get_item(key)
            if cached is not None:
                labels[i] = int(cached)
                continue

            messages[i] = self._messages(template, q, a_use, b_use)
            pending.append(i)

        if not pending:
            return (labels, 0, []) if return_missing else (labels, 0)

        def judge_one(idx: int):
            text = ""
            for attempt in range(self.max_retries + 1):
                try:
                    text = self._call_once(messages[idx])
                    break
                except Exception:
                    if attempt >= self.max_retries:
                        text = ""
                        break
                    time.sleep(self.sleep_sec)
            return idx, parse_yes_no(text)

        # one pool over all pending verdicts; a pool per chunk idles on each tail
        in_flight_cap = max(int(self.batch_size), 4 * self.max_workers)
        for _ in range(max(1, self.max_retries)):
            if not pending:
                break
            still_pending: list[int] = []
            workers = min(self.max_workers, len(pending)) or 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, len(pending), in_flight_cap):
                    window = pending[start:start + in_flight_cap]
                    futures = [pool.submit(judge_one, idx) for idx in window]
                    for future in as_completed(futures):
                        idx, verdict = future.result()
                        stats["total"] += 1
                        if verdict is None:
                            still_pending.append(idx)
                            continue
                        labels[idx] = int(verdict)
                        stats["parsed_yes" if verdict == 1 else "parsed_no"] += 1
                        self.cache.set_item(keys[idx], int(verdict))
            pending = still_pending

        unparsed = len(pending)
        if unparsed:
            stats["unparsed"] += unparsed
            # Not cached: one transient failure would become a permanent NO. Callers
            # get label 0 plus the index in `missing` and can drop it instead.
            for idx in pending:
                labels[idx] = 0

        return (labels, unparsed, list(pending)) if return_missing else (labels, unparsed)

    # -- convenience wrappers -------------------------------------------
    def symmetric_batched(self, a_list, b_list, questions):
        # one symmetric verdict per unordered pair; returns as yes_no_batched
        return self.yes_no_batched(a_list, b_list, questions,
                                   stats=self.same_stats, return_missing=True)

    def correctness_batched(self, preds, golds, questions,
                            template: str | None = None) -> tuple[list[int], int]:
        return self.yes_no_batched(preds, golds, questions,
                                   stats=self.correctness_stats, template=template)

    def _gold_template(self, gold, kind: str):
        # "judge": the equivalence prompt (template None). "gold": single/multi by
        # gold count. Shared so the batch path cannot drift.
        if kind == "judge":
            return None
        if isinstance(gold, (list, tuple)) and len(gold) > 1:
            return self.multi_gold_prompt
        return self.single_gold_prompt

    def score_against_gold(self, preds: Sequence[str], gold, question: str,
                           kind: str = "judge") -> list[int]:
        # every prediction for one question against its gold answer(s)
        gold_text = format_gold(gold, self.chat_end_markers)
        template = self._gold_template(gold, kind)
        labels, _ = self.correctness_batched(
            preds=list(preds),
            golds=[gold_text] * len(preds),
            questions=[question or ""] * len(preds),
            template=template,
        )
        return labels

    def score_against_gold_batch(self, preds_list: Sequence[Sequence[str]], golds,
                                 questions: Sequence[str], kind: str = "judge") -> list[list[int]]:
        # one correctness call per template, chosen as in score_against_gold
        groups: dict = {}
        for pos, (preds, gold, question) in enumerate(zip(preds_list, golds, questions)):
            gold_text = format_gold(gold, self.chat_end_markers)
            template = self._gold_template(gold, kind)
            group = groups.setdefault(template, ([], [], [], []))
            for k, pred in enumerate(preds):
                group[0].append(pred)
                group[1].append(gold_text)
                group[2].append(question or "")
                group[3].append((pos, k))

        out = [[0] * len(preds) for preds in preds_list]
        for template, (preds, gold_texts, qs, where) in groups.items():
            labels, _ = self.correctness_batched(preds=preds, golds=gold_texts,
                                                 questions=qs, template=template)
            for (pos, k), label in zip(where, labels):
                out[pos][k] = label
        return out

    def assert_parsed(self, max_unparsed_rate: float = 0.02) -> None:
        """Refuse to let a run be written when the judge was not actually answering.

        A reasoning judge whose verdict does not fit in max_tokens returns an empty
        message, every call is scored unparsed, and the run still completes -- with
        every label defaulted. That has to stop the run, not reach results.json.
        """
        summary = self.stats_summary()
        broken = []
        for name, stats in summary.items():
            if not isinstance(stats, dict) or "unparsed_rate" not in stats:
                continue
            total = int(stats.get("total") or 0)
            rate = float(stats.get("unparsed_rate") or 0.0)
            if total >= 20 and rate > max_unparsed_rate:
                broken.append(
                    f"{name}: {stats.get('unparsed')}/{total} unparsed ({rate:.1%}), "
                    f"parsed_yes={stats.get('parsed_yes')} parsed_no={stats.get('parsed_no')}")
        if broken:
            raise SystemExit(
                "[judge] refusing to write results: the judge did not return parseable "
                "verdicts.\n  " + "\n  ".join(broken) +
                f"\n  model_id={summary.get('model_id')}\n"
                "  A reasoning judge needs room for its verdict -- retry with "
                "--set judge.reasoning_effort=low --set judge.max_tokens=256 "
                "(gpt-oss-20b), or raise max_tokens.")

    def stats_summary(self) -> dict:
        def block(stats):
            total = int(stats["total"])
            unparsed = int(stats["unparsed"])
            return {
                "total": total,
                "parsed_yes": int(stats["parsed_yes"]),
                "parsed_no": int(stats["parsed_no"]),
                "unparsed": unparsed,
                "unparsed_rate": float(unparsed / max(1, total)),
            }

        return {
            "model_id": self.model_id,
            "correctness": block(self.correctness_stats),
            "semantic_equivalence": block(self.same_stats),
            "cache_size": len(self.cache),
        }

    def reset_stats(self) -> None:
        self.same_stats = new_stats()
        self.correctness_stats = new_stats()


def make_judge(cfg, chat_end_markers: Sequence[str] = ()):
    # Correctness is always the LLM judge, as every reported number is. The one switch
    # replaces rollout-vs-rollout equivalence with token F1 -- Table 8's CSR-F1:
    #     --set judge.equivalence=f1 --set judge.f1_threshold=0.5
    judge_cfg = cfg["judge"] if "judge" in cfg else cfg
    if "backend" in judge_cfg:
        # --set creates unknown keys silently, so a stale judge.backend would be ignored
        raise ValueError("judge.backend was removed; correctness is always the LLM judge. "
                         "Use --set judge.equivalence=f1 for the CSR-F1 configuration.")
    judge = Judge(cfg, chat_end_markers=chat_end_markers)
    equivalence = str(judge_cfg.get("equivalence", "llm")).lower()
    if equivalence in ("f1", "token_f1"):
        from .f1_judge import F1EquivalenceJudge
        return F1EquivalenceJudge(judge, cfg, chat_end_markers=chat_end_markers)
    if equivalence not in ("llm", "api", ""):
        raise ValueError(f"unknown judge.equivalence {equivalence!r}; expected 'llm' or 'f1'")
    return judge


def format_gold(gold, chat_end_markers: Sequence[str] = ()) -> str:
    from .parsing import normalize_for_equiv

    if isinstance(gold, (list, tuple)):
        parts = [normalize_for_equiv(g, chat_end_markers) for g in gold]
        return "; ".join(p for p in parts if p)
    return normalize_for_equiv(gold, chat_end_markers)

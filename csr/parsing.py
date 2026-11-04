# Answer extraction and canonicalization for everything a model emits; the only
# model-dependent piece is the chat end-of-turn markers from
# configs/models/<model>.yaml.

from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, Sequence

_CANON_PREFIX_RE = re.compile(
    r"^\s*(answer\s*:\s*|final\s*answer\s*:\s*|the\s*answer\s*is\s*|answer\s*is\s*)",
    flags=re.I,
)
_CANON_TAIL_RE = re.compile(r"\s*(explanation\s*:.*)$", flags=re.I | re.S)

# against prompt/format injection from model output and dataset text
_CHAT_TEMPLATE_TOKEN_RE = re.compile(r"<\|[^|>\s]{1,80}\|>")
_JUDGE_OUTPUT_MIMIC_RE = re.compile(r"\bcorrect\s*:\s*(?:yes|no)\b", flags=re.I)


def strip_chat_markers(text: str, markers: Sequence[str] = ()) -> str:
    out = str(text or "").strip()
    if not markers:
        return out
    pattern = "(?:%s)" % "|".join(re.escape(m) for m in markers if m)
    changed = True
    while changed:
        new = re.sub(rf"(?:{pattern}\s*)+$", "", out).strip()
        changed = new != out
        out = new
    return out


def _completion_to_str(completion) -> str:
    # TRL passes a plain string or a chat-style list
    if isinstance(completion, list) and completion and isinstance(completion[0], dict) \
            and "content" in completion[0]:
        return completion[0]["content"]
    if isinstance(completion, list) and completion and isinstance(completion[0], str):
        return completion[0]
    if isinstance(completion, str):
        return completion
    return str(completion)


def clean_generation(completion, chat_end_markers: Sequence[str] = ()) -> str:
    # leaves "Answer:" intact; the verbalized parsers split on it themselves
    text = strip_chat_markers(_completion_to_str(completion), chat_end_markers)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^\s*(cutting knowledge date|today date)\s*:", stripped, flags=re.I):
            continue
        if re.match(r"^\s*(system|user|assistant)\s*$", stripped, flags=re.I):
            continue
        stripped = re.sub(r"\.(system|user|assistant)\s*$", "", stripped, flags=re.I).strip()
        cleaned_lines.append(stripped)

    return "\n".join(x for x in cleaned_lines if x).strip()


def strip_prompt_prefix(completion, chat_end_markers: Sequence[str] = ()) -> str:
    # everything after the last "Answer:"; "Final answer:" (context prompts) wins if present
    text = clean_generation(completion, chat_end_markers)
    if "Final answer:" in text:
        return text.split("Final answer:")[-1].strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text


def normalize_for_equiv(text, chat_end_markers: Sequence[str] = (), max_chars: int = 2000) -> str:
    t = strip_prompt_prefix(text, chat_end_markers)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def canonicalize_answer(text, chat_end_markers: Sequence[str] = ()) -> str:
    # the judge's cache key and exact-match shortcut
    t = normalize_for_equiv(text, chat_end_markers)
    t = _CANON_TAIL_RE.sub("", t).strip()
    t = _CANON_PREFIX_RE.sub("", t).strip()
    t = re.sub(r"[ \t]*\.\s*$", "", t).strip()
    return t


def sanitize_for_judge(text, max_chars: int = 1000) -> str:
    t = "" if text is None else str(text)
    t = _CHAT_TEMPLATE_TOKEN_RE.sub(" ", t)
    t = _JUDGE_OUTPUT_MIMIC_RE.sub("[redacted]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def clean_unicode_text(text) -> str:
    # NFKC + zero-width / exotic-space removal (MSMARCO and NQ-Open need it)
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = re.sub("[\u200b\u200c\u200d\ufeff]", "", t)                       # zero-width
    t = re.sub("[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", t)       # exotic spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_answer(s) -> str:
    # SQuAD-style, for exact match against aliases
    text = str(s or "").lower().replace("_", " ")
    exclude = set(string.punctuation + "‘’´`")
    text = "".join(ch if ch not in exclude else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def extract_user_question(prompt) -> str:
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
    return str(prompt)


# verbalized-confidence parsing (RD and RLCR)
_RLCR_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.I | re.S)
_RLCR_CONFIDENCE_RE = re.compile(r"<confidence>\s*(.*?)\s*</confidence>", flags=re.I | re.S)


def _separator_class(separators: Sequence[str]) -> str:
    chars = "".join(re.escape(s) for s in separators if s and s != "\n")
    newline = r"|\r?\n" if any(s == "\n" for s in separators) else ""
    if not chars and not newline:
        return r","
    return rf"[{chars}]{newline}" if chars else newline.lstrip("|")


def parse_rd_output(text, separators: Sequence[str] = (",",),
                    chat_end_markers: Sequence[str] = (), scale: float = 10.0):
    # Answer: <a>, Confidence: <0-10>
    # -> (answer | None, confidence in [0,1] | None, raw_conf)
    raw = strip_chat_markers(_completion_to_str(text), chat_end_markers)
    # a reasoning draft's own "confidence:" must not outrank the final line; an
    # unclosed <think> is a thought the budget cut off, so nothing after it counts
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        raw = raw.split("<think>", 1)[0]
    sep = _separator_class(separators)

    # clean_generation, not strip_prompt_prefix: the "Answer:" marker must survive
    # for the Answer/Confidence pair to be found
    body = clean_generation(raw, chat_end_markers)
    match = re.search(
        rf"Answer:\s*(.*?)(?:\s*(?:{sep})?\s*Confidence:\s*[-+]?\d+(?:\.\d+)?|$)",
        body, flags=re.I | re.S,
    )
    answer_text = match.group(1).strip() if match else body

    answer = canonicalize_answer(answer_text, chat_end_markers) or None

    conf_match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)", body, flags=re.I)
    if not conf_match:
        return answer, None, None
    raw_conf = conf_match.group(1)
    try:
        value = float(raw_conf)
    except ValueError:
        return answer, None, raw_conf
    if not (0.0 <= value <= scale):
        return answer, None, raw_conf
    # a decimal <= 1 is read as a 0-1 probability, so "1.0" means certainty, not 0.1
    if "." in raw_conf and value <= 1.0:
        return answer, float(min(max(value, 0.0), 1.0)), raw_conf
    return answer, float(min(max(value / scale, 0.0), 1.0)), raw_conf


def parse_rlcr_output(text, chat_end_markers: Sequence[str] = (), scale: float = 1.0):
    # <answer>...</answer> ... <confidence>...</confidence>
    raw = strip_chat_markers(_completion_to_str(text), chat_end_markers)

    answers = _RLCR_ANSWER_RE.findall(raw)
    answer = canonicalize_answer(answers[-1], chat_end_markers) if answers else ""
    answer = answer or None

    confidences = _RLCR_CONFIDENCE_RE.findall(raw)
    if not confidences:
        return answer, None, None
    raw_conf = confidences[-1].strip()
    try:
        value = float(raw_conf)
    except ValueError:
        return answer, None, raw_conf
    if not (0.0 <= value <= scale):
        return answer, None, raw_conf
    return answer, float(min(max(value / scale, 0.0), 1.0)), raw_conf


# unparsable confidence counts as 0, as in RD's evaluation (RLCR's re-prompting is not reproduced)
UNPARSED_CONFIDENCE = 0.0


def parse_verbalized_output(text, style: str, separators: Sequence[str] = (",",),
                            chat_end_markers: Sequence[str] = (), scale: float = 10.0):
    if style == "rd":
        return parse_rd_output(text, separators, chat_end_markers, scale)
    if style == "rlcr":
        return parse_rlcr_output(text, chat_end_markers, scale)
    raise ValueError(f"unknown verbalized confidence style: {style!r}")

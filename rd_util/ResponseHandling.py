from dataclasses import dataclass
import dataclasses
from typing import List, Optional
import re
import json
from dacite import from_dict as dict_to_class
import numpy as np
import math
import torch
import torch.nn.functional as F

from rd_util.util import remove_padding


@dataclass
class QAResult:
    question: str
    prediction: Optional[str]
    confidence: Optional[float]
    gt_candidates: List[str]
    
    is_multiple_choice: bool
    is_wrong_format: bool

def QAResult_to_dict(result: QAResult) -> dict:
    return dataclasses.asdict(result)

def dict_to_QAResult(result: dict) -> QAResult:
    return dict_to_class(QAResult, result)

def parse_answer_confidence(output, is_multiple_choice, strict=False):
    """
    Looks for answer and confidence values with regex. \n
    pattern = r"Answer:\s*(?P<answer>.*?)\s*Confidence:\s*(?P<confidence>\d+)"

    Returns:
        Tuple[str, int]: answer, confidence
        is None, None if pattern not found
    """
    text = str(output or "").strip()
    text = re.sub(r"(?:</s>\s*)+$", "", text).strip()
    text = re.sub(r"(?:<\|im_end\|>\s*)+$", "", text).strip()

    # Main path: support both comma and newline between Answer / Confidence.
    # Confidence stays on the 0-10 scale the prompt asks for, but a decimal <= 1 is
    # accepted and mapped onto it, because that is the model answering on a 0-1
    # probability scale instead. RD training induces exactly that: base Mistral emits
    # 99/100 integers (almost always "10"), while the step-2500 checkpoint collapses
    # onto a 0/1 verdict and writes 22/100 of them as floats. The integer-only regex
    # this replaces read "0.9" as 0, and reward_function hands a conf-0 report the top
    # reward on every question the model gets wrong -- at Mistral's 29% accuracy that
    # misread paid for the collapse on 71% of the data. Mapping into 0-10 here leaves
    # reward_function, the ECE/AUROC helpers and the wrong-format penalty untouched.
    conf_re = r"-?\d+(?:\.\d+)?"
    main_pattern = re.compile(
        rf"Answer\s*:\s*(?P<answer>.*?)\s*(?:,|\n|\r\n)\s*Confidence\s*:\s*(?P<confidence>{conf_re})\b",
        flags=re.I | re.S,
    )
    match = main_pattern.search(text)

    def to_scale10(token: str) -> float:
        value = float(token)
        if "." in token and 0.0 <= value <= 1.0:
            return value * 10.0
        return value

    # Fallback path: if formatting is noisy, extract fields independently.
    # strict skips it, as upstream's single regex does: the fallback reads
    # "Answer: Confidence: 0" as the answer "Confidence: 0", and an abstention
    # with confidence 0 is the highest-paid wrong answer.
    if not match:
        if strict:
            return None, None
        conf_match = re.search(rf"Confidence\s*:\s*(?P<confidence>{conf_re})\b", text, flags=re.I)
        ans_match = re.search(r"Answer\s*:\s*(?P<answer>.*)", text, flags=re.I)
        if not (conf_match and ans_match):
            return None, None
        answer = ans_match.group("answer").strip()
        confidence = to_scale10(conf_match.group("confidence"))
    else:
        answer = match.group("answer").strip()
        token = match.group("confidence")
        if strict and "." in token:
            # the prompt asks for an integer 0-10: "5.0" is the integer 5, but "5.1"
            # or "0.9" is a format failure (-30) like any other departure from the
            # requested format, not an integer prefix (upstream reads "0.9" as 0) and
            # not a value on another scale
            value = float(token)
            if not value.is_integer():
                return None, None
            confidence = value
        else:
            confidence = to_scale10(token)

    # Trim potential trailing markers in answer.
    answer = re.sub(r"(?:</s>|<\|im_end\|>)\s*$", "", answer).strip()
    if not answer:
        return None, None
    # an answer that itself holds a "Confidence:" field is an empty answer followed by
    # a second attempt, not an answer
    if strict and re.search(r"Confidence\s*:", answer, flags=re.I):
        return None, None

    if is_multiple_choice and ":" in answer:
        splitted_answer = answer.split(":")
        if not (len(splitted_answer[0]) == 1):
            return None, None
        answer = splitted_answer[0]

    return answer, confidence

def response_to_QAResult(question: str, response: str, gt_candidates: List[str], is_multiple_choice: bool = False,
                         strict: bool = False) -> QAResult:
    prediction, confidence = parse_answer_confidence(response, is_multiple_choice=is_multiple_choice, strict=strict)
    is_wrong_format = prediction == None

    result = QAResult(question, prediction, confidence, gt_candidates, is_multiple_choice, is_wrong_format)
    return result

def save_QAResults(results: List[QAResult], out_dir: str):
    results_dict = [QAResult_to_dict(res) for res in results]

    with open(out_dir, 'w') as fout:
        json.dump(results_dict , fout, indent=4)

def load_QAResults(read_dir: str) -> List[QAResult]:
    with open(read_dir, "r") as f:
        results_dicts = json.load(f) 

    results = [dict_to_QAResult(r) for r in results_dicts]
    return results

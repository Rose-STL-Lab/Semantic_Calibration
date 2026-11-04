#!/usr/bin/env python
"""RD (Rewarding Doubt) baseline: PPO on a verbalized confidence head.

Port of https://github.com/pasta99/RewardingDoubt (arXiv:2503.02623); rd_util/ stays
close to upstream. Needs the csr-rd env (trl 0.8.6 PPO API, from requirements-rd.txt);
RD *evaluation* instead runs in the main env, see evaluation/eval_verbalized.py.

    python train/train_RD.py --model qwen --exp-name RD_qwen_hotpot
"""

import argparse
import sys
import csv
import torch
import os
import numpy as np
import re
import random


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csr.config import (add_common_args, checkpoint_dir, dump_config, load_config,
                        seed_everything)

from trl import PPOConfig
import trl.trainer.ppo_trainer as _trl_ppo
# unsloth >= 2025.3 rebinds PPOTrainer/PPOConfig in trl.trainer.ppo_trainer with variants
# that break on trl 0.8.6: subclass before unsloth loads, then restore the module's names.
_trl_ppo_names = dict(vars(_trl_ppo))
from rd_util.ppo_trainer_no_cache import PPOTrainerNoCache
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm.auto import tqdm
from rd_util.DataHelper import DataCollatorForTokenizedQueries
from unsloth import FastLanguageModel
for _name, _obj in _trl_ppo_names.items():
    if vars(_trl_ppo).get(_name) is not _obj:
        setattr(_trl_ppo, _name, _obj)
from typing import Tuple

from rd_util.DataHelper import load_prepared_dataset
from rd_util.ResponseHandling import response_to_QAResult
from rd_util.RLHelper import QAResult_to_reward
from rd_util.EvaluationMetrics import Metric, QAResults_to_accuracy, QAResults_to_ECE, QAResults_to_auroc_score, is_answer_correct
from rd_util.util import remove_padding
from rd_util.ModelLoader import load_lora_model_tokenizer, load_model_tokenizer


# RD's dataset registry (rd_util/DataHelper.py) uses its own names, and only
# covers a subset of this repo's datasets. RD is trained on HotpotQA.
RD_DATASET_NAMES = {
    "hotpot": "hotpotqa",
    "hotpot_context": "hotpotqa_context",
    "trivia": "triviaqa",
}


def rd_dataset_name(name: str) -> str:
    if name not in RD_DATASET_NAMES:
        raise SystemExit(
            f"RD training has no dataset descriptor for {name!r}. "
            f"Available: {sorted(RD_DATASET_NAMES)}. Add one to "
            f"rd_util/DataHelper.py to train RD on another corpus."
        )
    return RD_DATASET_NAMES[name]


class StopAtConfidence(StoppingCriteria):
    """Stop stage 1 once the generation has written the word "Confidence".

    Upstream stops on the token " Confidence", which Mistral lacks; the word's
    final token pre-filters and a short decoded tail confirms the whole word.
    """

    def __init__(self, tokenizer, prompt_len: int, word: str = "Confidence", tail: int = 16):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.word = word
        self.tail = tail
        ends = set()
        for lead in (", ", "\n", " ", ""):
            ids = tokenizer(lead + word, add_special_tokens=False)["input_ids"]
            if ids:
                ends.add(int(ids[-1]))
        self.end_ids = torch.tensor(sorted(ends), dtype=torch.long)

    def __call__(self, input_ids, scores, **kwargs):
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        hits = torch.isin(input_ids[:, -1], self.end_ids.to(input_ids.device))
        rows = hits.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return done
        start = max(self.prompt_len, int(input_ids.shape[1]) - self.tail)
        tails = self.tokenizer.batch_decode(input_ids[rows, start:], skip_special_tokens=True)
        for row, text in zip(rows.tolist(), tails):
            done[row] = self.word in text
        return done

class StopAfterConfidenceValue(StoppingCriteria):
    """End the PPO continuation once the confidence value is complete.

    The value must be followed by a non-digit, so "10" is not cut to "1". Needed
    for models (Mistral) that keep generating after the value.
    """

    # a dot ends the value only if no digit follows ("7." vs "5.0")
    PATTERN = re.compile(r"Confidence\s*:\s*[*_`\s]*-?\d+(?:\.\d+)?(?:[^\d.]|\.[^\d])")

    def __init__(self, tokenizer, tail: int = 48):
        self.tokenizer = tokenizer
        self.tail = tail

    def __call__(self, input_ids, scores, **kwargs):
        window = input_ids[:, -self.tail:]
        tails = self.tokenizer.batch_decode(window, skip_special_tokens=True)
        # only the last Confidence field is the one being continued
        return torch.tensor([bool(self.PATTERN.search(tail[tail.rfind("Confidence"):])) for tail in tails],
                            dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def sample_without_cache(model, tokenizer, ids, *, max_new_tokens: int, temperature: float,
                         top_p: float, stop_ids, value_stop=None):
    """Sample without KV cache, one full forward per token.

    Works around unsloth's fast generate corrupting some long-prompt continuations.
    Same semantics as generate(do_sample=True, top_k=0) with nucleus top_p.
    """
    seq = ids[None] if ids.dim() == 1 else ids
    start = seq.shape[1]
    for _ in range(int(max_new_tokens)):
        logits = model(input_ids=seq).logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                keep = torch.cumsum(sp, 0) - sp < top_p
                probs = torch.zeros_like(probs)
                probs[si[keep]] = sp[keep]
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1))
        seq = torch.cat((seq, seq.new_tensor([[nxt]])), dim=1)
        if nxt in stop_ids:
            break
        if value_stop is not None and bool(value_stop(seq, None)[0]):
            break
    return seq[0, start:]


# EOS + the model's end-of-turn markers (upstream hardcoded Llama's <|eot_id|>);
# markers absent from the vocabulary are skipped
def terminator_ids(tokenizer, chat_end_markers):
    ids = [tokenizer.eos_token_id]
    for marker in chat_end_markers or []:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None or not isinstance(token_id, int) or token_id < 0:
            continue
        if token_id == getattr(tokenizer, "unk_token_id", None):
            continue
        if tokenizer.convert_ids_to_tokens(token_id) != marker:
            continue
        ids.append(token_id)
    return _clean_token_ids(sorted(set(i for i in ids if i is not None)))


# F1@0.5 on the datasets with free-form answers; upstream RD scores its PPO
# reward by exact match
def get_metric_and_threshold(dataset_name: str):
    if dataset_name in {"triviaqa", "hotpotqa", "hotpotqa_context"}:
        return Metric.F1, 0.5
    return Metric.EXACT, 0.5


def safe_metric(metric_fn, results, metric: Metric, threshold: float = 0.5, max_confidence: int = 10):
    try:
        return metric_fn(results, metric, threshold, max_confidence)
    except ValueError:
        return float("nan")


def evaluate_subset(model, dataloader, tokenizer, generation_kwargs, device, split_name: str, metric: Metric, threshold: float,
                    strict: bool = False):
    rewards_epoch = []
    qa_results = []

    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Eval {split_name}", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_candidates = batch["gt_candidates"]
        questions = batch["question"]
        is_multiple_choice = batch["is_multiple_choice"]

        prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        responses = [prediction[i][len(input_ids[i]):] for i in range(len(prediction))]
        responses_decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)

        results = [response_to_QAResult(question, response, gt, is_mc, strict=strict) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
        rewards = [QAResult_to_reward(r, metric=metric, threshold=threshold) for r in results]
        rewards_epoch += rewards
        qa_results += results

    metrics = {
        "RewardMean": float(np.mean(rewards_epoch)),
        "RewardStd": float(np.std(rewards_epoch)),
        "Accuracy": float(QAResults_to_accuracy(qa_results, metric, threshold, 10)),
        "ECE": float(safe_metric(QAResults_to_ECE, qa_results, metric, threshold, 10)),
        "AUROC": float(safe_metric(QAResults_to_auroc_score, qa_results, metric, threshold, 10)),
        "NumResults": len(qa_results),
    }
    return metrics


def append_eval_metrics(csv_path: str, rows):
    file_exists = os.path.exists(csv_path)
    fieldnames = ["Step", "Split", "NumResults", "RewardMean", "RewardStd", "Accuracy", "ECE", "AUROC"]
    with open(csv_path, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def shuffle_and_limit_dataset(dataset, max_samples: int, seed: int = 42):
    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _clean_token_ids(token_ids):
    if isinstance(token_ids, (list, tuple)):
        cleaned = []
        for token_id in token_ids:
            if isinstance(token_id, int) and token_id >= 0:
                cleaned.append(token_id)
        return cleaned
    return token_ids if isinstance(token_ids, int) and token_ids >= 0 else None


def _capture_rng_state():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.random.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "torch_cuda_random_all" in state:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_all"])
        except Exception:
            pass


def save_training_state(ppo_trainer, out_dir: str, global_step: int, epoch_idx: int, best_reward: float, best_reward_step: int, ckpt_dir: str | None = None):
    state = {
        "global_step": int(global_step),
        "epoch_idx": int(epoch_idx),
        "best_reward": float(best_reward),
        "best_reward_step": int(best_reward_step),
        "rng_state": _capture_rng_state(),
        # a resume with a different batch_size or max_train_samples is refused
        "batches_per_epoch": int(len(ppo_trainer.dataloader)),
        # checked on resume; kl_coef would otherwise restart at init_kl_coef
        "train_flags": dict(getattr(ppo_trainer, "rd_train_flags", {}) or {}),
        "kl_coef": float(ppo_trainer.kl_ctl.value),
    }
    if getattr(ppo_trainer, "optimizer", None) is not None:
        state["optimizer"] = ppo_trainer.optimizer.state_dict()
    if getattr(ppo_trainer, "lr_scheduler", None) is not None:
        state["lr_scheduler"] = ppo_trainer.lr_scheduler.state_dict()

    latest_path = os.path.join(out_dir, "training_state_latest.pt")
    step_path = os.path.join(out_dir, f"training_state_step_{int(global_step)}.pt")
    torch.save(state, latest_path)
    torch.save(state, step_path)
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_state_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save(state, ckpt_state_path)
    else:
        ckpt_state_path = None
    print(f"[Resume] saved trainer state -> {latest_path}")
    if ckpt_state_path is not None:
        print(f"[Resume] saved trainer state -> {ckpt_state_path}")
    return latest_path, step_path, ckpt_state_path


# the adapter and the value head have to be in place before the model is built, so
# the state file is located before the load and read again after
def resolve_resume(out_dir: str, resume_state_path: str | None = None, resume_auto: bool = False):
    if resume_state_path:
        cand = resume_state_path
    elif resume_auto:
        cand = os.path.join(out_dir, "training_state_latest.pt")
    else:
        return None, None
    if not os.path.exists(cand):
        print(f"[Resume] no state file at {cand}, starting fresh", flush=True)
        return None, None

    state = torch.load(cand, map_location="cpu", weights_only=False)
    if "batches_per_epoch" not in state:
        raise SystemExit(
            f"{cand} has no batches_per_epoch and cannot be resumed; start the run over."
        )
    step = int(state["global_step"])
    # a state saved inside a checkpoint directory names its own adapter
    if os.path.basename(cand) == "training_state.pt":
        adapter_dir = os.path.dirname(cand)
    else:
        adapter_dir = os.path.join(out_dir, f"model_finetuned_step_{step}")
    if not os.path.isdir(adapter_dir):
        raise SystemExit(
            f"{cand} is at step {step} but {adapter_dir} does not exist, so that "
            "step's adapter cannot be reloaded"
        )
    return cand, adapter_dir


def maybe_load_training_state(ppo_trainer, state_path: str | None):
    if not state_path:
        return 0, 0, -100.0, -1

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    batches = len(ppo_trainer.dataloader)
    saved_batches = int(state["batches_per_epoch"])
    if saved_batches != batches:
        raise SystemExit(
            f"{state_path} was written with {saved_batches} batches per epoch but this "
            f"run has {batches}; batch_size and max_train_samples must match to resume"
        )
    saved_flags = state.get("train_flags")
    flags = dict(getattr(ppo_trainer, "rd_train_flags", {}) or {})
    if saved_flags is not None:
        saved_flags = {k: v for k, v in saved_flags.items() if k != "freeze_stage1"}
    if saved_flags is None:
        print(f"[Resume] WARNING: {state_path} has no train_flags; cannot check that "
              f"this resume trains under the same settings: {flags}", flush=True)
    elif saved_flags != flags:
        diff = {k: (saved_flags.get(k), flags.get(k))
                for k in sorted(set(saved_flags) | set(flags)) if saved_flags.get(k) != flags.get(k)}
        raise SystemExit(f"{state_path} was trained under different settings "
                         f"(saved, now): {diff}; pass the same --set flags to resume")
    if "kl_coef" in state:
        ppo_trainer.kl_ctl.value = float(state["kl_coef"])
        print(f"[Resume] kl_coef={ppo_trainer.kl_ctl.value:.6f}", flush=True)
    if "optimizer" in state and getattr(ppo_trainer, "optimizer", None) is not None:
        ppo_trainer.optimizer.load_state_dict(state["optimizer"])
    if "lr_scheduler" in state and getattr(ppo_trainer, "lr_scheduler", None) is not None:
        ppo_trainer.lr_scheduler.load_state_dict(state["lr_scheduler"])
    _restore_rng_state(state.get("rng_state"))

    start_global_step = int(state["global_step"])
    # derived, not read back: global_step is absolute, and the saved epoch_idx is the
    # epoch that produced the step, which is the previous one on an epoch boundary
    start_epoch_idx = start_global_step // batches
    skip = start_global_step % batches
    best_reward = float(state.get("best_reward", -100.0))
    best_reward_step = int(state.get("best_reward_step", -1))
    print(f"[Resume] state <- {state_path}", flush=True)
    print(f"[Resume] global_step={start_global_step} epoch={start_epoch_idx} "
          f"skip={skip}/{batches}", flush=True)
    return start_epoch_idx, skip, best_reward, best_reward_step


def summarize_reward_batch(results, responses_decoded, rewards, metric: Metric, threshold: float):
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "parse_fail": 0,
            "conf_none": 0,
            "conf_out_of_range": 0,
            "correct": 0,
            "reward_floor_hits": 0,
            "long_digit_conf_pattern": 0,
            "examples_parse_fail": [],
            "examples_conf_out_of_range": [],
            "examples_long_digit_conf": [],
            "conf_values": [],
        }

    parse_fail = 0
    conf_none = 0
    conf_out_of_range = 0
    correct = 0
    reward_floor_hits = 0
    long_digit_conf_pattern = 0
    examples_parse_fail = []
    examples_conf_out_of_range = []
    examples_long_digit_conf = []
    conf_values = []

    for idx, (res, resp_text, rew) in enumerate(zip(results, responses_decoded, rewards)):
        text = str(resp_text or "")
        if res.is_wrong_format:
            parse_fail += 1
            if len(examples_parse_fail) < 3:
                examples_parse_fail.append((idx, text[:220].replace("\n", "\\n")))

        conf = res.confidence
        conf_values.append(conf)
        if conf is None:
            conf_none += 1
        elif conf < 0 or conf > 10:
            conf_out_of_range += 1
            if len(examples_conf_out_of_range) < 3:
                examples_conf_out_of_range.append((idx, conf, text[:220].replace("\n", "\\n")))

        if re.search(r"Confidence:\s*\d{3,}", text):
            long_digit_conf_pattern += 1
            if len(examples_long_digit_conf) < 3:
                examples_long_digit_conf.append((idx, text[:220].replace("\n", "\\n")))

        if float(rew) <= -29.999:  # -30 = wrong_format_penalty in rd_util/RLHelper.py
            reward_floor_hits += 1

        if res.prediction is not None and is_answer_correct(res.prediction, res.gt_candidates, metric, threshold):
            correct += 1

    return {
        "total": total,
        "parse_fail": parse_fail,
        "conf_none": conf_none,
        "conf_out_of_range": conf_out_of_range,
        "correct": correct,
        "reward_floor_hits": reward_floor_hits,
        "long_digit_conf_pattern": long_digit_conf_pattern,
        "examples_parse_fail": examples_parse_fail,
        "examples_conf_out_of_range": examples_conf_out_of_range,
        "examples_long_digit_conf": examples_long_digit_conf,
        # the collapse to a constant confidence is only visible here
        "conf_values": conf_values,
    }


def train(out_dir: str, lr: float, epochs: int, batchsize: int, model_dir: str, tokenizer_dir: str, dataset: str, log_with: str = "tensorboard", is_unsloth: bool = True, use_lora: bool = True, max_train_samples: int = 10000, max_validation_samples: int = 1000, reward_log_interval: int = 10, shuffle_seed: int = 42, eval_every_steps: int = 0, save_every_steps: int = 250, max_eval_train_samples: int = 1000, debug_sample_log_interval: int = 100, resume_state_path: str | None = None, resume_auto: bool = False,
          chat_end_markers=(), max_seq_length: int = 1048,
          use_exact_model_name: bool = True, ppo_max_new_tokens: int = 500,
          ppo_temperature: float | None = None, ppo_repetition_penalty: float | None = 1.0,
          rollout_max_new_tokens: int = 256, rollout_temperature: float = 0.6,
          rollout_top_p: float = 0.9, chat_template_kwargs=None, lora=None,
          system_prompt: str | None = None, confidence_stop: bool = False,
          kl_horizon: int = 10000, max_grad_norm: float | None = None,
          strict_format: bool = False,
          ppo_early_stopping: bool = False, ppo_target_kl: float = 1.0,
          ppo_mini_batch_size: int | None = None, ppo_gradient_accumulation_steps: int = 1,
          unpadded_generation: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    resume_path, adapter_dir = resolve_resume(out_dir, resume_state_path, resume_auto)
    if use_lora:
        model, tokenizer = load_lora_model_tokenizer(
            model_dir, is_unsloth, device, max_seq_length=max_seq_length,
            use_exact_model_name=use_exact_model_name, lora=lora,
            adapter_dir=adapter_dir)
    else:
        if adapter_dir:
            raise SystemExit("resume needs use_lora: true; there is no adapter to reload")
        model, tokenizer = load_model_tokenizer(
            model_dir, is_unsloth, device, max_seq_length=max_seq_length,
            use_exact_model_name=use_exact_model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side = "left"
    reward_metric, reward_threshold = get_metric_and_threshold(dataset)

    dataset_train = load_prepared_dataset(dataset, "train", "verbalize", tokenizer,
                                          template_kwargs=chat_template_kwargs,
                                          system_prompt=system_prompt)
    dataset_train = shuffle_and_limit_dataset(dataset_train, max_train_samples, shuffle_seed)
    print(f"Training samples: {len(dataset_train)}")
    dataset_train_eval = shuffle_and_limit_dataset(dataset_train, max_eval_train_samples, shuffle_seed)
    print(f"Train-eval samples: {len(dataset_train_eval)}")
    dataset_validation = load_prepared_dataset(dataset, "validation", "verbalize", tokenizer,
                                               template_kwargs=chat_template_kwargs,
                                               system_prompt=system_prompt)
    dataset_validation = shuffle_and_limit_dataset(dataset_validation, max_validation_samples, shuffle_seed)
    print(f"Validation samples: {len(dataset_validation)}")

    pad_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    pad_tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_tokenizer.padding_side = "left"

    data_collator = DataCollatorForTokenizedQueries(pad_tokenizer)
    dataloader_train_eval = DataLoader(dataset_train_eval, batch_size=max(1, 2 * batchsize), collate_fn=data_collator)
    dataloader_validation = DataLoader(dataset_validation, batch_size=max(1, 2 * batchsize), collate_fn=data_collator)

    if is_unsloth:
        FastLanguageModel.for_inference(model.pretrained_model)

    config = PPOConfig(
        learning_rate=lr,
        task_name="gpt",
        batch_size=batchsize,
        # default (batch/2, 1) is upstream; halving the mini-batch and doubling the
        # accumulation gives the same update at half the activation memory (e.g. 32B models)
        mini_batch_size=int(ppo_mini_batch_size) if ppo_mini_batch_size else int(batchsize / 2),
        gradient_accumulation_steps=int(ppo_gradient_accumulation_steps),
        log_with=log_with,
        project_kwargs={"logging_dir": out_dir},
        remove_unused_columns=False,
        optimize_device_cache=True,
        init_kl_coef=0.05,
        horizon=int(kl_horizon),
        max_grad_norm=max_grad_norm,
        early_stopping=ppo_early_stopping,
        target_kl=ppo_target_kl,
    )

    ppo_trainer = PPOTrainerNoCache(
        model=model,
        config=config,
        dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    ppo_trainer.rd_train_flags = {
        "lr": float(lr), "confidence_stop": bool(confidence_stop),
        "strict_format": bool(strict_format),
        "ppo_early_stopping": bool(ppo_early_stopping), "ppo_target_kl": float(ppo_target_kl),
        "kl_horizon": int(kl_horizon), "max_grad_norm": max_grad_norm,
        "ppo_temperature": ppo_temperature, "ppo_repetition_penalty": ppo_repetition_penalty,
        "ppo_mini_batch_size": int(ppo_mini_batch_size) if ppo_mini_batch_size else int(batchsize / 2),
        "ppo_gradient_accumulation_steps": int(ppo_gradient_accumulation_steps),
        "unpadded_generation": bool(unpadded_generation),
        "batch_size": int(batchsize), "epochs": int(epochs),
        "lora_rank": int((lora or {}).get("rank", 32)), "lora_alpha": int((lora or {}).get("alpha", 32)),
    }

    # upstream also stops stage 1 at " Confidence", so the PPO response is the
    # confidence itself
    prediction_markers = list(chat_end_markers or [])
    if confidence_stop:
        prediction_markers.append("\u0120Confidence")
    prediction_terminators = terminator_ids(tokenizer, prediction_markers)
    # the token only covers a space-led " Confidence", so the decoded word backs it up
    word_stop = confidence_stop
    if confidence_stop:
        print(f"[train] confidence_stop terminators={prediction_terminators} "
              f"word_stop={word_stop}", flush=True)

    ppo_terminators = terminator_ids(tokenizer, chat_end_markers)

    generation_kwargs_prediction = {
        "max_new_tokens": int(rollout_max_new_tokens),
        "eos_token_id": prediction_terminators,
        "do_sample": True,
        "temperature": float(rollout_temperature),
        "top_p": float(rollout_top_p),
        "pad_token_id": tokenizer.eos_token_id
    }

    generation_kwargs_ppo = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": int(ppo_max_new_tokens),
        "eos_token_id": ppo_terminators,
    }
    # upstream leaves the temperature to each model's generation_config (qwen 0.7,
    # llama 0.6, mistral 1.0) while PPO scores the samples at 1.0; None keeps that
    if ppo_temperature is not None:
        generation_kwargs_ppo["temperature"] = float(ppo_temperature)
    if ppo_repetition_penalty is not None:
        generation_kwargs_ppo["repetition_penalty"] = float(ppo_repetition_penalty)
    value_stop = StopAfterConfidenceValue(tokenizer, tail=int(ppo_max_new_tokens) + 48)
    ppo_end_ids = set(ppo_terminators) - {tokenizer.eos_token_id}
    if confidence_stop:
        generation_kwargs_ppo["stopping_criteria"] = StoppingCriteriaList([value_stop])

    eval_terminators = terminator_ids(tokenizer, chat_end_markers)

    generation_kwargs_eval = {
        "max_new_tokens": int(rollout_max_new_tokens),
        "eos_token_id": eval_terminators,
        "do_sample": True,
        "temperature": float(rollout_temperature),
        "top_p": float(rollout_top_p),
        "pad_token_id": tokenizer.eos_token_id
    }

    start_epoch_idx, resume_skip, best_reward, best_reward_step = (
        maybe_load_training_state(ppo_trainer, resume_path)
    )
    csv_path = os.path.join(out_dir, "eval_metrics.csv")


    # start_epoch_idx is absolute, so epochs is the total, not the number still to run
    last_saved_step = -1
    for epoch in range(start_epoch_idx, epochs):
        rewards_epoch = []
        epoch_debug_acc = {
            "total": 0,
            "parse_fail": 0,
            "conf_none": 0,
            "conf_out_of_range": 0,
            "correct": 0,
            "reward_floor_hits": 0,
            "long_digit_conf_pattern": 0,
        }
        progress_bar = tqdm(
            enumerate(ppo_trainer.dataloader),
            total=len(ppo_trainer.dataloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        # the iterator restarts at 0 on resume while global_step carries on, so skip
        # what this epoch already consumed. TRL reshuffles on every __iter__, so the
        # remaining order is not the one the interrupted run would have seen.
        skip = resume_skip if epoch == start_epoch_idx else 0
        if skip:
            print(f"[Resume] skipping {skip} batches already consumed this epoch", flush=True)
        for idx, batch in progress_bar:
            if idx < skip:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_candidates = batch["gt_candidates"]
            questions = batch["question"]
            is_multiple_choice = batch["is_multiple_choice"]

            if is_unsloth:
                FastLanguageModel.for_inference(model.pretrained_model)
            else:
                model.eval()

            # the PPO query is prompt + stage-1 output, so the split is after the
            # answer, as upstream does
            prediction_stop = (
                StoppingCriteriaList([StopAtConfidence(tokenizer, int(input_ids.shape[1]))])
                if word_stop else None)
            if unpadded_generation:
                # one prompt at a time, unpadded: unsloth's batched inference corrupts
                # left-padded rows on long prompts (hotpot_context)
                prediction = []
                for i in range(len(input_ids)):
                    row = input_ids[i][attention_mask[i].bool()][None]
                    stop_i = (StoppingCriteriaList([StopAtConfidence(tokenizer, int(row.shape[1]))])
                              if word_stop else None)
                    out = model.generate(input_ids=row, attention_mask=torch.ones_like(row),
                                         stopping_criteria=stop_i, **generation_kwargs_prediction)
                    prediction.append(remove_padding(out[0], tokenizer.pad_token_id))
            else:
                prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            stopping_criteria=prediction_stop,
                                            **generation_kwargs_prediction)
                prediction = [remove_padding(p, tokenizer.pad_token_id) for p in prediction]
            # strip left pads, which TRL would otherwise feed to stage 2 and the PPO
            # forward unmasked (an upstream issue)
            prompt_lens = [int(attention_mask[i].sum()) for i in range(len(prediction))]
            if not unpadded_generation:
                prediction = [p[int(input_ids.shape[1]) - n:] for p, n in zip(prediction, prompt_lens)]

            if not is_unsloth:
                model.train()
            if unpadded_generation:
                # stage 2 off the KV cache as well (see sample_without_cache)
                response_tensors = [
                    sample_without_cache(
                        model.pretrained_model if hasattr(model, "pretrained_model") else model,
                        tokenizer, p, max_new_tokens=int(ppo_max_new_tokens),
                        temperature=float(ppo_temperature) if ppo_temperature is not None else 1.0,
                        top_p=1.0, stop_ids=set(ppo_terminators),
                        value_stop=value_stop if confidence_stop else None)
                    for p in prediction]
            else:
                response_tensors = ppo_trainer.generate(prediction, return_prompt=False,
                                                        **generation_kwargs_ppo)
            if confidence_stop and not unpadded_generation:
                # drop the trailing pad (== eos) TRL keeps on rows a stopping criterion
                # or non-eos terminator ended; that eos was never sampled
                response_tensors = [
                    r[:-1] if (len(r) > 1 and int(r[-1]) == tokenizer.eos_token_id and (
                        int(r[-2]) in ppo_end_ids
                        or bool(value_stop(torch.cat((p, r[:-1]))[None], None)[0])))
                    else r
                    for p, r in zip(prediction, response_tensors)]

            total_tensor = [torch.cat((p, c), 0) for p, c in zip(prediction, response_tensors)]
            answer_only_tensor = [total_tensor[i][prompt_lens[i]:] for i in range(len(input_ids))]
            responses_decoded = tokenizer.batch_decode(answer_only_tensor, skip_special_tokens=True)
            query_tensors = prediction

            results = [response_to_QAResult(question, response, gt, is_mc, strict=strict_format) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
            rewards = [QAResult_to_reward(r, metric=reward_metric, threshold=reward_threshold) for r in results]
            rewards_epoch += rewards
            batch_debug = summarize_reward_batch(results, responses_decoded, rewards, reward_metric, reward_threshold)
            # diagnostic only: how long the PPO continuation ran
            ppo_resp_lens = [int(r.shape[0]) for r in response_tensors]
            for k in epoch_debug_acc.keys():
                epoch_debug_acc[k] += int(batch_debug[k])
            batch_reward = float(np.mean(rewards))
            rewards = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]

            batch["response"] = responses_decoded
            batch["query"] = batch["question"]

            try:
                if is_unsloth:
                    FastLanguageModel.for_training(model.pretrained_model)
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            except IndexError:
                print(f"INDEX ERROR detected and ignored ({idx})")
                continue

            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "answer"])
            progress_bar.set_postfix(batch_reward=f"{batch_reward:.4f}")
            if reward_log_interval > 0 and ((idx + 1) % reward_log_interval == 0):
                running_reward = float(np.mean(rewards_epoch))
                print(
                    f"Epoch {epoch + 1} Step {idx + 1}/{len(ppo_trainer.dataloader)} "
                    f"batch_reward={batch_reward:.4f} running_avg_reward={running_reward:.4f}"
                )
                bt = max(1, batch_debug["total"])
                et = max(1, epoch_debug_acc["total"])
                print(
                    f"[RewardDebug][batch] parse_fail={batch_debug['parse_fail']}/{bt} "
                    f"conf_none={batch_debug['conf_none']}/{bt} conf_out_of_range={batch_debug['conf_out_of_range']}/{bt} "
                    f"correct={batch_debug['correct']}/{bt} floor_hits={batch_debug['reward_floor_hits']}/{bt} "
                    f"long_digit_conf={batch_debug['long_digit_conf_pattern']}/{bt} "
                    f"ppo_resp_len mean={float(np.mean(ppo_resp_lens)):.1f} max={max(ppo_resp_lens)} "
                    f"conf={[('-' if c is None else c) for c in batch_debug['conf_values']]}"
                )
                print(
                    f"[RewardDebug][epoch_acc] parse_fail={epoch_debug_acc['parse_fail']}/{et} "
                    f"conf_none={epoch_debug_acc['conf_none']}/{et} conf_out_of_range={epoch_debug_acc['conf_out_of_range']}/{et} "
                    f"correct={epoch_debug_acc['correct']}/{et} floor_hits={epoch_debug_acc['reward_floor_hits']}/{et} "
                    f"long_digit_conf={epoch_debug_acc['long_digit_conf_pattern']}/{et}"
                )
                if debug_sample_log_interval > 0 and ((idx + 1) % debug_sample_log_interval == 0):
                    if batch_debug["examples_parse_fail"]:
                        print("[RewardDebug][examples][parse_fail]")
                        for ex_idx, ex_txt in batch_debug["examples_parse_fail"]:
                            print(f"  - sample#{ex_idx}: {ex_txt}")
                    if batch_debug["examples_conf_out_of_range"]:
                        print("[RewardDebug][examples][conf_out_of_range]")
                        for ex_idx, ex_conf, ex_txt in batch_debug["examples_conf_out_of_range"]:
                            print(f"  - sample#{ex_idx} conf={ex_conf}: {ex_txt}")
                    if batch_debug["examples_long_digit_conf"]:
                        print("[RewardDebug][examples][long_digit_conf]")
                        for ex_idx, ex_txt in batch_debug["examples_long_digit_conf"]:
                            print(f"  - sample#{ex_idx}: {ex_txt}")

            # absolute: with the fast-forward above, (epoch, idx) fixes the step
            global_step = epoch * len(ppo_trainer.dataloader) + idx + 1
            # checkpoints are written regardless of eval; model_finetuned_best needs eval on
            if save_every_steps > 0 and (global_step % save_every_steps == 0):
                step_ckpt_dir = os.path.join(out_dir, f"model_finetuned_step_{global_step}")
                ppo_trainer.save_pretrained(step_ckpt_dir)
                ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))
                save_training_state(
                    ppo_trainer=ppo_trainer,
                    out_dir=out_dir,
                    global_step=global_step,
                    epoch_idx=epoch,
                    best_reward=best_reward,
                    best_reward_step=best_reward_step,
                    ckpt_dir=step_ckpt_dir,
                )
                last_saved_step = global_step

            # the final checkpoint is scored by the formal evaluation, not here
            if (eval_every_steps > 0 and global_step % eval_every_steps == 0
                    and global_step < len(ppo_trainer.dataloader) * epochs):
                if is_unsloth:
                    FastLanguageModel.for_inference(model.pretrained_model)
                else:
                    model.eval()
                # validation samples with torch's global RNG; fork it so the training run
                # (PPO sampling, data order) does not depend on how often it validates
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    train_eval_metrics = evaluate_subset(
                        model,
                        dataloader_train_eval,
                        tokenizer,
                        generation_kwargs_eval,
                        device,
                        "train",
                        reward_metric,
                        reward_threshold,
                        strict=strict_format,
                    )
                    val_metrics = evaluate_subset(
                        model,
                        dataloader_validation,
                        tokenizer,
                        generation_kwargs_eval,
                        device,
                        "validation",
                        reward_metric,
                        reward_threshold,
                        strict=strict_format,
                    )
                append_eval_metrics(
                    csv_path,
                    [
                        {
                            "Step": global_step,
                            "Split": "train",
                            "NumResults": train_eval_metrics["NumResults"],
                            "RewardMean": train_eval_metrics["RewardMean"],
                            "RewardStd": train_eval_metrics["RewardStd"],
                            "Accuracy": train_eval_metrics["Accuracy"],
                            "ECE": train_eval_metrics["ECE"],
                            "AUROC": train_eval_metrics["AUROC"],
                        },
                        {
                            "Step": global_step,
                            "Split": "validation",
                            "NumResults": val_metrics["NumResults"],
                            "RewardMean": val_metrics["RewardMean"],
                            "RewardStd": val_metrics["RewardStd"],
                            "Accuracy": val_metrics["Accuracy"],
                            "ECE": val_metrics["ECE"],
                            "AUROC": val_metrics["AUROC"],
                        },
                    ],
                )
                print(
                    f"Validation step {global_step}: "
                    f"reward={val_metrics['RewardMean']:.4f}, "
                    f"acc={val_metrics['Accuracy']:.4f}, "
                    f"ece={val_metrics['ECE']:.4f}, "
                    f"auroc={val_metrics['AUROC']:.4f}"
                )

                mean_reward = val_metrics["RewardMean"]
                if mean_reward > best_reward:
                    ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned_best"))
                    best_reward = mean_reward
                    best_reward_step = global_step
                    # refresh the state this step already saved, so its best_reward is
                    # current; a step without an adapter must not become the resume point
                    if save_every_steps > 0 and global_step % save_every_steps == 0:
                        save_training_state(
                            ppo_trainer=ppo_trainer,
                            out_dir=out_dir,
                            global_step=global_step,
                            epoch_idx=epoch,
                            best_reward=best_reward,
                            best_reward_step=best_reward_step,
                            ckpt_dir=os.path.join(out_dir, f"model_finetuned_step_{global_step}"),
                        )

        avg_reward = float(np.mean(rewards_epoch))
        print(f"Finished epoch {epoch + 1}. Average train reward: {avg_reward}")

    # the run length need not be a multiple of save_every_steps
    final_step = len(ppo_trainer.dataloader) * epochs
    if last_saved_step < final_step:
        final_dir = os.path.join(out_dir, f"model_finetuned_step_{final_step}")
        ppo_trainer.save_pretrained(final_dir)
        ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))
        save_training_state(ppo_trainer=ppo_trainer, out_dir=out_dir, global_step=final_step,
                            epoch_idx=epochs - 1, best_reward=best_reward,
                            best_reward_step=best_reward_step, ckpt_dir=final_dir)

    print("Finished Training!")
    print(f"Best validation reward {best_reward} at step {best_reward_step}")

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, with_method=False, dataset_default="hotpot",
                    dataset_choices=sorted(RD_DATASET_NAMES))
    parser.add_argument("--resume_state_path", type=str, default=None,
                        help="training_state.pt to resume from")
    parser.add_argument("--resume_auto", action="store_true",
                        help="resume from the latest training state in the output dir")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config("RD", args.model, args.dataset, args.overrides)

    train_cfg = cfg["method"]["train"]
    seed_everything(int(cfg["seed"]))

    exp_name = args.exp_name or f"RD_{cfg['model']['name']}_{cfg['dataset']['name']}"
    out_dir = checkpoint_dir(cfg, args.exp_name)
    run_dir = os.path.join(cfg["paths"]["results"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[train] {exp_name}\n[train] checkpoints -> {out_dir}\n[train] logs -> {run_dir}")

    hf_id = cfg["model"]["hf_id"]
    train(
        out_dir,
        lr=float(train_cfg["learning_rate"]),
        epochs=int(train_cfg["epochs"]),
        batchsize=int(train_cfg["batch_size"]),
        model_dir=hf_id,
        tokenizer_dir=hf_id,
        dataset=rd_dataset_name(cfg["dataset"]["name"]),
        log_with="tensorboard",
        is_unsloth=bool(train_cfg.get("is_unsloth", True)),
        use_lora=bool(train_cfg.get("use_lora", True)),
        max_train_samples=int(train_cfg["max_train_samples"]),
        max_validation_samples=int(train_cfg["max_validation_samples"]),
        max_eval_train_samples=int(train_cfg["max_eval_train_samples"]),
        reward_log_interval=int(train_cfg["reward_log_interval"]),
        debug_sample_log_interval=int(train_cfg.get("debug_sample_log_interval", 100)),
        shuffle_seed=int(train_cfg["shuffle_seed"]),
        eval_every_steps=int(train_cfg["eval_every_steps"]),
        save_every_steps=int(train_cfg["save_every_steps"]),
        resume_state_path=args.resume_state_path,
        resume_auto=args.resume_auto,
        chat_end_markers=cfg["model"].get("chat_end_markers", []),
        max_seq_length=int(train_cfg["max_seq_length"]),
        use_exact_model_name=bool(cfg["model"].get("use_exact_model_name", True)),
        ppo_max_new_tokens=int(train_cfg["ppo_max_new_tokens"]),
        ppo_temperature=(float(train_cfg["ppo_temperature"])
                         if train_cfg.get("ppo_temperature") is not None else None),
        ppo_repetition_penalty=(float(train_cfg.get("ppo_repetition_penalty", 1.0))
                                if train_cfg.get("ppo_repetition_penalty", 1.0) is not None else None),
        rollout_max_new_tokens=int(train_cfg["rollout_max_new_tokens"]),
        rollout_temperature=float(train_cfg["rollout_temperature"]),
        rollout_top_p=float(train_cfg["rollout_top_p"]),
        chat_template_kwargs=cfg["model"].get("chat_template_kwargs") or {},
        lora=cfg["model"].get("lora") or {},
        system_prompt=cfg["method"]["system_prompt"],
        confidence_stop=bool(train_cfg.get("confidence_stop", False)),
        kl_horizon=int(train_cfg.get("kl_horizon", 10000)),
        max_grad_norm=(float(train_cfg["max_grad_norm"])
                       if train_cfg.get("max_grad_norm") is not None else None),
        strict_format=bool(train_cfg.get("strict_format", False)),
        ppo_early_stopping=bool(train_cfg.get("ppo_early_stopping", False)),
        ppo_target_kl=float(train_cfg.get("ppo_target_kl", 1.0)),
        ppo_mini_batch_size=(int(train_cfg["ppo_mini_batch_size"])
                             if train_cfg.get("ppo_mini_batch_size") is not None else None),
        ppo_gradient_accumulation_steps=int(train_cfg.get("ppo_gradient_accumulation_steps", 1)),
        unpadded_generation=bool(train_cfg.get("unpadded_generation", False)),
    )


if __name__ == "__main__":
    main()

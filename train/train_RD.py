#!/usr/bin/env python
"""RD (Rewarding Doubt) baseline: PPO on a verbalized confidence head.

Port of https://github.com/pasta99/RewardingDoubt (arXiv:2503.02623); rd_util/ stays
close to upstream. Needs the csr-rd env (trl 0.8.6 PPO API, from environment-rd.yml);
RD *evaluation* instead runs in the main env, see evaluation/eval_verbalized.py.

    python train/train_RD.py --model qwen
    python train/train_RD.py --model llama --set method.train.batch_size=8
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from rd_util.DataHelper import DataCollatorForTokenizedQueries
from unsloth import FastLanguageModel
from rd_util.ppo_trainer_no_cache import PPOTrainerNoCache
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


# stop ids = EOS + the model's end-of-turn markers from its config (upstream hardcoded
# Llama's <|eot_id|>); markers absent from this vocabulary are skipped
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
# reward by exact match, which is unreachable for HotpotQA-style answers.
def get_metric_and_threshold(dataset_name: str):
    if dataset_name in {"triviaqa", "qampari", "hotpotqa", "hotpotqa_context"}:
        return Metric.F1, 0.5
    return Metric.EXACT, 0.5


def safe_metric(metric_fn, results, metric: Metric, threshold: float = 0.5, max_confidence: int = 10):
    try:
        return metric_fn(results, metric, threshold, max_confidence)
    except ValueError:
        return float("nan")


def evaluate_subset(model, dataloader, tokenizer, generation_kwargs, device, split_name: str, metric: Metric, threshold: float):
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

        results = [response_to_QAResult(question, response, gt, is_mc) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
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


def maybe_load_training_state(ppo_trainer, out_dir: str, resume_state_path: str | None = None, resume_auto: bool = False):
    cand = None
    if resume_state_path:
        cand = resume_state_path
    elif resume_auto:
        auto_path = os.path.join(out_dir, "training_state_latest.pt")
        if os.path.exists(auto_path):
            cand = auto_path
    if not cand:
        return 0, 0, -100.0, -1
    if not os.path.exists(cand):
        print(f"[Resume] state file not found, skip: {cand}")
        return 0, 0, -100.0, -1

    state = torch.load(cand, map_location="cpu")
    if "optimizer" in state and getattr(ppo_trainer, "optimizer", None) is not None:
        ppo_trainer.optimizer.load_state_dict(state["optimizer"])
    if "lr_scheduler" in state and getattr(ppo_trainer, "lr_scheduler", None) is not None:
        ppo_trainer.lr_scheduler.load_state_dict(state["lr_scheduler"])
    _restore_rng_state(state.get("rng_state"))

    start_global_step = int(state.get("global_step", 0))
    start_epoch_idx = int(state.get("epoch_idx", 0))
    best_reward = float(state.get("best_reward", -100.0))
    best_reward_step = int(state.get("best_reward_step", -1))
    print(f"[Resume] loaded trainer state <- {cand}")
    print(f"[Resume] start_global_step={start_global_step} start_epoch_idx={start_epoch_idx}")
    return start_global_step, start_epoch_idx, best_reward, best_reward_step


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

    for idx, (res, resp_text, rew) in enumerate(zip(results, responses_decoded, rewards)):
        text = str(resp_text or "")
        if res.is_wrong_format:
            parse_fail += 1
            if len(examples_parse_fail) < 3:
                examples_parse_fail.append((idx, text[:220].replace("\n", "\\n")))

        conf = res.confidence
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
    }


def train(out_dir: str, lr: float, epochs: int, batchsize: int, model_dir: str, tokenizer_dir: str, dataset: str, log_with: str = "tensorboard", is_unsloth: bool = True, use_lora: bool = True, max_train_samples: int = 10000, max_validation_samples: int = 1000, reward_log_interval: int = 10, shuffle_seed: int = 42, eval_every_steps: int = 0, save_every_steps: int = 250, max_eval_train_samples: int = 1000, debug_sample_log_interval: int = 100, resume_state_path: str | None = None, resume_auto: bool = False,
          chat_end_markers=(), max_seq_length: int = 1048,
          use_exact_model_name: bool = True, ppo_max_new_tokens: int = 500,
          rollout_max_new_tokens: int = 256, rollout_temperature: float = 0.6,
          rollout_top_p: float = 0.9) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_lora:
        model, tokenizer = load_lora_model_tokenizer(
            model_dir, is_unsloth, device, max_seq_length=max_seq_length,
            use_exact_model_name=use_exact_model_name)
    else:
        model, tokenizer = load_model_tokenizer(
            model_dir, is_unsloth, device, max_seq_length=max_seq_length,
            use_exact_model_name=use_exact_model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side = "left"
    reward_metric, reward_threshold = get_metric_and_threshold(dataset)

    dataset_train = load_prepared_dataset(dataset, "train", "verbalize", tokenizer)
    dataset_train = shuffle_and_limit_dataset(dataset_train, max_train_samples, shuffle_seed)
    print(f"Training samples: {len(dataset_train)}")
    dataset_train_eval = shuffle_and_limit_dataset(dataset_train, max_eval_train_samples, shuffle_seed)
    print(f"Train-eval samples: {len(dataset_train_eval)}")
    dataset_validation = load_prepared_dataset(dataset, "validation", "verbalize", tokenizer)
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
        mini_batch_size=int(batchsize / 2),
        log_with=log_with,
        project_kwargs={"logging_dir": out_dir},
        remove_unused_columns=False,
        optimize_device_cache=True,
        init_kl_coef=0.05,
    )

    ppo_trainer = PPOTrainerNoCache(
        model=model,
        config=config,
        dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    prediction_terminators = terminator_ids(tokenizer, chat_end_markers)

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

    eval_terminators = terminator_ids(tokenizer, chat_end_markers)

    generation_kwargs_eval = {
        "max_new_tokens": int(rollout_max_new_tokens),
        "eos_token_id": eval_terminators,
        "do_sample": True,
        "temperature": float(rollout_temperature),
        "top_p": float(rollout_top_p),
        "pad_token_id": tokenizer.eos_token_id
    }

    start_global_step, start_epoch_idx, best_reward, best_reward_step = maybe_load_training_state(
        ppo_trainer=ppo_trainer,
        out_dir=out_dir,
        resume_state_path=resume_state_path,
        resume_auto=resume_auto,
    )
    csv_path = os.path.join(out_dir, "eval_metrics.csv")


    for epoch in range(start_epoch_idx, start_epoch_idx + epochs):
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
        for idx, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_candidates = batch["gt_candidates"]
            questions = batch["question"]
            is_multiple_choice = batch["is_multiple_choice"]

            if is_unsloth:
                FastLanguageModel.for_inference(model.pretrained_model)
            else:
                model.eval()

            prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs_prediction)
            prediction = [remove_padding(p, tokenizer.pad_token_id) for p in prediction]

            if not is_unsloth:
                model.train()
            response_tensors = ppo_trainer.generate(prediction, return_prompt=False, **generation_kwargs_ppo)

            total_tensor = [torch.cat((p, c), 0) for p, c in zip(prediction, response_tensors)]
            answer_only_tensor = [total_tensor[i][len(input_ids[i]):] for i in range(len(input_ids))]
            responses_decoded = tokenizer.batch_decode(answer_only_tensor, skip_special_tokens=True)

            results = [response_to_QAResult(question, response, gt, is_mc) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
            rewards = [QAResult_to_reward(r, metric=reward_metric, threshold=reward_threshold) for r in results]
            rewards_epoch += rewards
            batch_debug = summarize_reward_batch(results, responses_decoded, rewards, reward_metric, reward_threshold)
            for k in epoch_debug_acc.keys():
                epoch_debug_acc[k] += int(batch_debug[k])
            batch_reward = float(np.mean(rewards))
            rewards = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]

            batch["response"] = responses_decoded
            batch["query"] = batch["question"]

            try:
                if is_unsloth:
                    FastLanguageModel.for_training(model.pretrained_model)
                stats = ppo_trainer.step(prediction, response_tensors, rewards)
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
                    f"long_digit_conf={batch_debug['long_digit_conf_pattern']}/{bt}"
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

            local_step = (epoch - start_epoch_idx) * len(ppo_trainer.dataloader) + (idx + 1)
            global_step = start_global_step + local_step
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

            if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                if is_unsloth:
                    FastLanguageModel.for_inference(model.pretrained_model)
                else:
                    model.eval()
                train_eval_metrics = evaluate_subset(
                    model,
                    dataloader_train_eval,
                    tokenizer,
                    generation_kwargs_eval,
                    device,
                    "train",
                    reward_metric,
                    reward_threshold,
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
                    save_training_state(
                        ppo_trainer=ppo_trainer,
                        out_dir=out_dir,
                        global_step=global_step,
                        epoch_idx=epoch,
                        best_reward=best_reward,
                        best_reward_step=best_reward_step,
                    )

        avg_reward = float(np.mean(rewards_epoch))
        print(f"Finished epoch {epoch + 1}. Average train reward: {avg_reward}")

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
        rollout_max_new_tokens=int(train_cfg["rollout_max_new_tokens"]),
        rollout_temperature=float(train_cfg["rollout_temperature"]),
        rollout_top_p=float(train_cfg["rollout_top_p"]),
    )


if __name__ == "__main__":
    main()

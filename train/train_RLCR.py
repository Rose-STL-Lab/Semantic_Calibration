#!/usr/bin/env python
"""RLCR baseline: GRPO over an explicit think / answer / analysis / confidence
interface, with format, exact-match and Brier rewards.

Rewards and weights match upstream (Damani et al. 2025), so they are not ours to
tune; scaled to one GPU: 8 rollouts x accum 4 against upstream's 32 x 64,
rank-32 LoRA on an instruct backbone against their full fine-tune. HotpotQA only, since RLCR's prompt
embeds the supporting paragraphs. Hyperparameters: configs/methods/rlcr.yaml.

    python train/train_RLCR.py --model qwen
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import (add_common_args, checkpoint_dir, dump_config, load_config,
                        seed_everything, setup_env)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # hotpot = closed-book; hotpot_context = RLCR's own "Paragraph <i>" layout
    add_common_args(parser, with_method=False, dataset_default="hotpot",
                    dataset_choices=["hotpot", "hotpot_context"])
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        default=None,
                        help="path to a checkpoint-* directory to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config("RLCR", args.model, args.dataset, args.overrides)
    setup_env(cfg)
    seed = int(cfg["seed"])
    seed_everything(seed)

    # import unsloth before trl: trl bound first gives stock GRPOConfig and NaN grads
    import unsloth  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    from csr.callbacks import JudgeCacheCallback, MetricCSVCallback, VerbalizedEvalCallback
    from csr.data import correctness_prompt_kind
    from csr.judge import make_judge
    from csr.modeling import load_policy
    from csr.rlcr import RLCRRewards, load_hotpot

    train_cfg = cfg["method"]["train"]
    exp_name = args.exp_name or f"RLCR_{cfg['model']['name']}_{cfg['dataset']['name']}"
    ckpt_dir = checkpoint_dir(cfg, args.exp_name)
    run_dir = os.path.join(cfg["paths"]["results"], exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[train] {exp_name}\n[train] checkpoints -> {ckpt_dir}\n[train] logs -> {run_dir}")

    train_ds = load_hotpot(cfg, split="train")

    # RLCR sets prompt and completion lengths independently (upstream does too),
    # so the context has to be wide enough to hold both.
    context = max(int(train_cfg["max_seq_length"]),
                  int(train_cfg["max_prompt_length"]) + int(train_cfg["max_completion_length"]))
    model, tokenizer = load_policy(
        cfg,
        context,
        gpu_memory_utilization=float(train_cfg.get("gpu_memory_utilization", 0.5)),
    )

    rewards = RLCRRewards(train_cfg.get("format_pattern", "tabc"))

    training_args = GRPOConfig(
        output_dir=ckpt_dir,
        beta=float(train_cfg["beta"]),
        loss_type=train_cfg["loss_type"],
        shuffle_dataset=True,
        learning_rate=float(train_cfg["learning_rate"]),
        adam_beta1=float(train_cfg["adam_beta1"]),
        adam_beta2=float(train_cfg["adam_beta2"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        logging_steps=int(train_cfg["logging_steps"]),
        logging_strategy="steps",
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_generations=int(train_cfg["num_generations"]),
        num_iterations=int(train_cfg["num_iterations"]),
        max_prompt_length=int(train_cfg["max_prompt_length"]),
        max_completion_length=int(train_cfg["max_completion_length"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        eval_strategy="no",
        eval_on_start=False,
        use_vllm=bool(train_cfg["use_vllm"]),
        vllm_gpu_memory_utilization=float(train_cfg["vllm_gpu_memory_utilization"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        temperature=float(train_cfg["temperature"]),
        top_p=float(train_cfg["top_p"]),
        reward_weights=list(train_cfg["reward_weights"]),
        scale_rewards=bool(train_cfg["scale_rewards"]),
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rewards.reward_funcs(),
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.add_callback(MetricCSVCallback(run_dir))
    if int(train_cfg.get("eval_every_steps", 0)) > 0:
        judge = make_judge(cfg, chat_end_markers=cfg["model"].get("chat_end_markers", []))
        validation_ds = load_hotpot(cfg, split="eval",
                                    size=int(train_cfg.get("eval_size", 1000)))
        trainer.add_callback(VerbalizedEvalCallback(
            cfg, validation_ds, tokenizer, judge, run_dir,
            correctness_prompt_kind(cfg),
            every_steps=int(train_cfg["eval_every_steps"]),
        ))
        trainer.add_callback(JudgeCacheCallback(judge))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(ckpt_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] saved final adapter -> {final_dir}")


if __name__ == "__main__":
    main()

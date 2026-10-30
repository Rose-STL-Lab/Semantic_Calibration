# Shared GRPO setup for RLVR and CSR; per-method settings from configs/methods/

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csr.config import (add_common_args, checkpoint_dir, load_config,
                        seed_everything)


def parse_args(method: str, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # 2wiki_comp is evaluation-only (no train_size); naming the trainable datasets
    # fails before the model loads, as train_RLCR.py and train_RD.py already do
    add_common_args(parser, with_method=False, dataset_default="hotpot",
                    dataset_choices=["hotpot", "hotpot_context", "trivia",
                                     "msmarco", "nq_open", "musique"])
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        default=None,
                        help="path to a checkpoint-* directory to resume training from")
    args = parser.parse_args()
    args.method = method
    return args


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.method, args.model, args.dataset, args.overrides)
    seed = int(cfg["seed"])
    seed_everything(seed)

    # import unsloth before trl: trl bound first gives stock GRPOConfig and NaN grads
    import unsloth  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    from csr.callbacks import (
        CalibrationRampCallback,
        JudgeCacheCallback,
        MetricCSVCallback,
        PeriodicEvalCallback,
    )
    from csr.data import correctness_prompt_kind, load_split
    from csr.judge import make_judge
    from csr.modeling import load_policy
    from csr.rewards import RewardComputer

    train_cfg = cfg["method"]["train"]
    method_name = cfg["method"]["name"]
    exp_name = args.exp_name or f"{method_name}_{cfg['model']['name']}_{cfg['dataset']['name']}"

    ckpt_dir = checkpoint_dir(cfg, args.exp_name)
    run_dir = os.path.join(cfg["paths"]["results"], exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    from csr.config import dump_config
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[train] {exp_name}\n[train] checkpoints -> {ckpt_dir}\n[train] logs -> {run_dir}")

    train_ds = load_split(cfg, split="train")
    validation_ds = load_split(cfg, split="eval",
                               size=int(train_cfg.get("eval_size", 1000)))

    model, tokenizer = load_policy(cfg, int(train_cfg["max_seq_length"]))

    # configs/models/<model>.yaml:chat_template_kwargs, applied at the tokenizer so
    # it reaches every renderer at once: TRL builds the GRPO prompts through
    # processing_class.apply_chat_template, and csr/callbacks.py renders the
    # periodic validation through the same method. Without this a reasoning
    # backbone trains on its own chain of thought -- the reward judges the trace
    # instead of the answer, and at this completion budget most traces never
    # close, so the judge sees no answer at all. setdefault, so an explicit
    # caller argument still wins.
    _tmpl_kwargs = cfg["model"].get("chat_template_kwargs") or {}
    if _tmpl_kwargs:
        _orig_apply = tokenizer.apply_chat_template

        def _apply_with_cfg(*a, **kw):
            for _k, _v in _tmpl_kwargs.items():
                kw.setdefault(_k, _v)
            return _orig_apply(*a, **kw)

        tokenizer.apply_chat_template = _apply_with_cfg
        print(f"[train] chat_template_kwargs = {_tmpl_kwargs}")

    judge = make_judge(cfg, chat_end_markers=cfg["model"].get("chat_end_markers", []))
    rewards = RewardComputer(cfg, judge, correctness_prompt_kind(cfg))

    max_completion_length = int(train_cfg["max_seq_length"]) - int(train_cfg["max_prompt_length"])
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
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_generations=int(train_cfg["num_generations"]),
        temperature=float(train_cfg["temperature"]),
        top_p=float(train_cfg["top_p"]),
        max_prompt_length=int(train_cfg["max_prompt_length"]),
        max_completion_length=max_completion_length,
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        report_to="none",
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rewards.reward_funcs(train_cfg["reward"]),
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.add_callback(MetricCSVCallback(
        run_dir, extra={"cali_w": lambda: float(rewards.cali_w)}))
    trainer.add_callback(JudgeCacheCallback(judge))
    if train_cfg["reward"] == "calibration_correctness":
        trainer.add_callback(CalibrationRampCallback(rewards))
    if int(train_cfg.get("eval_every_steps", 0)) > 0:
        trainer.add_callback(PeriodicEvalCallback(
            cfg, validation_ds, tokenizer, judge, run_dir,
            correctness_prompt_kind(cfg),
            every_steps=int(train_cfg["eval_every_steps"]),
        ))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(ckpt_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] saved final adapter -> {final_dir}")

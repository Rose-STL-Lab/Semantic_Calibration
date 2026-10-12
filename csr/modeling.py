"""Model loading: ``load_policy`` for a fresh LoRA policy to train,
``load_for_eval`` for a base model plus an optional adapter to score.
"""

from __future__ import annotations

import os

# unsloth must be imported before transformers/trl for its patches to apply
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

from .unsloth_compat import disable_shared_gpu_false_positive


def _lora_kwargs(cfg) -> dict:
    lora = cfg["model"]["lora"]
    return {
        "r": int(lora["rank"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora.get("dropout", 0.0)),
        "target_modules": list(lora["target_modules"]),
        # LoRA-init seed, separate from cfg["seed"] (which splits the data); 3407 is unsloth's default
        "random_state": int(lora.get("random_state", 3407)),
    }


def _prepare_tokenizer(tokenizer):
    # decoder-only models must pad on the left for correct batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_policy(cfg, max_seq_length: int, gpu_memory_utilization: float | None = None):
    """Base model + freshly initialized LoRA adapter, ready for GRPO."""
    model_cfg = cfg["model"]
    lora = model_cfg["lora"]

    disable_shared_gpu_false_positive()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["hf_id"],
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        fast_inference=True,
        max_lora_rank=int(lora["rank"]),
        use_exact_model_name=bool(model_cfg.get("use_exact_model_name", True)),
        gpu_memory_utilization=float(
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else model_cfg.get("gpu_memory_utilization", 0.7)
        ),
    )
    tokenizer = _prepare_tokenizer(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing="unsloth",
        **_lora_kwargs(cfg),
    )
    return model, tokenizer


def load_for_eval(cfg, max_seq_length: int, checkpoint: str | None = None):
    """Base model for evaluation; returns ``(model, tokenizer, lora_request)``, where
    lora_request routes the vLLM fast path through ``checkpoint`` (None for a base run)."""
    model_cfg = cfg["model"]
    lora = model_cfg["lora"]

    disable_shared_gpu_false_positive()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["hf_id"],
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        fast_inference=True,
        max_lora_rank=int(lora["rank"]),
        use_exact_model_name=bool(model_cfg.get("use_exact_model_name", True)),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.7)),
    )
    tokenizer = _prepare_tokenizer(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing=False,
        **_lora_kwargs(cfg),
    )

    lora_request = None
    if checkpoint:
        path = os.path.abspath(checkpoint) if os.path.exists(checkpoint) else checkpoint
        print(f"[model] loading LoRA adapter: {path}")
        lora_request = model.load_lora(path)
    else:
        print("[model] no adapter: evaluating the frozen base model")

    model = model.eval()
    return model, tokenizer, lora_request

# resolve_checkpoints / checkpoint_step live in results.py so the judge-only
# evaluation phase can reach them without importing unsloth.
from .results import checkpoint_step, resolve_checkpoints  # noqa: E402,F401

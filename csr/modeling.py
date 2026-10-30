# Model loading: load_policy for a fresh LoRA policy to train, load_for_eval for
# a base model plus an optional adapter to score.

from __future__ import annotations

import os

# unsloth before transformers/trl, or its patches do not apply
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
        # LoRA-init seed, separate from cfg["seed"] (which splits the data); unsloth's default
        "random_state": int(lora.get("random_state", 3407)),
    }


def _prepare_tokenizer(tokenizer):
    # decoder-only: pad left for correct batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_policy(cfg, max_seq_length: int, gpu_memory_utilization: float | None = None):
    model_cfg = cfg["model"]
    lora = model_cfg["lora"]

    disable_shared_gpu_false_positive()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["hf_id"],
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        # vLLM-backed rollouts
        fast_inference=bool(model_cfg.get("fast_inference", True)),
        max_lora_rank=int(lora["rank"]),
        use_exact_model_name=bool(model_cfg.get("use_exact_model_name", True)),
        gpu_memory_utilization=float(
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else model_cfg.get("gpu_memory_utilization", 0.7)
        ),
        # embedding table to RAM
        offload_embedding=bool(model_cfg.get("offload_embedding", False)),
    )
    tokenizer = _prepare_tokenizer(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing="unsloth",
        **_lora_kwargs(cfg),
    )
    return model, tokenizer


def load_for_eval(cfg, max_seq_length: int, checkpoint: str | None = None):
    # -> (model, tokenizer, lora_request); lora_request routes the vLLM fast path
    # through checkpoint, None for a base run
    model_cfg = cfg["model"]
    lora = model_cfg["lora"]

    disable_shared_gpu_false_positive()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["hf_id"],
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        # vLLM-backed rollouts
        fast_inference=bool(model_cfg.get("fast_inference", True)),
        max_lora_rank=int(lora["rank"]),
        use_exact_model_name=bool(model_cfg.get("use_exact_model_name", True)),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.7)),
        # keeps the embedding table in RAM
        offload_embedding=bool(model_cfg.get("offload_embedding", False)),
        # unsloth's default is "sequential", which spills to CPU the moment it
        # thinks the model will not fit -- and bitsandbytes then refuses the
        # CPU-dispatched modules outright. {"": 0} pins everything to the single
        # visible device, so a model that really is too large fails as a plain
        # OOM instead of as a quantizer complaint.
        **({"device_map": model_cfg["device_map"]} if model_cfg.get("device_map") else {}),
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

# these live in results.py so the judge-only phase can reach them without unsloth
from .results import checkpoint_step, resolve_checkpoints  # noqa: E402,F401

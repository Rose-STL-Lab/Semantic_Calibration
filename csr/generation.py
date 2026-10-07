"""Rollout generation: unsloth's vLLM fast path returns all K rollouts in one call,
otherwise one HuggingFace ``generate`` per rollout.
"""

from __future__ import annotations

from typing import Sequence

import torch


def count_tokens(tokenizer, text) -> int:
    if tokenizer is None:
        return 0
    return int(len(tokenizer(str(text or ""), add_special_tokens=False)["input_ids"]))


def _use_vllm(model) -> bool:
    return hasattr(model, "fast_generate")


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, *, n: int = 1, temperature: float = 0.7,
             top_p: float = 0.95, max_new_tokens: int = 256, seed: int = 42,
             lora_request=None) -> list[str]:
    """Return ``n`` raw completions for one rendered prompt."""
    if _use_vllm(model):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=int(n),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            seed=int(seed),
        )
        outputs = model.fast_generate([prompt], sampling_params=sampling_params,
                                      lora_request=lora_request)
        return [str(o.text or "") for o in outputs[0].outputs]

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = encoded["input_ids"].shape[-1]

    do_sample = float(temperature) > 0.0
    kwargs = dict(
        **encoded,
        do_sample=do_sample,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)

    completions = []
    for _ in range(int(n)):
        output = model.generate(**kwargs)
        completions.append(
            tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        )
    return completions


def _token_report(tokenizer, prompt: str, raw: list[str]) -> dict:
    prompt_tokens = count_tokens(tokenizer, prompt)
    completion_tokens = [count_tokens(tokenizer, text) for text in raw]
    return {
        "raw": raw,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # what the paper's Tok column measures: the whole cost of answering once
        "total_tokens": prompt_tokens * len(raw) + sum(completion_tokens),
    }


def rollouts_with_tokens(model, tokenizer, prompt: str, *, n: int, temperature: float,
                         top_p: float, max_new_tokens: int, seed: int,
                         lora_request=None) -> dict:
    """Generate ``n`` rollouts and report the prompt / completion token cost."""
    raw = generate(model, tokenizer, prompt, n=n, temperature=temperature, top_p=top_p,
                   max_new_tokens=max_new_tokens, seed=seed, lora_request=lora_request)
    return _token_report(tokenizer, prompt, raw)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: Sequence[str], *, n: int, temperature: float,
                   top_p: float, max_new_tokens: int, seed: int,
                   lora_request=None) -> list[list[str]]:
    """``generate`` for several prompts in one vLLM call, same SamplingParams (seed included)."""
    if not _use_vllm(model):
        return [generate(model, tokenizer, p, n=n, temperature=temperature, top_p=top_p,
                         max_new_tokens=max_new_tokens, seed=seed, lora_request=lora_request)
                for p in prompts]
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=int(n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )
    outputs = model.fast_generate(list(prompts), sampling_params=sampling_params,
                                  lora_request=lora_request)
    return [[str(o.text or "") for o in out.outputs] for out in outputs]


def rollouts_with_tokens_batch(model, tokenizer, prompts: Sequence[str], *, n: int,
                               temperature: float, top_p: float, max_new_tokens: int,
                               seed: int, lora_request=None) -> list[dict]:
    """``rollouts_with_tokens`` for several prompts at once, one report per prompt."""
    raws = generate_batch(model, tokenizer, prompts, n=n, temperature=temperature, top_p=top_p,
                          max_new_tokens=max_new_tokens, seed=seed, lora_request=lora_request)
    return [_token_report(tokenizer, p, raw) for p, raw in zip(prompts, raws)]


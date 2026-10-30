# Rollout generation + the prompt rendering that feeds it.
# generate(prompt, n=K) -> K rollouts for ONE prompt: one vLLM call on unsloth's
# fast path, else K HuggingFace calls.
# generate_batch(prompts, n=K) -> K rollouts for EACH prompt, K x len(prompts)
# completions in one call. That is --question-batch: it amortises the judge's
# tail latency, same K.

from __future__ import annotations

from typing import Sequence

import torch


def render_generation_prompt(prompt_messages, tokenizer, template_kwargs=None) -> str:
    # chat template, else plain concat. template_kwargs = the model yaml's
    # chat_template_kwargs -- template variables, not generation args:
    #   Qwen3 enable_thinking=False -> empty <think></think>
    # a key the template never reads is ignored
    if isinstance(prompt_messages, list) and hasattr(tokenizer, "apply_chat_template") \
            and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            **dict(template_kwargs or {}),
        )
    if isinstance(prompt_messages, list):
        chunks = []
        for msg in prompt_messages:
            chunks.append(str(msg.get("content", "")) if isinstance(msg, dict) else str(msg))
        return "\n\n".join(chunks).strip() + "\n\nAnswer:"
    return str(prompt_messages)


def count_tokens(tokenizer, text) -> int:
    if tokenizer is None:
        return 0
    return int(len(tokenizer(str(text or ""), add_special_tokens=False)["input_ids"]))


def _use_vllm(model) -> bool:
    # fast_generate is on both branches: unsloth wraps model.generate with it even
    # when fast_inference is off, so it is not a vLLM test. The sibling attribute
    # is -- vLLM gets a functools.partial, plain HF the error_out_no_vllm stub.
    if not hasattr(model, "fast_generate"):
        return False
    batches = getattr(model, "fast_generate_batches", None)
    if getattr(batches, "__name__", "") == "error_out_no_vllm":
        return False
    return True


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, *, n: int = 1, temperature: float = 0.7,
             top_p: float = 0.95, max_new_tokens: int = 256, seed: int = 42,
             lora_request=None) -> list[str]:
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

    # one call for all n samples instead of n sequential calls; greedy decoding
    # has only one distinct sample, so it stays at 1
    n_return = int(n) if do_sample else 1
    if n_return > 1:
        kwargs["num_return_sequences"] = n_return
    output = model.generate(**kwargs)

    # skip_special_tokens=False on purpose: for a channel-structured model the
    # control tokens are the only thing separating reasoning from the answer,
    # and dropping them here is unrecoverable. The trailing ones are peeled
    # later by strip_chat_markers via the model's chat_end_markers.
    completions = [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=False)
        for row in output[:n_return]
    ]
    # only reachable at temperature 0, where the n samples are identical by
    # definition, so padding them out is not inventing diversity
    while len(completions) < int(n):
        completions.append(completions[-1] if completions else "")
    return completions


@torch.inference_mode()
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
    # plus the prompt / completion token cost
    raw = generate(model, tokenizer, prompt, n=n, temperature=temperature, top_p=top_p,
                   max_new_tokens=max_new_tokens, seed=seed, lora_request=lora_request)
    return _token_report(tokenizer, prompt, raw)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: Sequence[str], *, n: int, temperature: float,
                   top_p: float, max_new_tokens: int, seed: int,
                   lora_request=None) -> list[list[str]]:
    # one vLLM call, same SamplingParams (seed included)
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
    # one report per prompt
    raws = generate_batch(model, tokenizer, prompts, n=n, temperature=temperature, top_p=top_p,
                          max_new_tokens=max_new_tokens, seed=seed, lora_request=lora_request)
    return [_token_report(tokenizer, p, raw) for p, raw in zip(prompts, raws)]


"""Answer log-probabilities without vLLM, for the backbones vLLM cannot score.

A plain transformers forward pass, not vLLM: vLLM 0.11.2's ``prompt_logprobs``
path crashes for Llama-3.1-8B and Mistral-7B under LoRA. Writes
``logprobs_hf.json``, which aggregation.py and estimators.py read.

    python analysis/logprobs_hf.py --model llama --method Base \
        --datasets hotpot trivia msmarco nq_open
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
CKPT = {
    "Base": None,
    "RLVR": "ckpt/RLVR/{m}/checkpoint-2500",
    "CSR": "ckpt/CSR/{m}/checkpoint-2500",
}
EXP = {
    "Base": "Base_{m}_{d}_base_nano_k16",
    "RLVR": "RLVR_{m}_{d}_step2500_nano_k16",
    "CSR": "CSR_{m}_{d}_z10_step2500_nano_k16",
}
# --model takes the config key; results/ and ckpt/ use the yaml's ckpt_name
from csr.config import MODELS


def load_hf_policy(cfg, adapter: str | None, load_in_4bit: bool):
    # 4-bit NF4 with double quant, as trained and evaluated, plus its LoRA
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = cfg["model"]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", quantization_config=quant)
    print(f"[load] {hf_id}  {'4-bit nf4' if load_in_4bit else 'bf16'}", flush=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[lora] {adapter}", flush=True)
    model.eval()
    return model, tokenizer


def score_batch(model, tokenizer, pairs, device="cuda:0"):
    # mean answer log-prob for a list of (prompt_text, answer_text)
    import torch
    import torch.nn.functional as F

    # Tokenized apart on purpose: joining the strings first can merge the boundary
    # into one token and shift every answer position by one.
    enc_full, n_prompt = [], []
    for prompt, answer in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        text = answer if str(answer).strip() else " "
        a_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        if not a_ids:
            a_ids = [tokenizer.eos_token_id]
        enc_full.append(p_ids + a_ids)
        n_prompt.append(len(p_ids))

    width = max(len(x) for x in enc_full)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    ids = torch.full((len(enc_full), width), pad, dtype=torch.long)
    mask = torch.zeros((len(enc_full), width), dtype=torch.long)
    for i, f in enumerate(enc_full):
        ids[i, : len(f)] = torch.tensor(f)
        mask[i, : len(f)] = 1
    ids, mask = ids.to(device), mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        logprobs = F.log_softmax(logits[:, :-1], dim=-1)
        picked = logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    out = []
    for i, f in enumerate(enc_full):
        # picked[j] is the log-prob of token j+1, so the answer starts at n_prompt-1
        lo, hi = n_prompt[i] - 1, len(f) - 1
        span = picked[i, lo:hi]
        out.append(float(span.mean()) if span.numel() else 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen", choices=MODELS)
    parser.add_argument("--method", default="CSR", choices=list(EXP))
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpot", "trivia", "msmarco", "nq_open"])
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions (smoke test)")
    parser.add_argument("--ckpt", default=None,
                        help="adapter directory; overrides the default under ckpt/")
    parser.add_argument("--batch-seqs", type=int, default=64,
                        help="answers scored per forward pass")
    args = parser.parse_args()

    from csr.config import load_config
    from csr.data import load_split
    from csr.generation import render_generation_prompt

    model = tokenizer = None
    for ds in args.datasets:
        cfg = load_config(args.method if args.method != "Base" else "Base",
                          args.model, ds, [])
        name = cfg["model"]["ckpt_name"]          # results/ and ckpt/ directory name
        exp = EXP[args.method].format(m=name, d=ds)
        run_dir = RESULTS / exp
        payload = json.loads((run_dir / "results.json").read_text())
        if payload.get("status") != "final":
            print(f"[skip] {exp}: not final"); continue
        records = payload["records"][: args.limit]

        if model is None:
            ckpt = args.ckpt or CKPT[args.method]
            adapter = (str(ROOT / ckpt.format(m=name, model=name))
                       if ckpt else None)
            use_4bit = bool(cfg["model"].get("load_in_4bit", True))
            model, tokenizer = load_hf_policy(cfg, adapter, use_4bit)

        dataset = load_split(cfg, split="eval", size=len(payload["records"]))
        flat, owners = [], []
        for qi, rec in enumerate(records):
            prompt = render_generation_prompt(dataset[int(rec["idx"])]["prompt"], tokenizer)
            for r in rec["rollouts"]:
                flat.append((prompt, r["full_answer"])); owners.append(qi)

        scores = []
        for start in range(0, len(flat), args.batch_seqs):
            scores += score_batch(model, tokenizer, flat[start:start + args.batch_seqs])
            if (start // args.batch_seqs) % 20 == 0:
                print(f"[{exp}] {len(scores)}/{len(flat)} answers", flush=True)

        out: dict[str, list[float]] = {}
        for score, qi in zip(scores, owners):
            out.setdefault(str(records[qi]["idx"]), []).append(score)

        default_name = "logprobs_smoke.json" if args.limit else "logprobs_hf.json"
        path = run_dir / default_name
        path.write_text(json.dumps(out))
        print(f"[saved] {path}  ({len(out)} questions)", flush=True)


if __name__ == "__main__":
    main()

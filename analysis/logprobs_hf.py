"""Answer log-probabilities without vLLM, for the backbones vLLM cannot score.

Same job, arithmetic and output as ``logprobs.py``, by a plain transformers forward
pass; it exists because vLLM 0.11.2's ``prompt_logprobs`` path crashes for
Llama-3.1-8B and Mistral-7B under LoRA (Qwen2.5-7B does not trigger it).
Writes ``logprobs.json`` by default, but aggregation.py and estimators.py read
``logprobs_hf.json`` -- pass --out-name logprobs_hf.json for the published table.

    # verify against a run that vLLM already scored
    python analysis/logprobs_hf.py --model qwen7b --method Base \
        --datasets hotpot --limit 50 --check

    # produce the missing files
    python analysis/logprobs_hf.py --model llama8b --method Base \
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
    "Base": "Base_{m}_{d}_base_oss_k16",
    "RLVR": "RLVR_{m}_{d}_step2500_oss_k16",
    "CSR": "CSR_{m}_{d}_z10_step2500_oss_k16",
}
MODELS = ["llama8b", "qwen7b", "mistral7b"]   # results/ and ckpt/ share these names


def load_hf_policy(cfg, adapter: str | None, load_in_4bit: bool,
                   quant_type: str = "nf4"):
    """The policy as trained and evaluated -- 4-bit NF4 with double quant -- plus its LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = cfg["model"]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=(quant_type == "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", quantization_config=quant)
    print(f"[load] {hf_id}  {'4-bit ' + quant_type if load_in_4bit else 'bf16'}", flush=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[lora] {adapter}", flush=True)
    model.eval()
    return model, tokenizer


def score_batch(model, tokenizer, pairs, device="cuda:0"):
    """Mean answer log-prob for a list of (prompt_text, answer_text)."""
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
    parser.add_argument("--model", default="qwen7b", choices=MODELS)
    parser.add_argument("--method", default="CSR", choices=list(EXP))
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpot", "trivia", "msmarco", "nq_open"])
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions (smoke test)")
    parser.add_argument("--ckpt", default=None,
                        help="adapter directory; overrides the default under ckpt/")
    parser.add_argument("--batch-seqs", type=int, default=64,
                        help="answers scored per forward pass")
    parser.add_argument("--out-name", default=None,
                        help="file name to write next to the run; defaults to "
                             "logprobs.json (logprobs_smoke.json with --limit)")
    # vLLM logs fp4 without double quant, so its numbers are not interchangeable
    parser.add_argument("--quant-type", default="nf4", choices=("nf4", "fp4"),
                        help="nf4 with double quant is what the reference "
                             "implementation uses; fp4 without it is what vLLM "
                             "logged for the runs already on disk")
    parser.add_argument("--no-4bit", action="store_true",
                        help="load in bf16 instead of the 4-bit quantization the "
                             "evaluation used; the numbers then do not match the "
                             "files vLLM already wrote")
    parser.add_argument("--check", action="store_true",
                        help="compare against the run's existing logprobs.json "
                             "instead of writing anything")
    args = parser.parse_args()

    from csr.config import load_config
    from csr.data import load_split
    from csr.text import render_generation_prompt

    model = tokenizer = None
    for ds in args.datasets:
        exp = EXP[args.method].format(m=args.model, d=ds)
        run_dir = RESULTS / exp
        payload = json.loads((run_dir / "results.json").read_text())
        if payload.get("status") != "final":
            print(f"[skip] {exp}: not final"); continue
        records = payload["records"][: args.limit]

        cfg = load_config(args.method if args.method != "Base" else "Base",
                          args.model, ds, [])
        if model is None:
            ckpt = args.ckpt or CKPT[args.method]
            adapter = (str(ROOT / ckpt.format(m=args.model,
                                              model=args.model)) if ckpt else None)
            use_4bit = (not args.no_4bit) and bool(cfg["model"].get("load_in_4bit", True))
            model, tokenizer = load_hf_policy(cfg, adapter, use_4bit, args.quant_type)

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

        if args.check:
            ref_path = run_dir / "logprobs.json"
            if not ref_path.exists():
                print(f"[check] {exp}: no logprobs.json to compare against"); continue
            ref = json.loads(ref_path.read_text())
            pairs = [(a, b) for k, v in out.items() if k in ref
                     for a, b in zip(v, ref[k])]
            # Absolute, not relative: a confident answer's mean log-prob is near zero.
            import numpy as np
            mine = np.array([a for a, _ in pairs])
            ref_v = np.array([b for _, b in pairs])
            diffs = np.abs(mine - ref_v)
            diffs_sorted = np.sort(diffs)

            def _rank(v):
                order = v.argsort()
                r = np.empty(len(v), dtype=float)
                r[order] = np.arange(len(v), dtype=float)
                return r

            pear = float(np.corrcoef(mine, ref_v)[0, 1]) if len(pairs) > 1 else float("nan")
            spear = (float(np.corrcoef(_rank(mine), _rank(ref_v))[0, 1])
                     if len(pairs) > 1 else float("nan"))
            within = float((diffs <= 0.05).mean())
            print(f"[check] {exp}: {len(pairs)} answers compared\n"
                  f"        mean |diff| {diffs.mean():.5f}   "
                  f"median {diffs_sorted[len(diffs)//2]:.5f}   "
                  f"p95 {diffs_sorted[int(0.95*len(diffs))]:.5f}   "
                  f"max {diffs.max():.5f}\n"
                  f"        pearson {pear:.5f}   spearman {spear:.5f}   "
                  f"within 0.05: {within:.1%}")
            continue

        default_name = "logprobs_smoke.json" if args.limit else "logprobs.json"
        path = run_dir / (args.out_name or default_name)
        path.write_text(json.dumps(out))
        print(f"[saved] {path}  ({len(out)} questions)", flush=True)


if __name__ == "__main__":
    main()

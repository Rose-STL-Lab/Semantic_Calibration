r"""Scores answers with a plain transformers forward pass.

Writes ``logprobs_hf.json`` per run, which aggregation.py and estimators.py read.

    python analysis/logprobs_hf.py --model llama --method Base \
        --datasets hotpot trivia msmarco nq_open
    python analysis/logprobs_hf.py --model qwen --method CSR \
        --ckpt ckpt/CSR/CSR_qwen_hotpot/checkpoint-2500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csr.config import MODELS  # noqa: E402

RESULTS = ROOT / "results"
EXP = {
    "Base": "Base_{m}_{d}_base_nano_k16",
    "RLVR": "RLVR_{m}_{d}_step2500_nano_k16",
    "CSR": "CSR_{m}_{d}_z10_step2500_nano_k16",
}


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

    out = []
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
        for i, f in enumerate(enc_full):
            # logits[j] predicts token j+1, so the answer starts at n_prompt-1;
            # only those rows go to fp32, never the full batch x vocab tensor
            lo, hi = n_prompt[i] - 1, len(f) - 1
            span = logits[i, :-1][lo:hi].float()
            if not span.shape[0]:
                out.append(0.0); continue
            target = ids[i, 1:][lo:hi].unsqueeze(-1)
            picked = span.gather(-1, target).squeeze(-1) - torch.logsumexp(span, dim=-1)
            out.append(float(picked.mean()))
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
                        help="adapter directory, e.g. ckpt/CSR/<exp-name>/checkpoint-2500; "
                             "required unless --method Base")
    parser.add_argument("--batch-seqs", type=int, default=16,
                        help="answers scored per forward pass")
    args = parser.parse_args()
    if args.method == "Base" and args.ckpt:
        parser.error("--method Base scores the plain model; drop --ckpt")
    if args.method != "Base" and not args.ckpt:
        parser.error(f"--method {args.method} needs --ckpt (the trained adapter directory)")

    from csr.config import load_config
    from csr.data import load_split
    from csr.generation import render_generation_prompt

    model = tokenizer = None
    for ds in args.datasets:
        cfg = load_config(args.method, args.model, ds, [])
        name = cfg["model"]["ckpt_name"]          # backbone tag in run names
        exp = EXP[args.method].format(m=name, d=ds)
        run_dir = RESULTS / exp
        results_path = run_dir / "results.json"
        if not results_path.exists():
            print(f"[skip] {exp}: no {results_path}", file=sys.stderr); continue
        payload = json.loads(results_path.read_text())
        if payload.get("status") != "final":
            print(f"[skip] {exp}: not final", file=sys.stderr); continue
        records = payload["records"][: args.limit]

        if model is None:
            adapter = os.path.abspath(args.ckpt) if args.ckpt else None
            use_4bit = bool(cfg["model"].get("load_in_4bit", True))
            model, tokenizer = load_hf_policy(cfg, adapter, use_4bit)

        dataset = load_split(cfg, split="eval", size=len(payload["records"]))
        tmpl_kwargs = cfg["model"].get("chat_template_kwargs") or {}
        flat, owners = [], []
        for qi, rec in enumerate(records):
            example = dataset[int(rec["idx"])]
            if example["question"] != rec["question"]:
                raise SystemExit(
                    f"{exp}: record idx {rec['idx']} has question {rec['question']!r} "
                    f"but the rebuilt eval split has {example['question']!r}; the "
                    f"dataset or its filtering changed since this run was evaluated")
            prompt = render_generation_prompt(example["prompt"], tokenizer, tmpl_kwargs)
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

"""Average token log-probability of each stored answer in an evaluation run.

Scored with vLLM ``prompt_logprobs=0`` -- the vLLM counterpart of ``logprobs_hf.py``.
Writes ``{"<record idx>": [lp_rollout_0, ..., lp_rollout_15], ...}`` next to the run
as ``logprobs.json``, or ``logprobs_smoke.json`` with --limit.

    GPU=<id> python analysis/logprobs.py --model qwen7b --method CSR --datasets hotpot
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


BATCH_SEQS = 512        # default sequences per vLLM call; --batch-seqs overrides
                        # (one question contributes K of them)


def answer_logprobs(model, tokenizer, items, lora_request,
                    batch_seqs: int = BATCH_SEQS) -> list[list[float]]:
    """Mean log-prob of each answer's tokens; ``items`` is a list of (prompt, answers)."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    flat, owners, prompt_lens = [], [], []
    for qi, (prompt, answers) in enumerate(items):
        n_prompt = len(tokenizer(prompt)["input_ids"])
        for a in answers:
            flat.append(prompt + a); owners.append(qi); prompt_lens.append(n_prompt)

    scores = [0.0] * len(flat)
    for start in range(0, len(flat), batch_seqs):
        chunk = flat[start:start + batch_seqs]
        outputs = model.fast_generate(chunk, sampling_params=params, lora_request=lora_request)
        for j, o in enumerate(outputs):
            # prompt_logprobs[i] is None for the first token (nothing to condition on)
            lps = [next(iter(d.values())).logprob
                   for d in o.prompt_logprobs[prompt_lens[start + j]:] if d]
            scores[start + j] = float(sum(lps) / len(lps)) if lps else 0.0

    out = [[] for _ in items]
    for score, qi in zip(scores, owners):
        out[qi].append(score)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen7b", choices=MODELS)
    parser.add_argument("--method", default="CSR", choices=list(EXP))
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "trivia", "msmarco", "nq_open"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--ckpt", default=None,
                        help="adapter directory; overrides the default under ckpt/")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--batch-seqs", type=int, default=BATCH_SEQS,
                        help="sequences per vLLM call; raise it on a card with "
                             "more memory to cut the number of round trips")
    args = parser.parse_args()

    from csr.config import load_config
    from csr.data import load_split
    from csr.modeling import load_policy
    from csr.text import render_generation_prompt

    lora_request = None
    model = tokenizer = None

    for ds in args.datasets:
        exp = EXP[args.method].format(m=args.model, d=ds)
        run_dir = RESULTS / exp
        payload = json.loads((run_dir / "results.json").read_text())
        if payload.get("status") != "final":
            print(f"[skip] {exp}: not final"); continue
        records = payload["records"][: args.limit]

        cfg = load_config(args.method if args.method != "Base" else "Base", args.model, ds, [])
        if model is None:
            model, tokenizer = load_policy(cfg, int(cfg["method"]["eval"]["max_seq_length"]),
                                           gpu_memory_utilization=args.gpu_memory_utilization)
            ckpt = args.ckpt or CKPT[args.method]
            if ckpt:
                path = ckpt.format(m=args.model, model=args.model)
                lora_request = model.load_lora(str(ROOT / path))
                print(f"[lora] {path}")

        dataset = load_split(cfg, split="eval", size=len(payload["records"]))
        items = [(render_generation_prompt(dataset[int(rec["idx"])]["prompt"], tokenizer),
                  [r["full_answer"] for r in rec["rollouts"]]) for rec in records]
        out = {}
        step = 64                                    # questions per progress line
        for start in range(0, len(items), step):
            block = answer_logprobs(model, tokenizer, items[start:start + step],
                                    lora_request, args.batch_seqs)
            for rec, scores in zip(records[start:start + step], block):
                out[str(rec["idx"])] = scores
            print(f"[{exp}] {len(out)}/{len(records)}", flush=True)
        path = run_dir / ("logprobs_smoke.json" if args.limit else "logprobs.json")
        path.write_text(json.dumps(out))
        print(f"[saved] {path}  ({len(out)} questions)")


if __name__ == "__main__":
    main()

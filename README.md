# Calibration via Semantic Reward (CSR)

This repository contains training and evaluation scripts for the CSR method
together with the four comparison points used in our experiments —
**Base** (frozen `meta-Llama-3.1-8B-Instruct`), **RLVR**, **RD**, **RLCR**,
**CSR** — on multi-hop and open-domain question answering. All trainable
methods fine-tune `meta-Llama-3.1-8B-Instruct` with LoRA (rank 32) under GRPO
using [unsloth](https://github.com/unslothai/unsloth) +
[TRL](https://github.com/huggingface/trl) and an LLM judge (Azure / OpenAI
chat-completions API) for correctness scoring. RD checkpoints are produced by
the upstream
[Rewarding Doubt](https://github.com/pasta99/RewardingDoubt) repository
unmodified.

## Layout

```
├── train_CSR.py            # CSR: GRPO with linear-ramp semantic calibration + correctness reward
├── train_RLVR.py           # RLVR baseline: GRPO with correctness reward only
├── train_RLCR.py           # RLCR baseline: ported to unsloth/GRPO
└── evaluation/
    ├── Base/               # Frozen base model, no LoRA
    │   └── eval_Base_{hotpot,msmarco,nq_open,trivia}.py
    ├── RLVR/               # Semantic confidence eval, loads RLVR LoRA
    │   └── eval_RLVR_{hotpot,msmarco,nq_open,trivia}.py
    ├── RD/                 # Verbalized confidence eval (RD format)
    │   └── eval_RD_{hotpot,msmarco,nq_open,trivia}.py
    ├── RLCR/               # Verbalized confidence eval (RLCR format)
    │   └── eval_RLCR_{hotpot,msmarco,nq_open,trivia}.py
    └── CSR/                # Semantic confidence eval, loads CSR LoRA
        └── eval_CSR_{hotpot,msmarco,nq_open,trivia}.py
```

`evaluation/Base/` runs the frozen base model with no LoRA loaded.
`evaluation/RLVR/` and `evaluation/CSR/` use the same semantic confidence
confidence proxy (`exp(-H)` over `K=8` semantic clusters) and only differ in
which LoRA checkpoint is loaded via `LORA_ROOT_PATH`.
`evaluation/RD/` and `evaluation/RLCR/` parse the verbalized confidence
emitted by their respective formats. Each file is self-contained: data
loading, reward functions, training/eval loops, and CSV/JSONL/plot output
are all in one script.

## Setup

### Dependencies

The full conda environment used for all experiments is exported to
[`environment.yml`](environment.yml). Recreate it with:

```bash
conda env create -f environment.yml
conda activate csr
```

### Environment variables

LoRA checkpoint paths are passed per-eval-script via `--lora-path` (see
[Evaluation](#evaluation) below). All other knobs (HF cache directory, judge
model id, retry / sleep / worker counts) are hardcoded inside the scripts.

Judge API key (required for `train_CSR.py`, `train_RLVR.py`, and every
`evaluation/*/eval_*.py` script):

```bash
# Azure OpenAI: set all three
export OPENAI_API_KEY=...
export OPENAI_API_VERSION=2024-08-01-preview
export OPENAI_AZURE_ENDPOINT=https://<your-endpoint>.openai.azure.com

# Plain OpenAI: only OPENAI_API_KEY is required.
export OPENAI_API_KEY=...
```

The judge model is hardcoded to `gpt-4.1-nano`. Edit `JUDGE_MODEL_ID` in
each script (or the `model_name` constant in the RD/RLCR `api_yesno_judge`
helpers) to use a different model. The HF cache directory is hardcoded to
`./hf_cache`.

## Training

All training scripts assume a single GPU (≥40 GB recommended for 8B + 4-bit +
LoRA r=32) and write checkpoints to `outputs_<EXPERIMENT>/`.

```bash
# CSR: default cali-w end = 0.2
python train_CSR.py --cali-w 0.2

# RLVR baseline
python train_RLVR.py

# RLCR baseline
python train_RLCR.py
```

Common training settings (GRPO):

| Setting | Value |
| --- | --- |
| Train size | 10 000 |
| Epochs | 1 |
| Per-device batch / grad-accum | 1 / 4 |
| `num_generations` | 8 |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.99 |
| Optimizer | `paged_adamw_8bit` |
| Seed | 42 |

(See the `GRPOConfig` block in each script for the full hyperparameter list.)

## Evaluation

Each `evaluation/<METHOD>/eval_<METHOD>_<dataset>.py` script evaluates the corresponding
method on **1 000 validation examples per dataset**, then computes accuracy,
ECE, Brier, Spearman ρ, Kendall τ, and AUROC of confidence vs. correctness.
Each writes:

* `<dataset>_eval_<METHOD>.csv` — one row per evaluated model
* `<dataset>_<ckpt>.jsonl` — per-example predictions and judge labels

The five method folders use two different confidence proxies, but a unified
checkpoint interface:

* **Base / RLVR / CSR** (semantic confidence proxy): confidence is estimated
  from the semantic-cluster distribution of `K=8` rollouts.
* **RD / RLCR** (verbalized confidence): confidence is parsed from the
  model's output (RD: `Confidence: <int>`; RLCR: `<confidence>...</confidence>`).

For `RLVR / RD / RLCR / CSR`, the LoRA checkpoint is passed via the
`--lora-path` CLI argument. **If `--lora-path` is omitted, the script
evaluates the base model (no LoRA loaded).** `Base/` is hard-wired to
base-only and accepts no LoRA argument.

```bash
# Base — frozen model
python evaluation/Base/eval_Base_hotpot.py

# RLVR / RD / RLCR / CSR — pass the LoRA checkpoint dir with --lora-path
python evaluation/RLVR/eval_RLVR_hotpot.py   --lora-path outputs_RLVR/checkpoint-2500
python evaluation/RD/eval_RD_hotpot.py       --lora-path /path/to/rd/checkpoint
python evaluation/RLCR/eval_RLCR_hotpot.py   --lora-path /path/to/rlcr/checkpoint
python evaluation/CSR/eval_CSR_hotpot.py     --lora-path outputs_CSR_0.2/checkpoint-2500

# Same scripts run without --lora-path fall back to base eval.
python evaluation/RLVR/eval_RLVR_hotpot.py    # → BASE row in CSV
```

Substitute `_hotpot.py` with `_msmarco.py`, `_nq_open.py`, or `_trivia.py`
for the other datasets.

## Datasets

All datasets are pulled via the HuggingFace `datasets` API; no manual
download is needed once `HF_CACHE` is set.

| Script | Dataset |
| --- | --- |
| Training scripts and `evaluation/*/eval_*_hotpot.py` | `hotpot_qa` (`distractor`) |
| `evaluation/*/eval_*_msmarco.py` | `microsoft/ms_marco` (`v1.1`) |
| `evaluation/*/eval_*_nq_open.py` | `google-research-datasets/nq_open` |
| `evaluation/*/eval_*_trivia.py` | `mandarjoshi/trivia_qa` (`rc.web.nocontext`) |

## Reproducibility notes

* Seed is fixed to `42` everywhere (`random`, `numpy`, `torch`, dataset
  shuffling, GRPO).
* The judge is called with `temperature=0`, `max_tokens=16`, and
  `Correct: YES/NO` parsing; an LRU cache (≤200 K entries) deduplicates calls
  across rollouts and checkpoints.
* `train_RLCR.py` is an unsloth/GRPO port of the published RLCR recipe;
  reward functions, prompt (`tabc_long`), and reward weights match upstream.
  Single-GPU adaptations: per-device batch 1 with `gradient_accumulation_steps=4`,
  `paged_adamw_8bit`, full attention + MLP LoRA targets,
  `max_seq_length=4096`, `max_completion_length=1024`.

## Upstream code

* RLCR baseline (`train_RLCR.py`) — ported from
  [github.com/damanimehul/RLCR](https://github.com/damanimehul/RLCR).
* RD baseline — trained with the upstream
  [github.com/pasta99/RewardingDoubt](https://github.com/pasta99/RewardingDoubt)
  repository unmodified
  ([arXiv:2503.02623](https://arxiv.org/abs/2503.02623)). Evaluation of RD
  checkpoints uses the verbalized-confidence parser in
  [`evaluation/RD/`](evaluation/RD/).

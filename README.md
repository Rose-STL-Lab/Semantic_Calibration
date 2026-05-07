# Calibration via Semantic Reward (CSR)

This repository contains training and evaluation scripts for the CSR method on multi-hop and open-domain question
answering. All experiments fine-tune `meta-Llama-3.1-8B-Instruct` with LoRA
(rank 32) under GRPO using [unsloth](https://github.com/unslothai/unsloth) +
[TRL](https://github.com/huggingface/trl) and an LLM judge (Azure / OpenAI
chat-completions API) for correctness scoring.

## Layout

```
├── train_CSR.py            # CSR: GRPO with linear-ramp semantic calibration + correctness reward
├── train_RLVR.py           # Baseline: GRPO with correctness reward only
├── train_RLCR.py           # Baseline: RLCR ported to unsloth/GRPO
├── eval_RLSC_hotpot.py     # Self-consistency eval on HotpotQA (distractor)
├── eval_RLSC_msmarco.py    # Self-consistency eval on MS MARCO v1.1
├── eval_RLSC_nq_open.py    # Self-consistency eval on Natural Questions Open
└── eval_RLSC_trivia.py     # Self-consistency eval on TriviaQA (rc.web.nocontext)
```

Each file is self-contained: data loading, reward functions, training/eval
loops, and CSV/JSONL/plot output are all in one script.

## Methods at a glance

| Script | Reward signal |
| --- | --- |
| `train_CSR.py` | `cali_w(t) * calibration + 1.0 * correctness`, with `cali_w(t)` linearly ramped from `0.1` to `--cali-w` (default `0.2`) over training |
| `train_RLVR.py` | Correctness only |
| `train_RLCR.py` | RLCR's five rewards (`format`, `accuracy`, `brier`, `mean_confidence`, `confidence_one_or_zero`) with weights `[0.5, 0.5, 0.5, 1e-6, 1e-6]` |

`train_CSR.py` and `train_RLVR.py` use a GPT judge (one-line `Correct: YES/NO`)
to score correctness. `train_RLCR.py` uses exact-match against gold and parses
the `<answer>` / `<confidence>` tags emitted by the model.

## Setup

### Dependencies

The full conda environment used for all experiments is exported to
[`environment.yml`](environment.yml). Recreate it with:

```bash
conda env create -f environment.yml
conda activate unsloth
```

### Environment variables

Caches and paths (all optional; defaults shown):

```bash
export HF_CACHE=./hf_cache                    # HF Hub / datasets / transformers cache
export LORA_ROOT_PATH=./outputs/checkpoint-2500   # checkpoint to evaluate (eval scripts only)
```

Judge API (required for `train_CSR.py`, `train_RLVR.py`, all `eval_RLSC_*.py`):

```bash
# Option A: Azure OpenAI
export OPENAI_API_KEY=...
export OPENAI_API_VERSION=2024-08-01-preview
export OPENAI_AZURE_ENDPOINT=https://<your-endpoint>.openai.azure.com
export OPENAI_DEPLOYMENT_NAME=gpt-4.1-nano

# Option B: OpenAI / OpenAI-compatible
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional, defaults to OpenAI
export OPENAI_MODEL=gpt-4.1-nano
```

Optional judge tuning:

```bash
export JUDGE_API_MAX_RETRIES=2
export JUDGE_API_SLEEP_SEC=1.5
export JUDGE_API_MAX_WORKERS=6
```

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

The four `eval_RLSC_*.py` scripts evaluate a checkpoint with self-consistency
sampling (`num_rollout = 8`), then compute accuracy, ECE, Brier, Spearman ρ,
Kendall τ, and AUROC of confidence vs. correctness. Each writes:

* `<dataset>_eval_RLSC.csv` — one row per checkpoint
* `<dataset>_<ckpt>.jsonl` — per-example predictions and judge labels
* `figures_RLSC/<dataset>_RLSC_reliability_<ckpt>_<n>.png` — reliability diagram

Point `LORA_ROOT_PATH` at either a single checkpoint directory (e.g.
`outputs_CSR_0.2/checkpoint-2500`) or a parent directory; in the latter case
the script iterates over the names listed in `CHECKPOINT_NAMES` near the bottom
of each file. The base model (no LoRA) is always evaluated first, so every run
produces one `BASE` row in the CSV alongside the checkpoint rows.

```bash
LORA_ROOT_PATH=outputs_CSR_0.2/checkpoint-2500 python eval_RLSC_hotpot.py
LORA_ROOT_PATH=outputs_CSR_0.2/checkpoint-2500 python eval_RLSC_nq_open.py
LORA_ROOT_PATH=outputs_CSR_0.2/checkpoint-2500 python eval_RLSC_trivia.py
LORA_ROOT_PATH=outputs_CSR_0.2/checkpoint-2500 python eval_RLSC_msmarco.py
```

## Datasets

All datasets are pulled via the HuggingFace `datasets` API; no manual
download is needed once `HF_CACHE` is set.

| Script | Dataset |
| --- | --- |
| `train_CSR.py`, `train_RLVR.py`, `eval_RLSC_hotpot.py` | `hotpot_qa` (`distractor`) |
| `train_RLCR.py` | `hotpot_qa` (`distractor`) |
| `eval_RLSC_msmarco.py` | `microsoft/ms_marco` (`v1.1`) |
| `eval_RLSC_nq_open.py` | `google-research-datasets/nq_open` |
| `eval_RLSC_trivia.py` | `mandarjoshi/trivia_qa` (`rc.web.nocontext`) |

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
* Rewarding Doubt (RD) reference implementation —
  [github.com/pasta99/RewardingDoubt](https://github.com/pasta99/RewardingDoubt)
  ([arXiv:2503.02623](https://arxiv.org/abs/2503.02623)).

<h1 align="center">Calibrating LLMs with Semantic-level Reward</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2605.15588"> <img src="https://img.shields.io/badge/arXiv-2605.15588-b31b1b" alt="arXiv"> </a>
    <img src="https://img.shields.io/badge/NeurIPS'26-blue" alt="NeurIPS 2026">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="python">
    <img src="https://img.shields.io/badge/PyTorch-2.9-ee4c2c" alt="pytorch">
</p>

Hi, this is the code for our NeurIPS 2026 paper:
*[Calibrating LLMs with Semantic-level Reward](https://arxiv.org/abs/2605.15588)*.

![Overview of CSR](assets/CSR_overview.png)

**Overview of CSR.** For each question $x$ we draw $K$ rollouts from the policy
$\pi_\theta$, score each rollout with a verifiable correctness reward
$r_{\mathrm{RLVR}}$ and a semantic calibration reward $r_{\mathrm{Calibration}}$, and
update the policy via group-relative advantages. CSR concentrates correct rollouts into
a tight semantic cluster while keeping incorrect rollouts dispersed, so that semantic
agreement among sampled outputs becomes an informative signal of correctness.

Confidence is read off the rollouts as `c(x) = exp(-SE(x))`, the exponential of the
semantic entropy, so the model never emits a confidence token. Both reward terms live
on `[0, 1]`, which makes their weights directly comparable.

This repository holds the training code, the evaluation code, and the post-hoc
estimators reported in the appendix.

| Path | What it is |
| --- | --- |
| `csr/` | The library: rewards, semantic clustering, judge client, config loader, metrics |
| `train/` | One trainer per method — `train_CSR.py`, `train_RLVR.py`, `train_RLCR.py`, `train_RD.py` |
| `evaluation/` | `eval_semantic.py` (sampling-based) and `eval_verbalized.py` (verbalized-confidence baselines) |
| `configs/` | YAML defaults, split into `methods/`, `models/`, `datasets/`, plus `judge.yaml` |
| `analysis/` | Post-hoc estimators and aggregation rules reported in the appendix |
| `scripts/` | `train.sh`, `eval.sh`, `serve/serve_judge.sh`, and result collection |
| `rd_util/` | Utilities for the Rewarding-Doubt baseline, kept close to upstream |

# Getting started

## 1. Setup Environment

We conducted all experiments on NVIDIA A100 80GB PCIe GPUs. Each training or
evaluation run uses a single GPU. The software environment is Python 3.11.14,
PyTorch 2.9.0 and CUDA 12.8.

To create the conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yml
conda activate csr
```

The Rewarding-Doubt baseline pins an older stack (Python 3.10.20, PyTorch 2.5.1,
CUDA 12.4), so it gets a second environment. Activate it only for `train_RD.py`:

```bash
conda env create -f environment-rd.yml
```

## 2. Setup the Judge

CSR's reward and every sampling-based evaluation ask an LLM judge two kinds of
question: *is this answer correct against the gold answer*, and *are these two answers
semantically equivalent*. **A judge has to be reachable before any run starts.**

The default is GPT-4.1-nano on the hosted OpenAI endpoint, so all it needs is a key:

```bash
export OPENAI_API_KEY=sk-...
```

Nothing else: `configs/judge.yaml` already holds the model, the temperature and the
retry policy.

To run a local judge instead — gpt-oss-20b, or any other model vLLM can serve — start a
server and point the config at it:

```bash
export OPENAI_API_KEY=EMPTY     # a local server ignores it, the trainer still requires it
bash scripts/serve/serve_judge.sh 0 8310 openai/gpt-oss-20b
#                                 │   │    └ model
#                                 │   └ port
#                                 └ GPU
```

then append to any command:

```bash
--set judge.base_url=http://127.0.0.1:8310/v1 --set judge.model_id=gpt-oss-20b \
--set judge.reasoning_effort=low --set judge.max_tokens=1024
```

`reasoning_effort` and the larger `max_tokens` are what gpt-oss-20b needs to reason
before answering; a non-reasoning model rejects the first and does not need the second.

Token F1 can stand in for the judge entirely (`--set judge.backend=f1`), which removes
the dependency at some cost in reward quality.

## 3. Prepare Datasets

Datasets are pulled from the HuggingFace Hub on first use and cached under `data/`; no
manual download is needed. Checkpoints, evaluation records and logs are written beside
it, and none of these directories are tracked:

```
.
├── data            # HuggingFace datasets and hub cache
├── ckpt            # ckpt/<METHOD>/<run>/checkpoint-<step>
├── results         # results/<run>/{config.yaml,results.json,records.jsonl}
└── logs
```

| | Available |
| --- | --- |
| Methods | `Base` `CSR` `RLVR` `RLCR` `RD` |
| Models | `qwen` `llama` `mistral` `qwen14b` |
| Datasets | `hotpot` `hotpot_context` `trivia` `msmarco` `nq_open` |

## 4. Train

`scripts/train.sh` takes the method, the model and the GPU index.

* CSR

```bash
bash scripts/train.sh CSR qwen 0
```

* RLVR

```bash
bash scripts/train.sh RLVR llama 1
```

* RLCR

```bash
bash scripts/train.sh RLCR mistral 2
```

* RD — needs the `csr-rd` environment

```bash
conda activate csr-rd
bash scripts/train.sh RD qwen 3
```

The two settings that matter most for CSR are the calibration weight and the weight on
binary correctness:

* **`cali_w_start` / `cali_w_end`** — equal values give a constant weight; different
  values are interpolated by `cali_w_schedule` (`linear` or `sigmoid`, min-max rescaled
  so it hits both endpoints exactly).
* **`corr_w`** — 1.0 throughout the paper.

```bash
bash scripts/train.sh CSR qwen 0 \
  --set method.train.cali_w_start=0.8 \
  --set method.train.cali_w_end=1.2 \
  --set method.train.cali_w_schedule=sigmoid
```

Higher is not better: `configs/methods/csr.yaml` records where the policy fragments and
where it collapses onto a single answer.

## 5. Evaluate

* The frozen model

```bash
bash scripts/eval.sh Base qwen
```

* A trained checkpoint

```bash
bash scripts/eval.sh CSR qwen ckpt/CSR/<run>/checkpoint-2500
```

* A subset of datasets, on a chosen GPU

```bash
GPU=6 DATASETS="hotpot trivia" bash scripts/eval.sh CSR qwen ckpt/CSR/<run>/checkpoint-2500
```

* Every finished run as one table

```bash
python scripts/summarize.py
```

## 6. Configuration

Config is layered, each file overriding the one before: `default.yaml`, `judge.yaml`,
the model, the method, then the dataset. The dataset comes last so it can state its own
requirements — the context-grounded split needs a larger window than the answer-only
methods default to. Anything on the command line wins over all of them.

**Context-grounded runs** (`hotpot_context`) therefore need no extra flags: the
distractor paragraphs put the prompt at roughly 1 500 tokens, which does not fit the
1 024-token window the answer-only methods use, so
`configs/datasets/hotpot_context.yaml` raises the sequence lengths and lowers vLLM's
memory share itself.

Three things worth knowing before changing anything:

* **Rollout budget.** `num_generations` controls training; `method.eval.num_rollouts`
  controls evaluation. The paper trains at K=8 and evaluates at K=16.
* **Judge calls.** Clustering asks one symmetric equivalence question per unordered
  pair, so K(K−1)/2 calls per question. At K=16 that is 120 calls, which dominates
  evaluation time.
* **Rollouts per optimizer step.** `generation_batch_size` counts completions, not
  prompts, so prompts per step is `per_device_train_batch_size * grad_accum /
  num_generations` — 4 at the defaults, hence 2500 steps for one epoch over 10k prompts.

## 7. Appendix Experiments

Each script reads finished runs from `results/` and prints the corresponding table.
They are post-hoc: no estimator here is ever used during training.

| Script | Reports |
| --- | --- |
| `analysis/selective.py` | Selective prediction (AURAC, AUGRC) and inference cost |
| `analysis/estimators.py` | Semantic Density and SIMBA-UQ's Bayesian aggregation |
| `analysis/aggregation.py` | SIMBA-UQ's arithmetic aggregation and `clf-pairs+gen` classifier |
| `analysis/simba.py` | The same classifier under SIMBA-UQ's own training protocol |
| `analysis/logprobs_hf.py` | Per-answer log-probabilities for the `+gen` features |
| `scripts/f1_judge_eval.py` | The token-F1 judge variant |
| `scripts/rejudge.py` | Re-scoring a finished run with a different judge |

Each file's docstring states what it reproduces and where it departs from the method it
is named after. These scripts locate runs by name; `analysis/runs.py` holds the naming
scheme they expect, so point that module at your own runs first.

---

# References

If you find this code helpful in your research, please consider citing

```bibtex
@inproceedings{csr2026calibrating,
title={Calibrating LLMs with Semantic-level Reward},
author={Yu, Fengfei and Niu, Ruijia and Wu, Dongxia and Ma, Yian and Yu, Rose},
booktitle={NeurIPS},
year={2026}
}
```

# Implementation Credits

Some of our implementations rely on other repos. We want to thank the authors
([RLCR](https://github.com/damanimehul/RLCR),
[Rewarding Doubt](https://github.com/pasta99/RewardingDoubt),
[unsloth](https://github.com/unslothai/unsloth),
[TRL](https://github.com/huggingface/trl) and
[vLLM](https://github.com/vllm-project/vllm)) for making their code publicly
available.😄

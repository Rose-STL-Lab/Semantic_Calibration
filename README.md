<h1 align="center">Calibrating LLMs with Semantic-level Reward</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2605.15588"> <img src="https://img.shields.io/badge/arXiv-2605.15588-b31b1b" alt="arXiv"> </a>
    <img src="https://img.shields.io/badge/NeurIPS'26-blue" alt="NeurIPS 2026">
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
| `scripts/` | `serve/serve_judge.sh` (local judge) and `f1_judge_eval.py` (token-F1 clustering) |
| `rd_util/` | Utilities for the Rewarding-Doubt baseline, kept close to upstream |

# Getting started

## 1. Setup Environment

We conducted all experiments on NVIDIA A100 80GB PCIe GPUs. Each training or
evaluation run uses a single GPU. The software environment is Python 3.11.14,
PyTorch 2.9.0 and CUDA 12.8.

To install the dependencies, run:

```bash
conda create -n csr python=3.11 && conda activate csr
pip install torch==2.9.0            # first, and matching your CUDA
pip install -r requirements.txt
```

`requirements.txt` pins every direct dependency at the version the reported runs
used. Install `torch` by itself first: the stack is version-sensitive, and letting
pip resolve it alongside the rest can silently substitute a different build.

The Rewarding-Doubt baseline pins an older stack (Python 3.10.20, PyTorch 2.5.1,
CUDA 12.4), so it gets a second environment. Use it only for `train_RD.py`:

```bash
conda create -n csr-rd python=3.10 && conda activate csr-rd
pip install torch==2.5.1
pip install -r requirements-rd.txt
```

## 2. Setup the Judge

CSR's reward and every sampling-based evaluation ask an LLM judge two kinds of
question: *is this answer correct against the gold answer*, and *are these two answers
semantically equivalent*. **A judge has to be reachable before any run starts.**

The default is GPT-4.1-nano on the hosted OpenAI endpoint, so all it needs is a key:

```bash
export OPENAI_API_KEY=sk-...
```

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

Token F1 can stand in for the rollout-vs-rollout equivalence call
(`--set judge.equivalence=f1`), which cuts the judge's cost by the 88% of calls that are
pairwise; correctness still goes to the model. That is the CSR-F1 configuration the paper
reports, with `judge.f1_threshold` 0.50 for Llama and Mistral and 0.80 for Qwen.

## 3. Prepare Datasets

Datasets are pulled from the HuggingFace Hub on first use; no manual download is
needed. They land in the HuggingFace cache (`HF_HOME`, `~/.cache/huggingface` by
default). Checkpoints and evaluation records are written inside the repository, and
neither directory is tracked:

```
.
├── ckpt            # ckpt/<METHOD>/<run>/checkpoint-<step>
└── results         # results/<run>/{config.yaml,results.json,results.csv}
```

| | Available |
| --- | --- |
| Methods | `Base` `CSR` `RLVR` `RLCR` `RD` |
| Models | `llama` `qwen` `mistral` `qwen14b` `qwen32b` |
| Datasets | `hotpot` `hotpot_context` `trivia` `msmarco` `nq_open` `musique` `2wiki_comp` |

## 4. Train

One entry point per method; the environment is yours to set.

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=EMPTY           # a local vLLM judge ignores it, the trainer requires it
export HF_HOME=/path/to/hf/cache      # optional; otherwise ~/.cache/huggingface

python train/train_CSR.py --model qwen --dataset hotpot [--set key=value ...]
```

`train_RLVR.py`, `train_RLCR.py` and `train_RD.py` take the same arguments. RD pins the
older stack, so activate the environment built from `requirements-rd.txt` for it and the
unsloth one for the other three.

The RD defaults in `configs/methods/rd.yaml` follow the upstream Rewarding-Doubt
release, with two departures: LoRA rank 32 instead of 8, and an F1@0.5 correctness
reward instead of exact match on the free-form datasets. The RD rows in the paper also
turn on these flags:

```bash
python train/train_RD.py --model qwen --dataset hotpot \
  --set method.train.confidence_stop=true --set method.train.strict_format=true \
  --set method.train.learning_rate=1e-5 --set method.train.batch_size=8 \
  --set method.train.ppo_temperature=1.0 --set method.train.ppo_repetition_penalty=1.0
```

Two memory settings keep the same update at lower activation memory
(`ppo_mini_batch_size` x `ppo_gradient_accumulation_steps` samples per optimizer step):
the 32B run uses `2` x `2` and `hotpot_context` uses `1` x `4` together with
`method.train.unpadded_generation=true`, which generates one prompt at a time (unsloth's
batched inference is unreliable on prompts of one to two thousand tokens). The Qwen3
backbones (`qwen14b`, `qwen32b`) need `transformers>=4.51.3` and `unsloth>=2025.4.7` in
the RD environment in place of the versions pinned in `requirements-rd.txt`.

The two settings that matter most for CSR are the calibration weight and the weight on
binary correctness:

* **`cali_w_start` / `cali_w_end`** — equal values give a constant weight; different
  values are interpolated by `cali_w_schedule` (`linear` or `sigmoid`, min-max rescaled
  so it hits both endpoints exactly).
* **`corr_w`** — 1.0 throughout the paper.

The defaults ramp 0.8 → 1.2 on a sigmoid; to hold the weight constant instead:

```bash
python train/train_CSR.py --model qwen --dataset hotpot \
  --set method.train.cali_w_start=1.0 \
  --set method.train.cali_w_end=1.0
```

Higher is not better: `configs/methods/csr.yaml` records where the policy fragments and
where it collapses onto a single answer.

## 5. Evaluate

Two entry points, picked by method: `evaluation/eval_semantic.py` reads confidence off
the rollout spread for **Base, RLVR and CSR**, and `evaluation/eval_verbalized.py` parses
the confidence token for **RD and RLCR**.

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=...

python evaluation/eval_semantic.py --method Base --model qwen --dataset hotpot
python evaluation/eval_semantic.py --method CSR --model qwen --dataset hotpot \
  --ckpt ckpt/CSR/<run>/checkpoint-2500
```

Loop over the four datasets in your own shell to fill a table row:

```bash
for ds in hotpot trivia msmarco nq_open; do
  python evaluation/eval_semantic.py --method CSR --model qwen --dataset "$ds" --ckpt "$CKPT"
done
```

Each run writes `results/<run>/{config.yaml,results.json,results.csv}`. Pass `--exp-name`
to keep two variants of the same checkpoint from sharing a directory.

**Confidence from token F1 instead of the judge.** Re-clusters finished runs with token
overlap in place of the pairwise equivalence calls, at the thresholds the paper reports
(0.50 Llama/Mistral, 0.80 Qwen), writing each result to a separate `*_semf1_k16` run:

```bash
python scripts/f1_judge_eval.py              # the thresholds the paper reports
```

**Post-hoc estimators on a finished run.** These re-score `results/` and need no GPU;
no estimator here is ever used during training:

| Script | Reports |
| --- | --- |
| `analysis/estimators.py` | Semantic Density and SIMBA-UQ's Bayesian aggregation |
| `analysis/aggregation.py` | SIMBA-UQ's arithmetic aggregation and the `clf-pairs+gen` classifier |
| `analysis/selective.py` | Selective prediction (AURAC, AUGRC) |

Semantic Density and the `clf-pairs+gen` classifier read one extra per-rollout feature,
the mean token log-probability. `analysis/logprobs_hf.py` writes it once per run and is the only script
here that loads the backbone, so it is the one that needs a GPU:

```bash
python analysis/logprobs_hf.py --model qwen --method CSR --datasets hotpot trivia
```

Without that file SemDens falls back to uniform weights and `clf-pairs+gen` scores
similarities only, which `--no-logprobs` also forces for a like-for-like comparison
across backbones.

Each file's docstring states what it reproduces and where it departs from the method it
is named after. They locate runs by name; `analysis/runs.py` holds the naming scheme
they expect, so point that module at your own runs first.

## 6. Configuration

Config is layered, each file overriding the one before: `default.yaml`, `judge.yaml`,
the model, the method, then the dataset. The dataset comes last so it can state its own
requirements. Anything on the command line wins over all of them.

Three things worth knowing before changing anything:

* **Rollout budget.** `num_generations` controls training; `method.eval.num_rollouts`
  controls evaluation. The paper trains at K=8 and evaluates at K=16.
* **Judge calls.** Clustering asks one symmetric equivalence question per unordered
  pair, so K(K−1)/2 calls per question. At K=16 that is 120 calls, which dominates
  evaluation time.
* **Rollouts per optimizer step.** TRL counts completions, not prompts:
  `per_device_train_batch_size * gradient_accumulation_steps` is the completion
  budget, so prompts per step is that over `num_generations` — `8 * 4 / 8` = 4 at
  the defaults, hence 2500 steps for one epoch over 10k prompts.

The shipped defaults are `beta` 0.01, a sigmoid calibration weight from `cali_w`
0.8 to 1.2, training K=8 and evaluation K=16. Earlier preprint versions reported
`beta` 0.1 and a raw, un-normalized BCE with a 0.1 → 0.2 weight range; the reward
here is the affine-normalized form (`csr/rewards.py`, scale `DELTA_CE` = 13.8155),
under which a raw weight `w` corresponds to `w * 13.8155`, so that 0.1 → 0.2 range
is 1.3816 → 2.7631 in these units.

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

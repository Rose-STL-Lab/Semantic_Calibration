<h1 align="center">Calibrating LLMs with Semantic-level Reward</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2605.15588"> <img src="https://img.shields.io/badge/arXiv-2605.15588-b31b1b" alt="arXiv"> </a>
    <img src="https://img.shields.io/badge/NeurIPS'26-blue" alt="NeurIPS 2026">
</p>

This repository contains the code for our NeurIPS 2026 paper
*[Calibrating LLMs with Semantic-level Reward](https://arxiv.org/abs/2605.15588)*.

![Overview of CSR](assets/CSR_overview.png)

**Overview of CSR.** For each question $x$ we draw $K$ rollouts from the policy
$\pi_\theta$, score each rollout with a verifiable correctness reward
$r_{\mathrm{RLVR}}$ and a semantic calibration reward $r_{\mathrm{Calibration}}$, and
update the policy via group-relative advantages. CSR concentrates correct rollouts into
a tight semantic cluster while keeping incorrect rollouts dispersed, so that semantic
agreement among rollouts becomes an informative signal of correctness.

The confidence is $c(x) = \exp(-\mathrm{SE}(x))$, where $\mathrm{SE}$ is the semantic
entropy of the rollouts. The model never states a confidence itself. Both reward terms
lie in $[0, 1]$, so their weights are directly comparable.

| Path | What it is |
| --- | --- |
| `csr/` | The library: rewards, semantic clustering, judge client, config loader, metrics |
| `train/` | One trainer per trained method: `train_CSR.py`, `train_RLVR.py`, `train_RLCR.py`, `train_RD.py` |
| `evaluation/` | `eval_semantic.py` (Base, CSR, RLVR) and `eval_verbalized.py` (RLCR, RD) |
| `configs/` | YAML settings: `default.yaml`, `judge.yaml`, `models/`, `methods/`, `datasets/` |
| `analysis/` | Post-hoc estimators, aggregation rules and selective prediction from the appendix |
| `scripts/` | `serve/serve_judge.sh` (local judge server) and `f1_judge_eval.py` (rebuilds the paper's token-F1 tables) |
| `rd_util/` | Utilities for the Rewarding Doubt baseline, kept close to upstream |

# Getting Started

## 1. Environment

We ran all experiments on NVIDIA A100 80GB PCIe GPUs, one GPU per training or evaluation
run, with Python 3.11.14, PyTorch 2.9.0 and CUDA 12.8. Install `torch` before the other
packages so pip does not replace it with a different build:

```bash
conda create -n csr python=3.11 && conda activate csr
pip install torch==2.9.0            # pick the build for your CUDA
pip install -r requirements.txt
```

The Rewarding Doubt (RD) baseline needs Python 3.10.20, PyTorch 2.5.1 and CUDA 12.4.
RD training therefore runs in a second environment, `csr-rd`. Everything else, including
RD evaluation, runs in `csr`.

```bash
conda create -n csr-rd python=3.10 && conda activate csr-rd
pip install torch==2.5.1
pip install -r requirements-rd.txt
pip install "transformers>=4.51.3" "unsloth>=2025.4.7"   # only for qwen14b and qwen32b
```

## 2. Judge

An LLM judge answers two questions: whether an answer matches the gold answer, and
whether two answers mean the same thing. CSR and RLVR training, the periodic validation
in RLCR training, and every evaluation call the judge. RD training does not.

The default judge is GPT-4.1-nano on the OpenAI API. To use it, export your key:

```bash
export OPENAI_API_KEY=sk-...
```

To use a local judge instead, serve any model that vLLM supports. The paper uses
gpt-oss-20b. Start the server on its own GPU:

```bash
# bash scripts/serve/serve_judge.sh <GPU> <PORT> <HuggingFace model>
bash scripts/serve/serve_judge.sh 1 8310 openai/gpt-oss-20b
```

Training and evaluation then run on another GPU, for example `export CUDA_VISIBLE_DEVICES=0`.
A local server needs no API key. Store the judge settings in a shell variable, with
`judge.model_id` set to the last part of the model name:

```bash
JUDGE="--set judge.base_url=http://127.0.0.1:8310/v1 --set judge.model_id=gpt-oss-20b \
       --set judge.reasoning_effort=low --set judge.max_tokens=1024"
```

`reasoning_effort` and `max_tokens` are for reasoning models such as gpt-oss-20b. For
other models, keep only `base_url` and `model_id`. Every command below that calls the
judge includes `$JUDGE`. With the OpenAI API, leave `JUDGE` unset and it expands to
nothing.

To compare two answers by token F1 instead of asking the judge, add the flags below
after `$JUDGE` in CSR training and in `evaluation/eval_semantic.py`. Correctness is
still judged by the LLM. The paper uses a threshold of 0.50 for `llama` and `mistral`
and 0.80 for `qwen`.

```bash
--set judge.equivalence=f1 --set judge.f1_threshold=0.80
```

## 3. Datasets and Configs

Datasets download from the HuggingFace Hub on first use and are cached under `HF_HOME`
(default `~/.cache/huggingface`).

| Option | Values |
| --- | --- |
| Methods | `Base` `CSR` `RLVR` `RLCR` `RD` |
| Models | `llama` `qwen` `mistral` `qwen14b` `qwen32b` |
| Datasets | `hotpot` `trivia` `msmarco` `nq_open` `hotpot_context` `musique` `2wiki_comp` |

`Base` is the untrained model. RLVR trains on correctness only. RLCR and RD train the
model to state its confidence, RLCR with a Brier-score reward and RD (Rewarding Doubt)
with a log-score reward.

The main results train on `hotpot` and evaluate on `hotpot`, `trivia`, `msmarco` and
`nq_open`. The context experiments train on `hotpot_context` and evaluate on
`hotpot_context`, `musique` and `2wiki_comp`.

Training and evaluation read the same YAML files, in this order: `configs/default.yaml`,
`configs/judge.yaml`, `configs/models/<model>.yaml`, `configs/methods/<method>.yaml` and
`configs/datasets/<dataset>.yaml`. Each file overrides the ones before it, and
`--set key=value` on the command line overrides all of them. `$JUDGE` is a set of such
overrides.

## 4. Training

Every trainer takes `--model`, `--dataset`, `--exp-name` and `--set`. Checkpoints go to
`ckpt/<METHOD>/<exp-name>/` and training logs to `results/<exp-name>/`. Give each run its
own `--exp-name`. Without it, all runs of one method and model write to the same
checkpoint directory.

**CSR** uses the correctness reward plus the semantic calibration reward. The
calibration weight follows a sigmoid schedule from 0.8 to 1.2. This schedule is the
default in `configs/methods/csr.yaml`, so the command needs no extra flag:

```bash
python train/train_CSR.py --model qwen --dataset hotpot --exp-name CSR_qwen_hotpot $JUDGE
```

**RLVR** uses the correctness reward only. All other settings are the same as CSR:

```bash
python train/train_RLVR.py --model qwen --dataset hotpot --exp-name RLVR_qwen_hotpot $JUDGE
```

**RLCR** follows the published RLCR recipe, ported to Unsloth. As in that recipe, it
always trains on HotpotQA prompts that include the supporting paragraphs, whether
`--dataset` is `hotpot` or `hotpot_context`:

```bash
python train/train_RLCR.py --model qwen --dataset hotpot --exp-name RLCR_qwen_hotpot $JUDGE
```

**RD** follows the upstream Rewarding Doubt release and runs in `csr-rd`. The paper's RD
results use the flags below. For `qwen32b`, also add
`--set method.train.ppo_mini_batch_size=2 --set method.train.ppo_gradient_accumulation_steps=2`
to fit in memory.

```bash
conda activate csr-rd
python train/train_RD.py --model qwen --dataset hotpot --exp-name RD_qwen_hotpot \
  --set method.train.confidence_stop=true --set method.train.strict_format=true \
  --set method.train.learning_rate=1e-5 --set method.train.batch_size=8 \
  --set method.train.ppo_temperature=1.0 --set method.train.ppo_repetition_penalty=1.0
conda activate csr
```

## 5. Evaluation

Each evaluation writes `results/<exp-name>/{config.yaml,results.json,results.csv}`. Do
not reuse a training `--exp-name` here, because that directory already holds the training
logs. Without `--exp-name`, the name is `<METHOD>_<model>_<dataset>_<checkpoint>`, for example
`CSR_qwen_trivia_checkpoint-2500`, and `Base` runs end in `base`. If the directory
already exists, the run resumes from the saved records. Add `--fresh` to start over.

### 5.1 Base, CSR and RLVR

`evaluation/eval_semantic.py` samples $K = 16$ rollouts per question and computes the
confidence $c(x)$ from them. Training uses $K = 8$. `Base` is the untrained model and takes no `--ckpt`:

```bash
for ds in hotpot trivia msmarco nq_open; do
  python evaluation/eval_semantic.py --method Base --model qwen --dataset $ds $JUDGE
  python evaluation/eval_semantic.py --method CSR --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/CSR/CSR_qwen_hotpot/checkpoint-2500
  python evaluation/eval_semantic.py --method RLVR --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RLVR/RLVR_qwen_hotpot/checkpoint-2500
done
```

### 5.2 RLCR and RD

`evaluation/eval_verbalized.py` parses the confidence the model states:

```bash
for ds in hotpot trivia msmarco nq_open; do
  python evaluation/eval_verbalized.py --method RLCR --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RLCR/RLCR_qwen_hotpot/checkpoint-2500
  python evaluation/eval_verbalized.py --method RD --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RD/RD_qwen_hotpot/model_finetuned_step_1250
done
```

### 5.3 Analysis

These scripts re-score finished evaluations without a GPU and produce the appendix
tables:

| Script | Reports |
| --- | --- |
| `analysis/estimators.py` | Semantic Density and SIMBA-UQ's Bayesian aggregation |
| `analysis/aggregation.py` | SIMBA-UQ's arithmetic aggregation and the `clf-pairs+gen` classifier |
| `analysis/selective.py` | Selective prediction (AURAC, AUGRC) |

Semantic Density and `clf-pairs+gen` also use the mean token log-probability of each
rollout, which `analysis/logprobs_hf.py` computes once per run on a GPU:

```bash
python analysis/logprobs_hf.py --model qwen --method CSR \
  --ckpt ckpt/CSR/CSR_qwen_hotpot/checkpoint-2500 --datasets hotpot trivia msmarco nq_open
```

The analysis scripts and `scripts/f1_judge_eval.py` look up runs by the names the paper
used. These names are set in `analysis/runs.py`, `analysis/logprobs_hf.py` and
`scripts/f1_judge_eval.py`. Change them to your run names before running the scripts.

The run names use backbone tags (`llama8b`, `qwen7b`, `mistral7b`) in place of the
`--model` keys, so `analysis/aggregation.py` also takes `--model qwen7b`,
`llama8b` or `mistral7b`. It scores one backbone per call.

---

# Citation

If you find this code helpful in your research, please consider citing:

```bibtex
@inproceedings{csr2026calibrating,
title={Calibrating LLMs with Semantic-level Reward},
author={Yu, Fengfei and Niu, Ruijia and Wu, Dongxia and Ma, Yian and Yu, Rose},
booktitle={NeurIPS},
year={2026}
}
```

# Implementation Credits

Some of our implementations build on other repositories. We thank the authors of
[RLCR](https://github.com/damanimehul/RLCR),
[Rewarding Doubt](https://github.com/pasta99/RewardingDoubt),
[Unsloth](https://github.com/unslothai/unsloth),
[TRL](https://github.com/huggingface/trl) and
[vLLM](https://github.com/vllm-project/vllm) for making their code publicly available.

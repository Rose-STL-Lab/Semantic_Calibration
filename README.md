<h1 align="center">Calibrating LLMs with Semantic-level Reward</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2605.15588"> <img src="https://img.shields.io/badge/arXiv-2605.15588-b31b1b" alt="arXiv"> </a>
    <img src="https://img.shields.io/badge/NeurIPS'26-blue" alt="NeurIPS 2026">
</p>

This repository contains the code for our NeurIPS 2026 paper
*[Calibrating LLMs with Semantic-level Reward](https://arxiv.org/abs/2605.15588)*.

![Overview of CSR](assets/CSR_overview.png)

**Overview.** For each question $x$, CSR draws $K$ rollouts from the policy
$\pi_\theta$ and scores each one with a verifiable correctness reward
$r_{\mathrm{RLVR}}$ and a semantic calibration reward $r_{\mathrm{Calibration}}$. The
policy is updated with group-relative advantages. Training pulls correct rollouts into a
tight semantic cluster and leaves incorrect ones dispersed, so agreement among rollouts
becomes a signal of correctness.

The confidence is $c(x) = \exp(-\mathrm{SE}(x))$, where $\mathrm{SE}$ is the semantic
entropy of the rollouts. The model never states a confidence itself.

## Repository layout

| Path | Contents |
| --- | --- |
| `csr/` | Library code: rewards, semantic clustering, judge client, config loader, metrics |
| `train/` | Trainers: `train_CSR.py`, `train_RLVR.py`, `train_RLCR.py`, `train_RD.py` |
| `evaluation/` | `eval_semantic.py` (Base, RLVR, CSR) and `eval_verbalized.py` (RLCR, RD) |
| `configs/` | YAML settings: `default.yaml`, `judge.yaml`, `models/`, `methods/`, `datasets/` |
| `analysis/` | Appendix analyses: alternative estimators, aggregation rules, selective prediction |
| `scripts/` | `serve/serve_judge.sh` (local judge server) and `f1_judge_eval.py` (token-F1 re-clustering) |
| `rd_util/` | Rewarding Doubt baseline utilities, kept close to upstream |

## 1. Installation

All experiments ran on NVIDIA A100 80GB GPUs, one GPU per training or evaluation run,
with Python 3.11, PyTorch 2.9.0 and CUDA 12.8. Install `torch` first so that pip does not
replace it with another build:

```bash
conda create -n csr python=3.11 && conda activate csr
pip install torch==2.9.0            # choose the build for your CUDA
pip install -r requirements.txt
```

RD training needs an older stack (Python 3.10, PyTorch 2.5.1, CUDA 12.4) and runs in a
separate environment. Everything else, including RD evaluation, runs in `csr`.

```bash
conda create -n csr-rd python=3.10 && conda activate csr-rd
pip install torch==2.5.1
pip install -r requirements-rd.txt
pip install "transformers>=4.51.3" "unsloth>=2025.4.7"   # only for qwen14b and qwen32b
```

## 2. Judge

An LLM judge decides whether an answer matches the gold answer and whether two answers
mean the same thing. It is called by CSR and RLVR training, by the periodic validation in
RLCR training, and by every evaluation. RD training does not use it.

**OpenAI API (default).** The paper uses GPT-4.1-nano:

```bash
export OPENAI_API_KEY=sk-...
```

**Local server.** Any model that vLLM serves can act as the judge. Start it on its own
GPU and run training and evaluation on another (e.g. `CUDA_VISIBLE_DEVICES=0`):

```bash
# bash scripts/serve/serve_judge.sh <GPU> <PORT> <HuggingFace model>
bash scripts/serve/serve_judge.sh 1 8310 openai/gpt-oss-20b
JUDGE="--set judge.base_url=http://127.0.0.1:8310/v1 --set judge.model_id=gpt-oss-20b \
       --set judge.reasoning_effort=low --set judge.max_tokens=1024"
```

`judge.model_id` is the last part of the model name. `reasoning_effort` and the larger
`max_tokens` are for reasoning models; other models need only `base_url` and
`model_id`. The commands below include `$JUDGE`; with the OpenAI API, leave it unset.

**Token-F1 equivalence.** Adding `--set judge.equivalence=f1` to CSR training or
`eval_semantic.py` compares rollouts by token F1 instead of the judge. Correctness is
still judged by the LLM. The threshold defaults to the paper's value: 0.80 for `qwen` and
0.50 for the other models.

## 3. Configuration

| Option | Values |
| --- | --- |
| `--method` | `Base` `RLVR` `CSR` `RLCR` `RD` |
| `--model` | `llama` `qwen` `mistral` `qwen14b` `qwen32b` |
| `--dataset` | `hotpot` `trivia` `msmarco` `nq_open` `hotpot_context` `musique` `2wiki_comp` |

`Base` is the untrained model and RLVR trains on correctness alone. RLCR and RD train the
model to state its confidence, with a Brier-score and a log-score reward respectively.

- **Main results:** train on `hotpot`; evaluate on `hotpot`, `trivia`, `msmarco` and `nq_open`.
- **Context-grounded QA:** train on `hotpot_context`; evaluate on `hotpot_context`,
  `musique` and `2wiki_comp`.

Datasets download from the HuggingFace Hub on first use and are cached under `HF_HOME`.

Each run merges `configs/default.yaml`, `configs/judge.yaml`, `configs/models/<model>.yaml`,
`configs/methods/<method>.yaml` and `configs/datasets/<dataset>.yaml`, in that order, with
later files taking precedence. `--set key=value` overrides all of them.

## 4. Training

Each trainer takes `--model`, `--dataset`, `--exp-name` and `--set`. Checkpoints are
written to `ckpt/<METHOD>/<exp-name>/` and logs to `results/<exp-name>/`. Give every run
its own `--exp-name`; otherwise runs of the same method and model share a checkpoint
directory.

```bash
# CSR: correctness + semantic calibration reward (weight ramped 0.8 -> 1.2 on a sigmoid)
python train/train_CSR.py  --model qwen --dataset hotpot --exp-name CSR_qwen_hotpot  $JUDGE

# RLVR: correctness reward only, otherwise identical to CSR
python train/train_RLVR.py --model qwen --dataset hotpot --exp-name RLVR_qwen_hotpot $JUDGE

# RLCR: the published recipe, ported to Unsloth
python train/train_RLCR.py --model qwen --dataset hotpot --exp-name RLCR_qwen_hotpot $JUDGE

# RD: the Rewarding Doubt baseline, in the csr-rd environment
conda activate csr-rd
python train/train_RD.py   --model qwen --dataset hotpot --exp-name RD_qwen_hotpot
conda activate csr
```

The defaults in `configs/methods/` reproduce the paper. A few notes:

- RLCR always trains on HotpotQA prompts with supporting paragraphs, as in its original
  recipe, whether `--dataset` is `hotpot` or `hotpot_context`.
- For `qwen14b`, the paper ramps the CSR weight from 0.5 to 0.7:
  `--set method.train.cali_w_start=0.5 --set method.train.cali_w_end=0.7`.
- For RD on `qwen32b`, add `--set method.train.ppo_mini_batch_size=2
  --set method.train.ppo_gradient_accumulation_steps=2` to fit in memory.
- Every 0.1 epoch (250 steps; 125 for RD), the trainers score the policy on the
  evaluation split. Set `--set method.train.eval_every_steps=0` to turn this off.

## 5. Evaluation

Each evaluation writes `results/<exp-name>/{config.yaml,results.json,results.csv}`.
Without `--exp-name`, the name is `<METHOD>_<model>_<dataset>_<checkpoint>`, e.g.
`CSR_qwen_trivia_checkpoint-2500`. The commands below pass the names that the analysis
scripts in Section 6 look for; any other name works too, as long as it differs from the
training run's. An interrupted run resumes from its saved records; `--fresh` starts over.

**Base, RLVR and CSR** sample $K = 16$ rollouts per question (training uses $K = 8$) and
compute $c(x)$ from them. `Base` takes no `--ckpt`.

```bash
for ds in hotpot trivia msmarco nq_open; do
  python evaluation/eval_semantic.py --method Base --model qwen --dataset $ds $JUDGE \
    --exp-name Base_qwen7b_${ds}_base_nano_k16
  python evaluation/eval_semantic.py --method CSR  --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/CSR/CSR_qwen_hotpot/checkpoint-2500 --exp-name CSR_qwen7b_${ds}_z10_step2500_nano_k16
  python evaluation/eval_semantic.py --method RLVR --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RLVR/RLVR_qwen_hotpot/checkpoint-2500 --exp-name RLVR_qwen7b_${ds}_step2500_nano_k16
done
```

**RLCR and RD** answer once with greedy decoding and state a confidence, which is parsed
from the output.

```bash
for ds in hotpot trivia msmarco nq_open; do
  python evaluation/eval_verbalized.py --method RLCR --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RLCR/RLCR_qwen_hotpot/checkpoint-2500 --exp-name RLCR_qwen7b_${ds}_step2500_nano_k1
  python evaluation/eval_verbalized.py --method RD   --model qwen --dataset $ds $JUDGE \
    --ckpt ckpt/RD/RD_qwen_hotpot/model_finetuned_step_1250 --exp-name RD_qwen7b_${ds}_b8_step1250_nano_k1
done
```

## 6. Analysis

These scripts re-score finished evaluations for the appendix. They need no GPU and no
judge.

| Script | Reports |
| --- | --- |
| `analysis/estimators.py` | Semantic Density and SIMBA-UQ's Bayesian aggregation |
| `analysis/aggregation.py` | SIMBA-UQ's arithmetic aggregation and the `clf-pairs+gen` classifier (one backbone per call, `--model qwen7b` etc.) |
| `analysis/selective.py` | Selective prediction (AURAC, AUGRC) and confidence parse rates (`--table selective\|parse\|all`) |
| `scripts/f1_judge_eval.py` | Token-F1 re-clustering of saved CSR-F1 rollouts |

Semantic Density and `clf-pairs+gen` also use the mean token log-probability of each
rollout (without it they fall back to uniform weights and drop the `+gen` feature).
`analysis/logprobs_hf.py` computes it once per run, on a GPU:

```bash
DS="hotpot trivia msmarco nq_open"
python analysis/logprobs_hf.py --model qwen --method Base --datasets $DS
python analysis/logprobs_hf.py --model qwen --method RLVR --datasets $DS \
  --ckpt ckpt/RLVR/RLVR_qwen_hotpot/checkpoint-2500
python analysis/logprobs_hf.py --model qwen --method CSR  --datasets $DS \
  --ckpt ckpt/CSR/CSR_qwen_hotpot/checkpoint-2500
```

For the token-F1 rows, train CSR with `--set judge.equivalence=f1`, evaluate it as in
Section 5 with `--exp-name CSR_qwen7b_${ds}_f1eq080_step2500_nano_k16` (`f1eq050`
for `llama8b` and `mistral7b`), then re-cluster those rollouts with token F1:

```bash
python scripts/f1_judge_eval.py --models qwen7b
```

The scripts find runs through the name templates used above, such as
`CSR_{model}_{dataset}_z10_step2500_nano_k16`, where `{model}` is a backbone tag
(`llama8b`, `qwen7b`, `mistral7b`). The templates are defined in `analysis/runs.py`,
`analysis/logprobs_hf.py` and `scripts/f1_judge_eval.py`; edit them if you name runs
differently. Missing or unfinished runs are shown as `--` (`estimators.py`), skip their
backbone (`selective.py`), or stop the script with a list (`aggregation.py`).

## Citation

```bibtex
@inproceedings{csr2026calibrating,
title={Calibrating LLMs with Semantic-level Reward},
author={Yu, Fengfei and Niu, Ruijia and Wu, Dongxia and Ma, Yian and Yu, Rose},
booktitle={NeurIPS},
year={2026}
}
```

## Acknowledgements

This code builds on [RLCR](https://github.com/damanimehul/RLCR),
[Rewarding Doubt](https://github.com/pasta99/RewardingDoubt),
[Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl)
and [vLLM](https://github.com/vllm-project/vllm). We thank their authors for releasing
their code.

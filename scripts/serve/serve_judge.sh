#!/usr/bin/env bash
# Serve the equivalence / correctness judge that training and evaluation call.
#   bash scripts/serve/serve_judge.sh [GPU] [PORT] [MODEL]
# Defaults are the paper's judge, and port 8310 is configs/judge.yaml's default
# base_url, so a default run needs no --set. For anything else:
#   --set judge.base_url=http://127.0.0.1:<PORT>/v1 --set judge.model_id=<MODEL>
# HF_HOME is inherited; set HF_HUB_OFFLINE=1 to forbid downloads.
set -uo pipefail

GPU=${1:-0}
PORT=${2:-8310}
MODEL=${3:-openai/gpt-oss-20b}

export CUDA_VISIBLE_DEVICES=$GPU
# --served-model-name is the basename, hence judge.model_id: gpt-oss-20b.
# Serving limits only -- the judge's own generation budget is judge.max_tokens.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$(basename "$MODEL")" \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

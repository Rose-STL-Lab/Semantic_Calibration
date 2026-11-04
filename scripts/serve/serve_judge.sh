#!/usr/bin/env bash
# Serve the judge locally over an OpenAI-compatible API; see scripts/serve/README.md.
#   bash scripts/serve/serve_judge.sh [GPU] [PORT] [MODEL]
set -uo pipefail

GPU=${1:-0}
PORT=${2:-8310}
MODEL=${3:-openai/gpt-oss-20b}

export CUDA_VISIBLE_DEVICES=$GPU
# served under the basename, e.g. judge.model_id=gpt-oss-20b
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$(basename "$MODEL")" \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

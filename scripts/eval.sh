#!/usr/bin/env bash
# Evaluate one method / model over the four datasets; DATASETS= picks a subset.
#   bash scripts/eval.sh Base qwen                      # frozen model, takes no ckpt
#   bash scripts/eval.sh CSR  qwen <ckpt>
#   GPU=0 DATASETS="hotpot trivia" bash scripts/eval.sh CSR qwen <ckpt>
# All evaluation runs in the unsloth env, RD and RLCR included -- only RD *training* needs environment-rd.yml.
# Results land under results/; extra --set arguments are forwarded verbatim.

set -euo pipefail

usage() {
  echo "usage: eval.sh {Base|RLVR|CSR|RD|RLCR} <model> [ckpt] [--set k=v ...]" >&2
  exit 1
}
[ $# -ge 2 ] || usage
METHOD="$1"; MODEL="$2"; shift 2
CKPT=""
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CKPT="$1"; shift; fi

[ -f "$(dirname "$0")/../configs/models/$MODEL.yaml" ] \
  || { echo "unknown model: $MODEL (no configs/models/$MODEL.yaml)" >&2; usage; }
case "$METHOD" in
  Base|RLVR|CSR) ENTRY=evaluation/eval_semantic.py ;;
  RD|RLCR)       ENTRY=evaluation/eval_verbalized.py ;;
  *) echo "unknown method: $METHOD" >&2; usage ;;
esac
if [[ "$METHOD" == "Base" && -n "$CKPT" ]]; then
  echo "Base evaluates the frozen model and takes no checkpoint" >&2; exit 1
fi

cd "$(dirname "$0")/.."

# `--set judge.backend=f1` scores with token F1: no API key, no judge server.
F1_JUDGE=0
for arg in "$@"; do
  case "$arg" in judge.backend=f1|judge.backend=token_f1) F1_JUDGE=1 ;; esac
done

# Judge key: csr/judge.py accepts either OPENAI_API_KEY or OPENAI_API_KEY_4O.
if [[ $F1_JUDGE -eq 0 ]]; then
  if [[ -z "${OPENAI_API_KEY:-}${OPENAI_API_KEY_4O:-}" ]]; then
    echo "every evaluation needs the LLM judge: export OPENAI_API_KEY" >&2
    echo "  (or score with token F1 instead: --set judge.backend=f1)" >&2
    exit 1
  fi
fi
if ! python -c "import unsloth" >/dev/null 2>&1; then
  echo "unsloth is not importable; activate the environment built from" >&2
  echo "environment.yml before running this." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for ds in ${DATASETS:-hotpot trivia msmarco nq_open}; do
  echo "=== [$(date '+%F %T')] eval $METHOD / $MODEL / $ds  (GPU $CUDA_VISIBLE_DEVICES) ==="
  if [[ -n "$CKPT" ]]; then
    python "$ENTRY" --method "$METHOD" --model "$MODEL" --dataset "$ds" --ckpt "$CKPT" "$@"
  else
    python "$ENTRY" --method "$METHOD" --model "$MODEL" --dataset "$ds" "$@"
  fi
done
echo "=== [$(date '+%F %T')] eval done: $METHOD / $MODEL ==="

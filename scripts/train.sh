#!/usr/bin/env bash
# Train one method for one model: train.sh {CSR|RLVR|RLCR|RD} {llama|qwen|mistral} [gpu]
#   bash scripts/train.sh CSR qwen 0 --set method.train.cali_w_end=0.4   # --set is forwarded verbatim
# CSR/RLVR/RLCR train in the unsloth env (environment.yml); RD training needs the
# Rewarding Doubt env (environment-rd.yml) -- activate it yourself, this only warns.
# Outputs: ckpt/<METHOD>_<model>_<dataset>/ and results/<METHOD>_<model>_<dataset>/

set -euo pipefail

usage() {
    # Kept out of ${1:?...}: a brace in the message would close the expansion early.
    echo "usage: train.sh {CSR|RLVR|RLCR|RD} {llama|qwen|mistral} [gpu] [--set k=v ...]" >&2
    exit 1
}
[ $# -ge 2 ] || usage
METHOD="$1"
MODEL="$2"
GPU="${3:-0}"
shift $(( $# < 3 ? $# : 3 ))

case "$METHOD" in CSR|RLVR|RLCR|RD) ;; *) echo "unknown method: $METHOD" >&2; exit 1 ;; esac
# Any model with a configs/models/<model>.yaml (e.g. qwen14b), same rule csr/config.py uses.
[[ -f "$(dirname "$0")/../configs/models/$MODEL.yaml" ]] || { echo "unknown model: $MODEL (no configs/models/$MODEL.yaml)" >&2; exit 1; }

# `--set judge.backend=f1` scores with token F1: no API key, no judge server.
F1_JUDGE=0
for arg in "$@"; do
  case "$arg" in judge.backend=f1|judge.backend=token_f1) F1_JUDGE=1 ;; esac
done

if [[ "$METHOD" != "RLCR" && $F1_JUDGE -eq 0 ]]; then
  # csr/judge.py accepts either variable name.
  if [[ -z "${OPENAI_API_KEY:-}${OPENAI_API_KEY_4O:-}" ]]; then
    echo "$METHOD needs the LLM judge but no key is set." >&2
      echo "  export OPENAI_API_KEY=...   # EMPTY is fine for a local vLLM judge" >&2
    echo "  (or score with token F1 instead: --set judge.backend=f1)" >&2
    exit 1
  fi
fi
if [[ "$METHOD" == "RD" ]]; then
  echo "note: RD training needs the upstream Rewarding Doubt environment, not the" >&2
  echo "      unsloth one (which is what CSR, RLVR and RLCR use). Activate it first:" >&2
  echo "      conda activate <the env built from environment-rd.yml>" >&2
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== train $METHOD / $MODEL on GPU $GPU ==="
python "train/train_${METHOD}.py" --model "$MODEL" "$@"

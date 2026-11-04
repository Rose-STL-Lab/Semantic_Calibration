# Judge server

Training and evaluation reach the semantic-equivalence / correctness judge over
HTTP, so one has to be up before any run that calls the judge.

    bash scripts/serve/serve_judge.sh [GPU] [PORT] [MODEL]
    bash scripts/serve/serve_judge.sh 0 8310 openai/gpt-oss-20b

`configs/judge.yaml` leaves `base_url` null (the hosted OpenAI endpoint), so each
run must be pointed at the local server:

    --set judge.base_url=http://127.0.0.1:<PORT>/v1 --set judge.model_id=gpt-oss-20b
    --set judge.reasoning_effort=low --set judge.max_tokens=1024   # gpt-oss-20b only

`judge.model_id` must match the served model name, which is the basename of the
model path. Set `HF_HUB_OFFLINE=1` to forbid downloads.

Any OpenAI-compatible endpoint works in place of this script, including a hosted
one; `configs/judge.yaml` holds the defaults.

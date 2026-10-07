# Judge server

Training and evaluation reach the semantic-equivalence / correctness judge over
HTTP, so one has to be up before any run that is not scored by token F1.

    bash scripts/serve/serve_judge.sh [GPU] [PORT] [MODEL]
    bash scripts/serve/serve_judge.sh 0 8310 openai/gpt-oss-20b   # the paper's setup

The port is what `--set judge.base_url=http://127.0.0.1:<PORT>/v1` points at, and
`--set judge.model_id=<name>` has to match `--served-model-name`, which the script
derives from the model path.

A local server ignores the API key, but the trainer still requires the variable to
exist, so export `OPENAI_API_KEY=EMPTY` before launching a run.

Any OpenAI-compatible endpoint works in place of this script, including a hosted
one; `configs/judge.yaml` holds the defaults.

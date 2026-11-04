# Judge server

Training and evaluation reach the semantic-equivalence / correctness judge over
HTTP, so one has to be up before any run that calls the judge.

    bash scripts/serve/serve_judge.sh [GPU] [PORT] [MODEL]
    bash scripts/serve/serve_judge.sh 0 8310 openai/gpt-oss-20b

The port is what `--set judge.base_url=http://127.0.0.1:<PORT>/v1` points at, and
`--set judge.model_id=<name>` has to match `--served-model-name`, which the script
derives from the model path.

Any OpenAI-compatible endpoint works in place of this script, including a hosted
one; `configs/judge.yaml` holds the defaults.

# data/

Working directory for datasets. Nothing here is tracked by git except this file.

The four corpora download themselves on first use through the `datasets` API and
need no manual download -- see the table in the top-level README. They land in
whatever HuggingFace cache the environment points at, so set `HF_HOME` yourself
if you want them somewhere other than `~/.cache/huggingface`:

```bash
export HF_HOME=/path/to/a/shared/cache
```

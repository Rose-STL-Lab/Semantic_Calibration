# data/

Working directory for datasets. Nothing here is tracked by git except this file.

The datasets in `configs/datasets/` download on first use through the HuggingFace
`datasets` API (see the table in the top-level README) into the HuggingFace cache.
Set `HF_HOME` to put them somewhere other than `~/.cache/huggingface`:

```bash
export HF_HOME=/path/to/a/shared/cache
```

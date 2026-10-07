# data/

Working directory for datasets. Nothing here is tracked by git.

* `hf_cache/` — HuggingFace cache (`HF_HOME`, hub, datasets). All four corpora
  download themselves on first use; set `paths.hf_cache` in
  `configs/default.yaml` (or `--set paths.hf_cache=/somewhere/else`) to point
  elsewhere, for instance at a shared cache.

The datasets themselves are pulled through the `datasets` API and need no
manual download — see the table in the top-level README.

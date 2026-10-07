"""Layered YAML config, merged in order: default -> judge -> model -> method -> dataset -> --set."""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Iterable

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")

METHODS = ["Base", "RLVR", "CSR", "RD", "RLCR"]
MODELS = ["llama", "qwen", "mistral", "qwen14b"]
DATASETS = ["hotpot", "hotpot_context", "trivia", "msmarco", "nq_open"]


class Config(dict):
    """A dict that also answers ``cfg.model.hf_id``."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse an override value with YAML rules (ints, floats, bools, lists)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_override(cfg: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects key.path=value, got {assignment!r}")
    dotted, _, raw = assignment.partition("=")
    parts = [p for p in dotted.strip().split(".") if p]
    if not parts:
        raise ValueError(f"--set expects key.path=value, got {assignment!r}")

    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = _coerce(raw)


def _method_slug(method: str) -> str:
    for known in METHODS:
        if method.lower() == known.lower():
            return known.lower()
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def load_config(
    method: str,
    model: str,
    dataset: str,
    overrides: Iterable[str] = (),
    config_root: str | None = None,
) -> Config:
    root = config_root or CONFIG_ROOT
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")

    cfg: dict = {}
    for part in (
        os.path.join(root, "default.yaml"),
        os.path.join(root, "judge.yaml"),
        os.path.join(root, "models", f"{model}.yaml"),
        os.path.join(root, "methods", f"{_method_slug(method)}.yaml"),
        # dataset merged last so it overrides the method defaults (context split needs a longer window)
        os.path.join(root, "datasets", f"{dataset}.yaml"),
    ):
        cfg = _deep_merge(cfg, _read_yaml(part))

    for assignment in overrides:
        _apply_override(cfg, assignment)

    # Paths in the config are repo-relative unless the user gave an absolute one.
    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str) and not os.path.isabs(value):
            cfg["paths"][key] = os.path.join(REPO_ROOT, value)

    return Config(cfg)


def checkpoint_dir(cfg: Config, exp_name: str | None = None) -> str:
    """``ckpt/<METHOD>/<model.ckpt_name>`` (ckpt_name is llama8b/qwen7b/mistral7b, not the config key).

    ``exp_name`` replaces the model segment so two hyperparameter variants do not overwrite each other.
    """
    leaf = exp_name or cfg["model"].get("ckpt_name") or cfg["model"]["name"]
    return os.path.join(cfg["paths"]["ckpt"], str(cfg["method"]["name"]), str(leaf))


def default_exp_name(cfg: Config, checkpoint: str | None = None) -> str:
    """Fallback name (``CSR_qwen_trivia_checkpoint-2500``) when ``--exp-name`` is absent;
    ``results/`` dirs are named explicitly and the two conventions are not interchangeable.
    """
    parts = [cfg["method"]["name"], cfg["model"]["name"], cfg["dataset"]["name"]]
    if checkpoint:
        parts.append(os.path.basename(str(checkpoint).rstrip("/")))
    else:
        parts.append("base")
    slug = "_".join(str(p) for p in parts)
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in slug)


def add_common_args(parser: argparse.ArgumentParser, *, with_method: bool = True,
                    dataset_default: str | None = None,
                    dataset_choices: list[str] | None = None) -> None:
    """The arguments every entry point shares; ``dataset_default`` makes ``--dataset`` optional."""
    if with_method:
        parser.add_argument("--method", required=True, choices=METHODS,
                            help="which method's config to load")
    parser.add_argument("--model", required=True, choices=MODELS,
                        help="model family (configs/models/<model>.yaml)")
    parser.add_argument("--dataset", required=dataset_default is None,
                        default=dataset_default, choices=dataset_choices or DATASETS,
                        help="dataset (configs/datasets/<dataset>.yaml)")
    parser.add_argument("--exp-name", dest="exp_name", default=None,
                        help="results/<exp_name>/; defaults to METHOD_model_dataset_ckpt")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY.PATH=VALUE",
                        help="override any config value, repeatable "
                             "(e.g. --set method.eval.num_rollouts=16)")


def setup_env(cfg: Config) -> None:
    """Point HuggingFace at the repo-local cache and mute vLLM."""
    hf_cache = cfg["paths"]["hf_cache"]
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_cache, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "hub"))
    if cfg.get("quiet_vllm", True):
        os.environ.setdefault("VLLM_DISABLE_LOGGING", "1")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dump_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, allow_unicode=True, width=100)

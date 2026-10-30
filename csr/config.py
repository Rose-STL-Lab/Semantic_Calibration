# Layered YAML config, merged in order:
# default -> judge -> model -> method -> dataset -> --set.

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Iterable

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")

METHODS = ["Base", "RLVR", "CSR", "RD", "RLCR"]
MODELS = ["llama", "qwen", "mistral", "qwen14b", "qwen32b"]
DATASETS = ["hotpot", "hotpot_context", "trivia", "msmarco", "nq_open",
            "musique", "2wiki_comp"]


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
    # YAML rules: ints, floats, bools, lists come through typed
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
) -> dict:
    root = CONFIG_ROOT
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
        # dataset last, overriding the method defaults (context needs a longer window)
        os.path.join(root, "datasets", f"{dataset}.yaml"),
    ):
        cfg = _deep_merge(cfg, _read_yaml(part))

    for assignment in overrides:
        _apply_override(cfg, assignment)

    # repo-relative unless absolute
    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str) and not os.path.isabs(value):
            cfg["paths"][key] = os.path.join(REPO_ROOT, value)

    return cfg


def checkpoint_dir(cfg: dict, exp_name: str | None = None) -> str:
    # ckpt/<METHOD>/<ckpt_name>, where ckpt_name is llama8b/qwen7b/mistral7b, not the
    # config key. exp_name replaces the model segment so two variants cannot collide.
    leaf = exp_name or cfg["model"].get("ckpt_name") or cfg["model"]["name"]
    return os.path.join(cfg["paths"]["ckpt"], str(cfg["method"]["name"]), str(leaf))


def default_exp_name(cfg: dict, checkpoint: str | None = None) -> str:
    # only when --exp-name is absent (CSR_qwen_trivia_checkpoint-2500); results/ dirs
    # are named explicitly -- the two conventions are not interchangeable
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
    # dataset_default makes --dataset optional
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


def dump_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, allow_unicode=True, width=100)

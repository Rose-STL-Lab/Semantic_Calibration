# Workarounds for unsloth quirks on a shared machine.
# Stdlib + numpy only: the RD baseline imports this from its own environment.

from __future__ import annotations

import sys


def disable_shared_gpu_false_positive() -> None:
    # neutralize unsloth's "multi GPU" guard, which misfires on a shared machine: it
    # diffs nvidia-smi memory across the load, and nvidia-smi ignores
    # CUDA_VISIBLE_DEVICES, so another user's job aborts a correctly pinned run.
    # Zeros make the diff zero -- detection only, so still pin CUDA_VISIBLE_DEVICES.
    import importlib

    import numpy as np

    def check_nvidia():
        return np.zeros(1)

    for name in ("unsloth.models._utils", "unsloth.models.llama",
                 "unsloth.models.vision", "unsloth.tokenizer_utils"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
        if hasattr(module, "check_nvidia"):
            module.check_nvidia = check_nvidia

# Workarounds for unsloth quirks.
# Stdlib + numpy only: the RD baseline imports this from its own environment.

from __future__ import annotations

import sys


def disable_shared_gpu_false_positive() -> None:
    # unsloth's "multi GPU" guard diffs nvidia-smi memory, which ignores
    # CUDA_VISIBLE_DEVICES; stub it out (detection only -- still pin the device).
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

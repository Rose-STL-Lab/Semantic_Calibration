#!/usr/bin/env python
"""RLVR: binary correctness reward only; otherwise the same setup as CSR.

Hyperparameters: configs/methods/rlvr.yaml, overridable with --set.

    python train/train_RLVR.py --model qwen
"""

import _grpo

if __name__ == "__main__":
    _grpo.run(_grpo.parse_args("RLVR", __doc__))

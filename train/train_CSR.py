#!/usr/bin/env python
"""CSR: GRPO with a verifiable correctness reward plus a semantic calibration reward.

Answer-only: no chain-of-thought, no verbalized confidence. The calibration weight
runs cali_w_start -> cali_w_end (equal by default, so constant).
Hyperparameters: configs/methods/csr.yaml, overridable with --set.

    python train/train_CSR.py --model qwen
    python train/train_CSR.py --model llama --set method.train.cali_w_end=0.4
"""

import _grpo

if __name__ == "__main__":
    _grpo.run(_grpo.parse_args("CSR", __doc__))

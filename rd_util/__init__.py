"""Rewarding Doubt helper package, vendored from the upstream repository.

Source: https://github.com/pasta99/RewardingDoubt (arXiv:2503.02623).

Local changes: imports are namespaced as ``rd_util``; LoRA rank and alpha are
32/32 with explicit target modules, matching this repo's other methods, where
upstream uses 8/8 (ModelLoader.py); the Answer/Confidence parser also accepts a
newline separator and recovers the two fields independently (ResponseHandling.py);
HotpotQA was added to the dataset registry (DataHelper.py) with F1 scoring, since
exact match is unreachable for free-form answers (EvaluationMetrics.py).

Only ``train/train_RD.py`` uses this package, and only in the RD conda
environment (environment-rd.yml).
"""

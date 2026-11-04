"""Rewarding Doubt helper package, vendored from the upstream repository.

Source: https://github.com/pasta99/RewardingDoubt (arXiv:2503.02623).

Local changes: imports are namespaced as ``rd_util``; LoRA rank and alpha are
32/32 with explicit target modules, matching this repo's other methods, where
upstream uses 8/8 (ModelLoader.py); the Answer/Confidence parser also accepts a
newline separator and recovers the two fields independently (ResponseHandling.py);
HotpotQA and its context variant were added to the dataset registry (DataHelper.py),
and train/train_RD.py scores free-form answers by F1 instead of exact match.

Only ``train/train_RD.py`` uses this package, and only in the RD conda
environment (requirements-rd.txt).
"""

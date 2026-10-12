"""Calibration and ranking metrics.  ``p`` is the fraction of the K rollouts the
judge marked correct for the answer-only methods, 0 or 1 for the verbalized ones."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


def expected_calibration_error(confidences: Sequence[float], accuracies: Sequence[float],
                               n_bins: int = 10) -> float:
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    acc = np.asarray(accuracies, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # last bin is closed on the right so confidence == 1.0 is counted
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        ece += (n_k / n) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(confidences: Sequence[float], accuracies: Sequence[float]) -> float:
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(accuracies, dtype=float)
    if len(conf) == 0:
        return 0.0
    return float(np.mean((conf - acc) ** 2))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = spearmanr(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)


def safe_kendall(x: Sequence[float], y: Sequence[float]) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = kendalltau(x, y).correlation
    return None if (r is None or np.isnan(r)) else float(r)


def safe_auroc(accuracies: Sequence[float], confidences: Sequence[float]) -> float | None:
    """AUROC of confidence as a score for "this question was answered correctly"."""
    y_true = np.asarray(accuracies, dtype=float)
    if not np.all((y_true == 0) | (y_true == 1)):
        y_true = (y_true >= 0.5).astype(int)
    else:
        y_true = y_true.astype(int)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    try:
        return float(roc_auc_score(y_true, np.asarray(confidences, dtype=float)))
    except ValueError:
        return None


def summarize(confidences: Sequence[float], accuracies: Sequence[float],
              n_bins: int = 10) -> dict:
    """The metric block reported for every run."""
    conf = list(confidences)
    acc = list(accuracies)
    return {
        "n": len(acc),
        "accuracy": float(np.mean(acc)) if acc else float("nan"),
        "avg_confidence": float(np.mean(conf)) if conf else float("nan"),
        "ece": expected_calibration_error(conf, acc, n_bins),
        "brier": brier_score(conf, acc),
        "spearman": safe_spearman(conf, acc),
        "kendall": safe_kendall(conf, acc),
        "auroc": safe_auroc(acc, conf),
    }

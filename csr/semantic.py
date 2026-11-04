# Semantic clustering and the confidence proxy for Base / RLVR / CSR.
# One symmetric judgment per unordered pair (as in CSR's training reward), then
# SE = -sum p log p over the clusters and c = exp(-SE) in (0, 1].

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def semantic_cluster_ids(samples: Sequence[str], question: str, judge) -> list[int]:
    # an unparsable verdict counts as not-equivalent
    n = len(samples)
    if n <= 1:
        return list(range(n))

    a_list: list[str] = []
    b_list: list[str] = []
    pair_index: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_index[(i, j)] = len(a_list)
            a_list.append(samples[i])
            b_list.append(samples[j])

    verdicts, _unparsed, missing = judge.symmetric_batched(
        a_list, b_list, [question] * len(a_list))
    dropped = set(missing)

    return _assign_clusters(n, pair_index, verdicts, dropped)


def _assign_clusters(n: int, pair_index: dict, verdicts, dropped) -> list[int]:
    # greedy: each rollout joins the first earlier cluster head judged equivalent
    cluster_ids = [-1] * n
    next_id = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, n):
            if cluster_ids[j] != -1:
                continue
            idx = pair_index[(i, j)]
            if idx not in dropped and int(verdicts[idx]) == 1:
                cluster_ids[j] = next_id
        next_id += 1

    return cluster_ids


def cluster_entropy(cluster_ids: Sequence[int]) -> float:
    # natural log; fsum keeps equal cluster sizes tied exactly regardless of rollout order
    n = len(cluster_ids)
    if n <= 1:
        return 0.0
    counts = np.bincount(np.asarray(cluster_ids, dtype=int))
    p = counts[counts > 0] / n
    return float(-math.fsum(float(x) * math.log(float(x)) for x in p))


def entropy_to_confidence(entropy: float) -> float:
    return float(np.exp(-float(entropy)))


def _confidence_block(cluster_ids: list[int]) -> dict:
    entropy = cluster_entropy(cluster_ids)
    return {
        "cluster_ids": cluster_ids,
        "num_clusters": len(set(cluster_ids)),
        "semantic_entropy": entropy,
        "confidence": entropy_to_confidence(entropy),
    }


_EMPTY_BLOCK = {"cluster_ids": [], "num_clusters": 0, "semantic_entropy": 0.0, "confidence": 1.0}


def semantic_confidence(samples: Sequence[str], question: str, judge) -> dict:
    if not samples:
        return dict(_EMPTY_BLOCK)
    return _confidence_block(semantic_cluster_ids(samples, question, judge))


def semantic_confidence_batch(samples_list: Sequence[Sequence[str]], questions: Sequence[str],
                              judge) -> list[dict]:
    # every pair of every question in one symmetric_batched call
    out: list[dict | None] = [None] * len(samples_list)
    a_list: list[str] = []
    b_list: list[str] = []
    q_list: list[str] = []
    layout = []
    for pos, (samples, question) in enumerate(zip(samples_list, questions)):
        n = len(samples)
        if n == 0:
            out[pos] = dict(_EMPTY_BLOCK)
            continue
        if n == 1:
            out[pos] = _confidence_block([0])
            continue
        offset = len(a_list)
        pair_index: dict[tuple[int, int], int] = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_index[(i, j)] = len(a_list) - offset
                a_list.append(samples[i])
                b_list.append(samples[j])
                q_list.append(question)
        layout.append((pos, n, offset, pair_index))

    if layout:
        verdicts, _unparsed, missing = judge.symmetric_batched(a_list, b_list, q_list)
        missing = set(missing)
        for pos, n, offset, pair_index in layout:
            size = len(pair_index)
            local = verdicts[offset:offset + size]
            dropped = {m - offset for m in missing if offset <= m < offset + size}
            out[pos] = _confidence_block(_assign_clusters(n, pair_index, local, dropped))
    return out

"""Ranking metrics and bootstrap confidence intervals.

Conventions (see DECISIONS.md D9-D10):
- `ranked_rels`: (n_queries, k) relevance grades 0-3 of the returned items, best match first.
- `all_rels`: (n_queries, n_gallery) grades of EVERY gallery item the query may retrieve
  (items the query is not allowed to return must already be set to 0).
- Binary metrics call an item relevant when its grade >= `threshold` (default 2).
- A query with no relevant item gets NaN for binary metrics (undefined, not zero).
- nDCG uses linear gain: DCG = sum(grade_i / log2(i + 1)), positions i = 1..k.
"""

import numpy as np


def _check_k(ranked_rels: np.ndarray, k: int) -> np.ndarray:
    if ranked_rels.shape[1] < k:
        raise ValueError(f"need at least k={k} ranked items per query, got {ranked_rels.shape[1]}")
    return ranked_rels[:, :k]


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den where den > 0, NaN elsewhere."""
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def dcg_at_k(ranked_rels: np.ndarray, k: int) -> np.ndarray:
    top = _check_k(np.asarray(ranked_rels, dtype=float), k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # positions 1..k -> log2(2..k+1)
    return (top * discounts).sum(axis=1)


def ndcg_at_k(ranked_rels: np.ndarray, all_rels: np.ndarray, k: int) -> np.ndarray:
    """DCG of the returned list divided by DCG of the best possible list. NaN if no relevant."""
    ideal = -np.sort(-np.asarray(all_rels, dtype=float), axis=1)[:, :k]
    return _safe_divide(dcg_at_k(ranked_rels, k), dcg_at_k(ideal, k))


def _binary(ranked_rels, all_rels, k, threshold):
    top = _check_k(np.asarray(ranked_rels), k) >= threshold
    n_relevant = (np.asarray(all_rels) >= threshold).sum(axis=1)
    return top, n_relevant


def hit_at_k(ranked_rels, all_rels, k, threshold=2) -> np.ndarray:
    top, n_rel = _binary(ranked_rels, all_rels, k, threshold)
    return np.where(n_rel > 0, top.any(axis=1).astype(float), np.nan)


def recall_at_k(ranked_rels, all_rels, k, threshold=2) -> np.ndarray:
    top, n_rel = _binary(ranked_rels, all_rels, k, threshold)
    return _safe_divide(top.sum(axis=1).astype(float), n_rel)


def mrr(ranked_rels, all_rels, k, threshold=2) -> np.ndarray:
    """Reciprocal rank of the first relevant item within the top k (0 if none found)."""
    top, n_rel = _binary(ranked_rels, all_rels, k, threshold)
    first = top.argmax(axis=1) + 1  # argmax gives the first True; only valid if any True
    rr = np.where(top.any(axis=1), 1.0 / first, 0.0)
    return np.where(n_rel > 0, rr, np.nan)


def average_precision_at_k(ranked_rels, all_rels, k, threshold=2) -> np.ndarray:
    """AP@k = sum of precision@i over relevant positions i, divided by min(R, k)."""
    top, n_rel = _binary(ranked_rels, all_rels, k, threshold)
    precision_at_i = top.cumsum(axis=1) / np.arange(1, k + 1)
    numerator = (precision_at_i * top).sum(axis=1)
    return _safe_divide(numerator, np.minimum(n_rel, k).astype(float))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling queries with replacement. NaNs are dropped."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) and its bootstrap CI, using the same resampled queries for both.

    Only queries where both systems have a value are used. If the interval excludes 0, the
    systems differ beyond query noise.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci(a[keep] - b[keep], n_boot=n_boot, alpha=alpha, seed=seed)

"""Exact per-signal rankings and Reciprocal Rank Fusion (DECISIONS.md D22, D23).

Rankings are (n_queries, m) arrays of gallery indices, best first. -1 pads a list that has fewer
than m candidates (e.g. BM25 when few documents share a term); padding never counts.
"""

import numpy as np


def rank_by_scores(
    sims: np.ndarray, exclude: np.ndarray, top_k: int, min_score: float | None = None
) -> np.ndarray:
    """Top-k gallery indices per query by descending score, never returning `exclude`d items.

    With `min_score`, items scoring <= min_score are not returned (padded with -1). BM25 uses
    min_score=0 so documents sharing no term with the query are not ranked arbitrarily.
    Ties are broken by gallery index (stable), so results are deterministic.
    """
    sims = np.array(sims, dtype=np.float64)
    sims[exclude] = -np.inf
    top_k = min(top_k, sims.shape[1])
    order = np.argsort(-sims, axis=1, kind="stable")[:, :top_k]
    top = np.take_along_axis(sims, order, axis=1)
    invalid = ~np.isfinite(top)
    if min_score is not None:
        invalid |= top <= min_score
    return np.where(invalid, -1, order)


def rrf_fuse(
    rankings: list[np.ndarray], k: int = 60, top_k: int = 10, weights: list[float] | None = None
) -> np.ndarray:
    """Fuse rankings: score(d) = sum_s w_s / (k + rank_s(d)), rank counted from 1.

    Ties go to the better best rank, then the lower gallery index. Output: (n_queries, top_k),
    padded with -1 when fewer distinct candidates exist.
    """
    if not rankings:
        raise ValueError("need at least one ranking")
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    n_queries = rankings[0].shape[0]
    if any(r.shape[0] != n_queries for r in rankings):
        raise ValueError("all rankings need the same number of queries")

    out = np.full((n_queries, top_k), -1, dtype=np.int64)
    for q in range(n_queries):
        score: dict[int, float] = {}
        best: dict[int, int] = {}
        for ranking, w in zip(rankings, weights, strict=True):
            for rank, item in enumerate(ranking[q], start=1):
                if item < 0:
                    continue
                item = int(item)
                score[item] = score.get(item, 0.0) + w / (k + rank)
                best[item] = min(best.get(item, rank), rank)
        fused = sorted(score, key=lambda d: (-score[d], best[d], d))[:top_k]
        out[q, : len(fused)] = fused
    return out

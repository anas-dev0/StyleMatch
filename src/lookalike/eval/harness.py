"""Evaluation harness: rank a gallery for each query, then score the rankings.

Rule (DECISIONS.md D11): gallery items that share the query's `product_code` (its own colour
variants) can never be returned, and are also removed from the "all relevant items" used by the
ideal ranking and by recall. Otherwise scores would be inflated or capped below 1.
"""

import numpy as np
import pandas as pd

from lookalike.data.relevance import relevance_matrix
from lookalike.eval import metrics as M


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def same_product_mask(queries: pd.DataFrame, gallery: pd.DataFrame) -> np.ndarray:
    """(n_queries, n_gallery) True where the gallery item shares the query's product_code."""
    q = queries["product_code"].to_numpy()
    g = gallery["product_code"].to_numpy()
    return q[:, None] == g[None, :]


def rank_by_cosine(
    query_emb: np.ndarray, gallery_emb: np.ndarray, exclude: np.ndarray, top_k: int
) -> np.ndarray:
    """Indices of the top_k most cosine-similar gallery items per query, best first.

    `exclude` is a boolean (n_queries, n_gallery) matrix of items that must not be returned.
    """
    sims = _normalise(query_emb) @ _normalise(gallery_emb).T
    sims[exclude] = -np.inf
    part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]  # top_k unordered
    part_sims = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_sims, axis=1, kind="stable")  # sort those top_k, best first
    return np.take_along_axis(part, order, axis=1)


def evaluate_embeddings(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    ndcg_k: int = 10,
    map_k: int = 10,
    threshold: int = 2,
) -> dict[str, np.ndarray]:
    """Per-query metric arrays (one value per query) for a set of embeddings."""
    exclude = same_product_mask(queries, gallery)
    all_rels = relevance_matrix(queries, gallery)
    all_rels[exclude] = 0  # unreachable items must not count as relevant or as ideal

    top_k = max(*ks, ndcg_k, map_k)
    ranked_idx = rank_by_cosine(query_emb, gallery_emb, exclude, top_k)
    ranked_rels = np.take_along_axis(all_rels, ranked_idx, axis=1)

    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"hit@{k}"] = M.hit_at_k(ranked_rels, all_rels, k, threshold)
        out[f"recall@{k}"] = M.recall_at_k(ranked_rels, all_rels, k, threshold)
    out["mrr"] = M.mrr(ranked_rels, all_rels, top_k, threshold)
    out[f"map@{map_k}"] = M.average_precision_at_k(ranked_rels, all_rels, map_k, threshold)
    out[f"ndcg@{ndcg_k}"] = M.ndcg_at_k(ranked_rels, all_rels, ndcg_k)
    return out


def summarise(per_query: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per metric: mean, 95% bootstrap CI, and how many queries had a defined value."""
    rows = []
    for name, values in per_query.items():
        mean, lo, hi = M.bootstrap_ci(values, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "metric": name,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_queries": int((~np.isnan(values)).sum()),
            }
        )
    return pd.DataFrame(rows)

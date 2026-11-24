"""Score precomputed rankings (from fusion, FAISS or Qdrant) with the Phase 2 metrics.

`harness.evaluate_embeddings` ranks by cosine itself. This module scores a ranking produced
elsewhere, under the same rules: own product excluded from results and from the ideal (D11),
grade >= threshold is relevant (D10). Padding (-1) counts as a non-relevant result.
"""

import numpy as np
import pandas as pd

from lookalike.data.relevance import relevance_matrix
from lookalike.eval import metrics as M
from lookalike.eval.harness import same_product_mask


def evaluate_rankings(
    ranked_idx: np.ndarray,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    ndcg_k: int = 10,
    map_k: int = 10,
    threshold: int = 2,
) -> dict[str, np.ndarray]:
    """Per-query metric arrays for (n_queries, >= max k) gallery indices, best first."""
    ranked_idx = np.asarray(ranked_idx, dtype=np.int64)
    top_k = max(*ks, ndcg_k, map_k)
    if ranked_idx.ndim != 2 or len(ranked_idx) != len(queries) or ranked_idx.shape[1] < top_k:
        raise ValueError(f"need ({len(queries)}, >= {top_k}) rankings, got {ranked_idx.shape}")
    ranked_idx = ranked_idx[:, :top_k]

    exclude = same_product_mask(queries, gallery)
    all_rels = relevance_matrix(queries, gallery)
    all_rels[exclude] = 0

    pad = ranked_idx < 0
    safe = np.where(pad, 0, ranked_idx)
    if np.take_along_axis(exclude, safe, axis=1)[~pad].any():
        raise ValueError("a ranking returns an item of the query's own product (rule D11)")
    ranked_rels = np.where(pad, 0, np.take_along_axis(all_rels, safe, axis=1))

    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"hit@{k}"] = M.hit_at_k(ranked_rels, all_rels, k, threshold)
        out[f"recall@{k}"] = M.recall_at_k(ranked_rels, all_rels, k, threshold)
    out["mrr"] = M.mrr(ranked_rels, all_rels, top_k, threshold)
    out[f"map@{map_k}"] = M.average_precision_at_k(ranked_rels, all_rels, map_k, threshold)
    out[f"ndcg@{ndcg_k}"] = M.ndcg_at_k(ranked_rels, all_rels, ndcg_k)
    return out

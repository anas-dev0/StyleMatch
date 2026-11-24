"""Exact nearest-neighbour search with FAISS (IndexFlatIP on L2-normalised vectors).

This is the "ground truth" for approximate search: an approximate index (Qdrant HNSW) is judged
by how many of FAISS's exact top-k it also returns (DECISIONS.md D24). faiss is imported inside
functions so the package imports without it (`uv sync --group index`).
"""

import numpy as np


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_flat_index(emb: np.ndarray):
    """Exact inner-product index over normalised vectors (inner product = cosine)."""
    import faiss

    x = normalise(emb)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def search(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(scores, indices), each (n_queries, k), best first."""
    return index.search(normalise(queries), k)


def search_excluding(
    index, queries: np.ndarray, k: int, query_groups: np.ndarray, gallery_groups: np.ndarray
) -> np.ndarray:
    """Top-k gallery indices per query, skipping gallery items in the query's own group (D11).

    FAISS cannot filter, so it over-fetches by the size of the largest group and drops matches.
    """
    gallery_groups = np.asarray(gallery_groups)
    _, counts = np.unique(gallery_groups, return_counts=True)
    k_fetch = min(k + int(counts.max()), index.ntotal)
    _, idx = search(index, queries, k_fetch)
    out = np.full((len(queries), k), -1, dtype=np.int64)
    for row, (cand, group) in enumerate(zip(idx, np.asarray(query_groups), strict=True)):
        keep = cand[(cand >= 0) & (gallery_groups[cand] != group)][:k]
        out[row, : len(keep)] = keep
    return out


def ann_recall(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Per-query recall@k of an approximate search: |approx ∩ exact| / |exact|."""
    approx, exact = np.asarray(approx), np.asarray(exact)
    if approx.shape[0] != exact.shape[0]:
        raise ValueError("approx and exact must have the same number of queries")
    return np.array(
        [len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact, strict=True)], dtype=float
    )

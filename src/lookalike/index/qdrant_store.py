"""Qdrant collection with named dense vectors, sparse BM25 vectors and payload filters.

One point per article (point id = article_id). Named vectors: `image`, `text` (dense, cosine)
and `bm25` (sparse, weights from `index/sparse.py`, D21). The payload holds product_code and
attributes for filters. Hybrid search = one prefetch per signal + Qdrant's RRF (D22). Every
search can exclude the query's own product_code (D11).

`url=None` gives an in-memory local client. Local mode scans every vector, so it is right for
tests and small demos but cannot measure HNSW (D24): use a server for that.

    python -m lookalike.index.qdrant_store --config configs/index/qdrant.yaml
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config

PAYLOAD_COLS = [
    "product_code",
    "product_type_name",
    "colour_group_name",
    "graphical_appearance_name",
    "product_group_name",
    "department_name",
    "garment_group_name",
]


def connect(url: str | None = None, timeout: int = 60):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=timeout) if url else QdrantClient(":memory:")


def create_collection(
    client,
    name: str,
    dense_dims: dict[str, int],
    sparse_names: tuple[str, ...] = (),
    *,
    hnsw_m: int = 16,
    ef_construct: int = 100,
    full_scan_threshold: int | None = None,
    indexing_threshold: int | None = None,
) -> None:
    """(Re)create `name`. The two thresholds (in KB) are lowered for benchmarks so small
    collections really build and use the HNSW graph instead of a full scan (D24)."""
    from qdrant_client import models

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            v: models.VectorParams(size=d, distance=models.Distance.COSINE)
            for v, d in dense_dims.items()
        },
        sparse_vectors_config={s: models.SparseVectorParams() for s in sparse_names} or None,
        hnsw_config=models.HnswConfigDiff(
            m=hnsw_m, ef_construct=ef_construct, full_scan_threshold=full_scan_threshold
        ),
        optimizers_config=(
            models.OptimizersConfigDiff(indexing_threshold=indexing_threshold)
            if indexing_threshold is not None
            else None
        ),
    )
    with warnings.catch_warnings():  # local mode warns that payload indexes do nothing there
        warnings.simplefilter("ignore", UserWarning)
        client.create_payload_index(name, "product_code", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(name, "product_type_name", models.PayloadSchemaType.KEYWORD)


def _payload(row: dict) -> dict:
    out = {}
    for col in PAYLOAD_COLS:
        if col in row and not pd.isna(row[col]):
            value = row[col]
            out[col] = int(value) if col == "product_code" else str(value)
    return out


def upsert_items(
    client,
    name: str,
    items: pd.DataFrame,
    dense: dict[str, np.ndarray],
    sparse: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None,
    batch_size: int = 256,
) -> int:
    """Upload one point per row of `items` (rows aligned with every vector array)."""
    from qdrant_client import models

    sparse = sparse or {}
    n = len(items)
    for arr in [*dense.values(), *sparse.values()]:
        if len(arr) != n:
            raise ValueError(f"{len(arr)} vectors for {n} items")
    records = items.to_dict("records")
    for start in range(0, n, batch_size):
        points = []
        for i in range(start, min(start + batch_size, n)):
            vector = {v: np.asarray(arr[i], dtype=np.float32).tolist() for v, arr in dense.items()}
            for s, vecs in sparse.items():
                idx, val = vecs[i]
                if len(idx):  # an empty sparse vector is simply left out
                    vector[s] = models.SparseVector(
                        indices=np.asarray(idx).tolist(), values=np.asarray(val).tolist()
                    )
            points.append(
                models.PointStruct(
                    id=int(records[i]["article_id"]), vector=vector, payload=_payload(records[i])
                )
            )
        client.upsert(name, points=points, wait=True)
    return n


def wait_until_indexed(client, name: str, timeout_s: float = 600, poll_s: float = 1.0) -> dict:
    """Wait for the optimiser to finish (status green). Returns point and indexed counts."""
    deadline = time.monotonic() + timeout_s
    while True:
        info = client.get_collection(name)
        if str(info.status).lower().endswith("green") or time.monotonic() > deadline:
            return {
                "status": str(info.status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        time.sleep(poll_s)


def build_filter(exclude_product_code: int | None = None, must: dict | None = None):
    """Payload filter: never return the query's own product; optionally require field values."""
    from qdrant_client import models

    must_not = []
    if exclude_product_code is not None:
        must_not.append(
            models.FieldCondition(
                key="product_code", match=models.MatchValue(value=int(exclude_product_code))
            )
        )
    conditions = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in (must or {}).items()
    ]
    if not must_not and not conditions:
        return None
    return models.Filter(must=conditions or None, must_not=must_not or None)


def _query_value(q):
    from qdrant_client import models

    if isinstance(q, tuple):  # sparse (indices, values)
        idx, val = q
        return models.SparseVector(
            indices=np.asarray(idx).tolist(), values=np.asarray(val).tolist()
        )
    return np.asarray(q, dtype=np.float32).tolist()


def _ids(response) -> list[int]:
    return [int(p.id) for p in response.points]


def search_batch(
    client,
    name: str,
    using: str,
    queries: list,
    k: int,
    exclude_codes: list | None = None,
    *,
    hnsw_ef: int | None = None,
    exact: bool = False,
    chunk: int = 256,
) -> list[list[int]]:
    """One signal. `queries`: dense vectors or sparse (indices, values) tuples. Returns ids."""
    from qdrant_client import models

    params = models.SearchParams(hnsw_ef=hnsw_ef, exact=exact)
    exclude_codes = exclude_codes if exclude_codes is not None else [None] * len(queries)
    out: list[list[int]] = []
    for start in range(0, len(queries), chunk):
        requests = [
            models.QueryRequest(
                query=_query_value(q),
                using=using,
                limit=k,
                filter=build_filter(code),
                params=params,
            )
            for q, code in zip(
                queries[start : start + chunk], exclude_codes[start : start + chunk], strict=True
            )
        ]
        out.extend(_ids(r) for r in client.query_batch_points(name, requests=requests))
    return out


def search_hybrid(
    client,
    name: str,
    queries: dict[str, list],
    k: int,
    exclude_codes: list | None = None,
    *,
    prefetch_limit: int = 100,
    rrf_k: int = 60,
    must: dict | None = None,
) -> list[list[int]]:
    """RRF over one prefetch per named vector. `rrf_k` uses the offline 1-based convention
    (D22): Qdrant counts ranks from 0, so it is sent as rrf_k - 1."""
    from qdrant_client import models

    names = list(queries)
    n = len(queries[names[0]])
    exclude_codes = exclude_codes if exclude_codes is not None else [None] * n
    requests = []
    for i in range(n):
        flt = build_filter(exclude_codes[i], must)
        requests.append(
            models.QueryRequest(
                prefetch=[
                    models.Prefetch(
                        query=_query_value(queries[v][i]), using=v, limit=prefetch_limit, filter=flt
                    )
                    for v in names
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=rrf_k - 1)),
                limit=k,
                filter=flt,
            )
        )
    out: list[list[int]] = []
    for start in range(0, n, 256):
        out.extend(
            _ids(r) for r in client.query_batch_points(name, requests=requests[start : start + 256])
        )
    return out


def load_vectors(root: Path, name: str, article_ids: np.ndarray) -> np.ndarray:
    from lookalike.eval.run_baselines import align_embeddings

    emb = np.load(root / name / "embeddings.npy")
    ids = np.load(root / name / "ids.npy")
    return align_embeddings(emb, ids, article_ids)


def main() -> None:
    from lookalike.embed.text import build_captions
    from lookalike.index.sparse import BM25, tokenize

    parser = argparse.ArgumentParser(description="Build the Qdrant collection from the subset.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    items = pd.read_parquet(cfg["subset_path"]).reset_index(drop=True)
    root = Path(cfg["embeddings_root"])
    ids = items["article_id"].to_numpy()
    dense = {v: load_vectors(root, model, ids) for v, model in cfg["dense_vectors"].items()}
    docs = [tokenize(c) for c in build_captions(items, cfg["bm25_fields"])]
    bm25 = BM25(cfg["bm25_k1"], cfg["bm25_b"]).fit(docs)
    sparse = {"bm25": [bm25.doc_vector(d) for d in docs]}

    client = connect(cfg["qdrant_url"])
    hnsw = cfg["hnsw"]
    start = time.perf_counter()
    create_collection(
        client,
        cfg["collection"],
        {v: a.shape[1] for v, a in dense.items()},
        ("bm25",),
        hnsw_m=hnsw["m"],
        ef_construct=hnsw["ef_construct"],
        full_scan_threshold=hnsw.get("full_scan_threshold"),
        indexing_threshold=hnsw.get("indexing_threshold"),
    )
    upsert_items(client, cfg["collection"], items, dense, sparse)
    state = wait_until_indexed(client, cfg["collection"])
    print(f"Built '{cfg['collection']}' in {time.perf_counter() - start:.1f}s: {state}")


if __name__ == "__main__":
    main()

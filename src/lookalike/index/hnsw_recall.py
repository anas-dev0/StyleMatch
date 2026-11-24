"""HNSW recall and latency of Qdrant against exact FAISS search (DECISIONS.md D24).

    python -m lookalike.index.hnsw_recall --config configs/index/qdrant.yaml

Needs a running Qdrant server (`qdrant_url`): local mode scans every vector, so it would always
report recall 1.0. Gallery = image vectors of the whole subset; queries = a seeded sample of
gallery items, no exclusion. For each `hnsw_ef`, recall@k = overlap of Qdrant's top-k with
FAISS's exact top-k. The row `exact` asks Qdrant for an exact search as a sanity check (~1.0).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.eval.metrics import bootstrap_ci
from lookalike.index.faiss_baseline import ann_recall, build_flat_index, search


def _timed_searches(client, collection: str, vectors: np.ndarray, k: int, **params):
    from qdrant_client import models

    ids, latencies = [], []
    for v in vectors:
        start = time.perf_counter()
        res = client.query_points(
            collection,
            query=v.tolist(),
            using="image",
            limit=k,
            search_params=models.SearchParams(**params),
        )
        latencies.append(time.perf_counter() - start)
        ids.append([int(p.id) for p in res.points])
    return ids, np.array(latencies)


def benchmark(
    client,
    collection: str,
    gallery_ids: np.ndarray,
    gallery_emb: np.ndarray,
    query_rows: np.ndarray,
    k: int,
    efs: list[int],
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per setting: recall@k (mean, 95% CI) and query latency p50/p95 in ms."""
    _, exact_rows = search(build_flat_index(gallery_emb), gallery_emb[query_rows], k)
    exact_ids = np.asarray(gallery_ids)[exact_rows]
    queries = np.asarray(gallery_emb[query_rows], dtype=np.float32)

    settings = [("exact", {"exact": True})] + [(str(ef), {"hnsw_ef": int(ef)}) for ef in efs]
    rows = []
    for label, params in settings:
        approx, lat = _timed_searches(client, collection, queries, k, **params)
        width = max(len(a) for a in approx)
        approx_arr = np.array([a + [-1] * (width - len(a)) for a in approx])
        mean, lo, hi = bootstrap_ci(ann_recall(approx_arr, exact_ids), n_boot=n_boot, seed=seed)
        rows.append(
            {
                "hnsw_ef": label,
                f"recall@{k}": mean,
                "ci_low": lo,
                "ci_high": hi,
                "latency_p50_ms": float(np.percentile(lat, 50) * 1000),
                "latency_p95_ms": float(np.percentile(lat, 95) * 1000),
                "n_queries": len(query_rows),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    from lookalike.index.qdrant_store import (
        connect,
        create_collection,
        load_vectors,
        upsert_items,
        wait_until_indexed,
    )

    parser = argparse.ArgumentParser(description="Qdrant HNSW recall vs exact FAISS.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    bench = cfg["hnsw_benchmark"]
    if not cfg.get("qdrant_url"):
        raise SystemExit("hnsw_recall needs a Qdrant server: set qdrant_url (see D24)")

    items = pd.read_parquet(cfg["subset_path"]).reset_index(drop=True)
    ids = items["article_id"].to_numpy()
    emb = load_vectors(Path(cfg["embeddings_root"]), cfg["dense_vectors"]["image"], ids)

    client = connect(cfg["qdrant_url"])
    hnsw = cfg["hnsw"]
    start = time.perf_counter()
    create_collection(
        client,
        bench["collection"],
        {"image": emb.shape[1]},
        hnsw_m=hnsw["m"],
        ef_construct=hnsw["ef_construct"],
        full_scan_threshold=hnsw.get("full_scan_threshold"),
        indexing_threshold=hnsw.get("indexing_threshold"),
    )
    upsert_items(client, bench["collection"], items, {"image": emb})
    state = wait_until_indexed(client, bench["collection"])
    build_s = time.perf_counter() - start
    print(f"Indexed {state} in {build_s:.1f}s")
    if not state["indexed_vectors_count"]:
        print("WARNING: no vectors are HNSW-indexed; recall would reflect a full scan (D24).")

    rng = np.random.default_rng(cfg["seed"])
    query_rows = rng.choice(len(ids), size=min(bench["n_queries"], len(ids)), replace=False)
    table = benchmark(
        client, bench["collection"], ids, emb, query_rows, bench["k"], bench["efs"],
        n_boot=cfg["n_boot"], seed=cfg["seed"],
    )  # fmt: skip
    for key, value in {
        "n_vectors": len(ids),
        "m": hnsw["m"],
        "ef_construct": hnsw["ef_construct"],
        "build_s": round(build_s, 2),
        "indexed_vectors_count": state["indexed_vectors_count"],
    }.items():
        table[key] = value
    out = Path(bench["results_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

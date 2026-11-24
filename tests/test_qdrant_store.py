"""Qdrant tests use the in-memory local client (no server needed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qdrant_client")

from lookalike.eval.fusion import rrf_fuse  # noqa: E402
from lookalike.index.qdrant_store import (  # noqa: E402
    connect,
    create_collection,
    search_batch,
    search_hybrid,
    upsert_items,
    wait_until_indexed,
)
from lookalike.index.sparse import BM25, tokenize  # noqa: E402

ITEMS = pd.DataFrame(
    {
        "article_id": [101, 102, 103, 104, 105, 106],
        "product_code": [1, 1, 2, 3, 4, 5],
        "product_type_name": ["Shirt", "Shirt", "Shirt", "Dress", "Dress", "Shirt"],
        "detail_desc": [
            "black cotton shirt",
            "red cotton shirt",
            "black linen shirt",
            "black cotton dress",
            "floral summer dress",
            "white shirt",
        ],
    }
)
IMAGE = np.array(
    [[1.0, 0.0], [0.98, 0.2], [0.9, 0.4], [0.6, 0.8], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32
)


@pytest.fixture()
def store():
    docs = [tokenize(t) for t in ITEMS["detail_desc"]]
    bm25 = BM25().fit(docs)
    client = connect()
    create_collection(client, "t", {"image": 2}, ("bm25",))
    upsert_items(client, "t", ITEMS, {"image": IMAGE}, {"bm25": [bm25.doc_vector(d) for d in docs]})
    return client, bm25, docs


def test_upload_and_status(store):
    client, _, _ = store
    state = wait_until_indexed(client, "t", timeout_s=5)
    assert state["points_count"] == 6


def test_dense_search_excludes_own_product_and_orders_by_cosine(store):
    client, _, _ = store
    got = search_batch(client, "t", "image", [IMAGE[0]], 3, exclude_codes=[1])
    assert got == [[103, 106, 104]]  # 102 is the same product as the query


def test_sparse_scores_equal_offline_bm25(store):
    """Qdrant's sparse dot product reproduces the offline BM25 score of every item (D21)."""
    client, bm25, docs = store
    q_tokens = tokenize("black cotton")
    idx, val = bm25.query_vector(q_tokens)
    from qdrant_client import models

    res = client.query_points(
        "t", query=models.SparseVector(indices=idx.tolist(), values=val.tolist()), using="bm25",
        limit=6,
    )  # fmt: skip
    got = {int(p.id): p.score for p in res.points}
    scores = bm25.score_matrix([q_tokens], docs)[0]
    expected = {int(a): s for a, s in zip(ITEMS.article_id, scores, strict=True) if s > 0}
    assert got.keys() == expected.keys()  # only items sharing a term are returned
    for a in got:
        assert got[a] == pytest.approx(expected[a], rel=1e-5)


def test_payload_filter_restricts_type(store):
    from lookalike.index.qdrant_store import build_filter

    client, _, _ = store
    res = client.query_points(
        "t",
        query=IMAGE[0].tolist(),
        using="image",
        limit=6,
        query_filter=build_filter(1, {"product_type_name": "Dress"}),
    )
    assert sorted(int(p.id) for p in res.points) == [104, 105]


def test_qdrant_hybrid_reproduces_offline_rrf(store):
    """Offline RRF (1-based, k=60) and Qdrant RRF (sent as k-1) give the same top list (D22)."""
    client, bm25, _ = store
    q_text = tokenize("black cotton")
    got = search_hybrid(
        client,
        "t",
        {"image": [IMAGE[0]], "bm25": [bm25.query_vector(q_text)]},
        k=4,
        exclude_codes=[1],
        rrf_k=60,
    )[0]
    dense = search_batch(client, "t", "image", [IMAGE[0]], 100, exclude_codes=[1])[0]
    sparse = search_batch(client, "t", "bm25", [bm25.query_vector(q_text)], 100, exclude_codes=[1])[
        0
    ]
    pos = {a: i for i, a in enumerate(ITEMS.article_id)}
    pad = max(len(dense), len(sparse))
    lists = [np.array([[pos[a] for a in r] + [-1] * (pad - len(r))]) for r in (dense, sparse)]
    offline = [int(ITEMS.article_id[i]) for i in rrf_fuse(lists, k=60, top_k=4)[0] if i >= 0]
    assert got == offline

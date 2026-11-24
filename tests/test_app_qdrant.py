import numpy as np
import pandas as pd
import pytest
from app.backend.qdrant_store import QdrantStore, ingest, make_client
from app.backend.search import BruteForceStore, items_table

N, DIM = 300, 16
TYPES = ["Dress", "Shirt", "Shoe", "Hat"]
COLOURS = ["Black", "Red", "Blue"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    ids = rng.choice(10**9, size=N, replace=False).astype(np.int64)
    emb = rng.normal(size=(N, DIM)).astype(np.float32)
    meta = pd.DataFrame(
        {
            "article_id": ids,
            "prod_name": [f"item {i}" for i in ids],
            "product_type_name": [TYPES[i % 4] for i in range(N)],
            "colour_group_name": [COLOURS[i % 3] for i in range(N)],
            "detail_desc": [f"desc {i}" for i in ids],
        }
    )
    return emb, ids, items_table(meta)


@pytest.fixture
def stores(data):
    emb, ids, items = data
    client, is_server = make_client()  # in memory
    assert not is_server
    ingest(client, "t", emb, ids, items)
    return QdrantStore(client, "t"), BruteForceStore(emb, ids, items)


@pytest.mark.parametrize("filters", [{}, {"product_type": "Dress"}, {"colour": "Red"},
                                     {"product_type": "Shoe", "colour": "Blue"}])  # fmt: skip
def test_qdrant_returns_the_same_answers_as_exact_brute_force(stores, filters):
    qdrant, exact = stores
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=DIM).astype(np.float32)
        got = qdrant.search(q, 10, **filters)
        want = exact.search(q, 10, **filters)
        assert [h.article_id for h in got] == [h.article_id for h in want]
        np.testing.assert_allclose([h.score for h in got], [h.score for h in want], atol=1e-4)
        assert [(h.name, h.product_type, h.colour, h.description) for h in got] == [
            (h.name, h.product_type, h.colour, h.description) for h in want
        ]


def test_filters_really_filter(stores):
    qdrant, _ = stores
    q = np.ones(DIM, dtype=np.float32)
    hits = qdrant.search(q, 50, product_type="Hat", colour="Black")
    assert hits and all(h.product_type == "Hat" and h.colour == "Black" for h in hits)


def test_ingest_twice_does_not_duplicate_and_recreate_resets(data):
    emb, ids, items = data
    client, _ = make_client()
    assert ingest(client, "t", emb, ids, items) == N
    assert ingest(client, "t", emb, ids, items) == N  # same ids overwrite themselves
    assert ingest(client, "t", emb[:10], ids[:10], items, recreate=True) == 10


def test_facets_match_the_loaded_items(stores):
    qdrant, exact = stores
    assert (
        qdrant.facets()
        == exact.facets()
        == {
            "product_type": sorted(TYPES),
            "colour": sorted(COLOURS),
        }
    )


def test_len_and_dim_come_from_the_collection(stores):
    qdrant, _ = stores
    assert len(qdrant) == N
    assert qdrant.dim == DIM


def test_missing_collection_error_says_how_to_fix_it():
    client, _ = make_client()
    with pytest.raises(LookupError, match="app.tools.ingest"):
        QdrantStore(client, "nope")


def test_bad_queries_are_rejected_before_reaching_qdrant(stores):
    qdrant, _ = stores
    with pytest.raises(ValueError):
        qdrant.search(np.ones(DIM + 1, dtype=np.float32), 5)
    with pytest.raises(ValueError):
        qdrant.search(np.zeros(DIM, dtype=np.float32), 5)
    with pytest.raises(ValueError):
        qdrant.search(np.ones(DIM, dtype=np.float32), 0)


def test_ingest_rejects_mismatched_inputs(data):
    emb, ids, items = data
    client, _ = make_client()
    with pytest.raises(ValueError):
        ingest(client, "t", emb[:5], ids[:6], items)
    with pytest.raises(KeyError):
        ingest(client, "t", emb[:2], np.array([1, 2]), items)  # ids not in items


def test_embedded_mode_persists_to_disk(data, tmp_path):
    emb, ids, items = data
    client, _ = make_client(path=tmp_path / "q")
    ingest(client, "t", emb[:20], ids[:20], items)
    client.close()
    reopened, _ = make_client(path=tmp_path / "q")
    assert len(QdrantStore(reopened, "t")) == 20

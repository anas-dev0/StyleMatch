import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from lookalike.config import load_config

pytest.importorskip("faiss")
pytest.importorskip("qdrant_client")

from lookalike.index import qdrant_store  # noqa: E402
from lookalike.index.hnsw_recall import benchmark  # noqa: E402


@pytest.mark.filterwarnings("ignore:Local mode performs exact")
def test_benchmark_on_local_qdrant_has_perfect_recall():
    """Local mode is a full scan, so every setting must match FAISS exactly."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, 8)).astype(np.float32)
    items = pd.DataFrame({"article_id": np.arange(5000, 5300), "product_code": np.arange(300)})
    client = qdrant_store.connect()
    qdrant_store.create_collection(client, "b", {"image": 8})
    qdrant_store.upsert_items(client, "b", items, {"image": emb})
    table = benchmark(client, "b", items.article_id.to_numpy(), emb, np.arange(25), 5, [4, 16], 50)
    assert list(table.hnsw_ef) == ["exact", "4", "16"]
    np.testing.assert_allclose(table["recall@5"], 1.0)
    assert (table.latency_p95_ms >= table.latency_p50_ms).all()


def test_collection_cli_builds_all_named_vectors_in_local_mode(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    items = pd.DataFrame(
        {
            "article_id": [1, 2, 3],
            "product_code": [10, 10, 20],
            "product_type_name": ["Shirt"] * 3,
            "detail_desc": ["black shirt", None, "white shirt"],
        }
    )
    items.to_parquet(tmp_path / "subset.parquet")
    for name in ("img", "txt"):
        (tmp_path / name).mkdir()
        np.save(tmp_path / name / "embeddings.npy", rng.normal(size=(3, 4)).astype(np.float32))
        np.save(tmp_path / name / "ids.npy", np.array([1, 2, 3]))
    cfg = load_config("configs/index/qdrant.yaml")
    cfg.update(
        subset_path=str(tmp_path / "subset.parquet"),
        embeddings_root=str(tmp_path),
        qdrant_url=None,
        dense_vectors={"image": "img", "text": "txt"},
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    built = {}
    real_connect = qdrant_store.connect

    def keep_client(url=None):
        built["client"] = real_connect(url)
        return built["client"]

    monkeypatch.setattr(qdrant_store, "connect", keep_client)
    monkeypatch.setattr(sys, "argv", ["qdrant_store", "--config", str(path)])
    qdrant_store.main()

    client = built["client"]
    points, _ = client.scroll(cfg["collection"], with_vectors=True, limit=10)
    assert len(points) == 3
    by_id = {int(p.id): p for p in points}
    assert set(by_id[1].vector) == {"image", "text", "bm25"}
    assert "bm25" not in by_id[2].vector  # missing description: no sparse vector
    assert by_id[1].payload["product_code"] == 10

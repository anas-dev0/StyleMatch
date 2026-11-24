import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from app.backend.api import create_app, thumbnail_url  # noqa: E402
from app.backend.qdrant_store import QdrantStore, ingest, make_client  # noqa: E402
from app.backend.search import BruteForceStore, items_table  # noqa: E402
from app.backend.settings import load_settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

META = pd.DataFrame(
    {
        "article_id": [10, 20, 30],
        "prod_name": ["right", "up", "diag"],
        "product_type_name": ["Dress", "Shoe", "Shirt"],
        "colour_group_name": ["Red", "Blue", "Black"],
        "detail_desc": ["a", "b", "c"],
    }
)


class FakeEncoder:
    """Maps a few known words to directions; records what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        return np.array([1.0, 0.0]) if "red" in text else np.array([0.0, 1.0])


EMB = np.array([[2.0, 0], [0, 5.0], [1, 1.0]])
IDS = np.array([10, 20, 30])


def make_store(kind):
    items = items_table(META)
    if kind == "brute":
        return BruteForceStore(EMB, IDS, items)
    qdrant, _ = make_client()
    ingest(qdrant, "t", EMB, IDS, items)
    return QdrantStore(qdrant, "t")


@pytest.fixture(params=["brute", "qdrant"])  # every API test runs against both stores
def client(tmp_path, request):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    (thumbs / "0000000010.jpg").write_bytes(b"fake-jpeg-bytes")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<html>home</html>", encoding="utf-8")
    encoder = FakeEncoder()
    app = create_app(
        make_store(request.param), encoder, thumbs, default_k=2, max_k=3, frontend_dir=frontend
    )
    c = TestClient(app)
    c.encoder = encoder
    return c


def test_thumbnail_url_is_zero_padded_to_ten_digits():
    assert thumbnail_url(108775015) == "/thumbs/0108775015.jpg"
    assert thumbnail_url(10) == "/thumbs/0000000010.jpg"


def test_health_reports_item_count(client):
    assert client.get("/health").json() == {"status": "ok", "n_items": 3}


def test_search_returns_ranked_results_with_urls_not_image_bytes(client):
    body = client.post("/search", json={"query": "  a red dress "}).json()
    assert client.encoder.seen == ["a red dress"]  # whitespace trimmed before encoding
    assert body["query"] == "a red dress"
    first = body["results"][0]
    assert first["article_id"] == 10
    assert first["rank"] == 1
    assert first["score"] == 1.0
    assert first["name"] == "right"
    assert first["image_url"] == "/thumbs/0000000010.jpg"
    assert len(body["results"]) == 2  # default_k
    assert [r["rank"] for r in body["results"]] == [1, 2]


def test_filters_narrow_the_results_and_facets_feed_the_dropdowns(client):
    body = client.post("/search", json={"query": "red", "colour": "Black"}).json()
    # Without the filter the best match for "red" is item 10 (Red); with it only item 30 qualifies.
    unfiltered = client.post("/search", json={"query": "red"}).json()
    assert unfiltered["results"][0]["article_id"] == 10
    assert [r["article_id"] for r in body["results"]] == [30]
    assert body["results"][0]["colour"] == "Black"
    none = client.post("/search", json={"query": "red", "product_type": "Dress", "colour": "Black"})
    assert none.json()["results"] == []
    assert client.get("/facets").json() == {
        "product_type": ["Dress", "Shirt", "Shoe"],
        "colour": ["Black", "Blue", "Red"],
    }


def test_k_is_honoured_and_capped_at_max_k(client):
    assert len(client.post("/search", json={"query": "red", "k": 1}).json()["results"]) == 1
    assert len(client.post("/search", json={"query": "red", "k": 999}).json()["results"]) == 3


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "   "}, {"query": "x" * 201}, {"query": "red", "k": 0}, {}],
)
def test_invalid_requests_get_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_min_score_drops_weak_matches_and_can_return_nothing(tmp_path):
    thumbs = tmp_path / "t"
    thumbs.mkdir()

    # "red" points along x: item 10 scores 1.0, item 30 scores 0.7071, item 20 scores 0.
    def run(min_score, query="red"):
        app = create_app(
            make_store("brute"), FakeEncoder(), thumbs, max_k=3, min_score=min_score,
            frontend_dir=tmp_path,
        )  # fmt: skip
        body = TestClient(app).post("/search", json={"query": query}).json()
        return [r["article_id"] for r in body["results"]]

    assert run(None) == [10, 30, 20]  # no threshold keeps everything
    assert run(0.5) == [10, 30]
    assert run(0.7071) == [10, 30]  # a score equal to the threshold is kept
    assert run(1.5) == []  # nothing is good enough: empty list, not an error


def test_thumbnail_is_served_as_a_static_file(client):
    res = client.get("/thumbs/0000000010.jpg")
    assert res.status_code == 200
    assert res.content == b"fake-jpeg-bytes"
    assert client.get("/thumbs/0000000999.jpg").status_code == 404


def test_home_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "home" in res.text


def test_settings_env_override_picks_the_data_dir():
    settings = load_settings("configs/app/app.yaml", env={"LOOKALIKE_DATA_DIR": "D:/elsewhere"})
    assert str(settings["data_dir"]).replace("\\", "/") == "D:/elsewhere"
    default = load_settings("configs/app/app.yaml", env={})
    assert default["qdrant"] == {"url": None, "collection": "fashion_items"}
    with_url = load_settings("configs/app/app.yaml", env={"LOOKALIKE_QDRANT_URL": "http://q:6333"})
    assert with_url["qdrant"]["url"] == "http://q:6333"
    assert default["data_dir"].name == "app_data"
    assert default["min_score"] == 0.275
    assert default["model"]["revision"] == "7e3ba62ce16b379a1ab479346b66f192e76f51b7"

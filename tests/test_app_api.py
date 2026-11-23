import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from app.backend.api import create_app, thumbnail_url  # noqa: E402
from app.backend.search import SearchIndex  # noqa: E402
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


@pytest.fixture
def client(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    (thumbs / "0000000010.jpg").write_bytes(b"fake-jpeg-bytes")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<html>home</html>", encoding="utf-8")
    index = SearchIndex(np.array([[2.0, 0], [0, 5.0], [1, 1.0]]), np.array([10, 20, 30]), META)
    encoder = FakeEncoder()
    app = create_app(index, encoder, thumbs, default_k=2, max_k=3, frontend_dir=frontend)
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


def test_k_is_honoured_and_capped_at_max_k(client):
    assert len(client.post("/search", json={"query": "red", "k": 1}).json()["results"]) == 1
    assert len(client.post("/search", json={"query": "red", "k": 999}).json()["results"]) == 3


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "   "}, {"query": "x" * 201}, {"query": "red", "k": 0}, {}],
)
def test_invalid_requests_get_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


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
    assert default["data_dir"].name == "app_data"
    assert default["model"]["revision"] == "7e3ba62ce16b379a1ab479346b66f192e76f51b7"

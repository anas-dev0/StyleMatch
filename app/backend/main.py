"""Builds the real app (loads the model and the index). Start it from the repo root with:

    python -m app --data-dir C:\\lookalike_data\\app_data --hf-home C:\\lookalike_data\\hf_cache

then open http://127.0.0.1:8000. Load the vectors into Qdrant first: python -m app.tools.ingest
"""

from app.backend.api import create_app
from app.backend.encoder import TextEncoder
from app.backend.qdrant_store import QdrantStore, make_client
from app.backend.settings import load_settings


def build_app():
    settings = load_settings()
    data_dir = settings["data_dir"]
    thumbs_dir = data_dir / "thumbs"
    if not thumbs_dir.is_dir():
        raise FileNotFoundError(f"thumbnail folder not found: {thumbs_dir.resolve()}")
    qd = settings["qdrant"]
    client, _ = make_client(qd.get("url"), data_dir / "qdrant")
    store = QdrantStore(client, qd["collection"])  # fails clearly if `ingest` was not run
    model = settings["model"]
    encoder = TextEncoder(model["hf_id"], model.get("revision"))
    return create_app(
        store,
        encoder,
        thumbs_dir,
        default_k=settings["default_k"],
        max_k=settings["max_k"],
        min_score=settings.get("min_score"),
    )

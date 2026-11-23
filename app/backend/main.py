"""Builds the real app (loads the model and the index). Start it from the repo root with:

    python -m uvicorn app.backend.main:build_app --factory --port 8000

then open http://127.0.0.1:8000
"""

from app.backend.api import create_app
from app.backend.encoder import TextEncoder
from app.backend.search import load_index
from app.backend.settings import load_settings


def build_app():
    settings = load_settings()
    data_dir = settings["data_dir"]
    thumbs_dir = data_dir / "thumbs"
    if not thumbs_dir.is_dir():
        raise FileNotFoundError(f"thumbnail folder not found: {thumbs_dir.resolve()}")
    index = load_index(data_dir)
    model = settings["model"]
    encoder = TextEncoder(model["hf_id"], model.get("revision"))
    return create_app(
        index,
        encoder,
        thumbs_dir,
        default_k=settings["default_k"],
        max_k=settings["max_k"],
    )

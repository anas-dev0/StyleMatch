"""The HTTP API. Results carry image URLs; the browser fetches the images from /thumbs/...

`encoder` is anything with `.encode(text) -> vector`, so tests can plug in a fake one.
"""

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.backend.search import SearchIndex

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def thumbnail_url(article_id: int) -> str:
    """Where the browser can fetch an item's thumbnail. The only place that knows the URL scheme,
    so moving images to MinIO/S3 later means changing this one function."""
    return f"/thumbs/{article_id:010d}.jpg"


class SearchRequest(BaseModel):
    query: str = Field(max_length=200)
    k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class SearchResult(BaseModel):
    rank: int
    article_id: int
    score: float
    name: str
    product_type: str
    colour: str
    description: str
    image_url: str


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    results: list[SearchResult]


def create_app(
    index: SearchIndex,
    encoder,
    thumbs_dir: str | Path,
    *,
    default_k: int = 12,
    max_k: int = 50,
    frontend_dir: str | Path = FRONTEND_DIR,
) -> FastAPI:
    app = FastAPI(title="LookAlike text search")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_items": len(index)}

    # A plain `def` (not `async def`): encoding blocks, so FastAPI runs it in a worker thread
    # and other requests are not frozen while one query is being encoded.
    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        start = time.perf_counter()
        k = min(req.k or default_k, max_k)
        hits = index.search(encoder.encode(req.query), k)
        results = [
            SearchResult(
                rank=rank,
                article_id=hit.article_id,
                score=round(hit.score, 4),
                image_url=thumbnail_url(hit.article_id),
                **index.describe(hit),
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        took = round((time.perf_counter() - start) * 1000, 1)
        return SearchResponse(query=req.query, took_ms=took, results=results)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(Path(frontend_dir) / "index.html")

    app.mount("/thumbs", StaticFiles(directory=str(thumbs_dir)), name="thumbs")
    return app

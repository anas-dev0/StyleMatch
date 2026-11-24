"""Qdrant as the vector database: stores each item's vector with its display fields as payload,
searches by cosine similarity and can filter by product type / colour inside the search.

Two ways to connect (`make_client`): a Qdrant server (Docker, `url`) or embedded storage in a
local folder (`path`, no server needed). Tests use in-memory mode.
"""

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from app.backend.search import TEXT_COLUMNS, Hit, check_query

FILTER_FIELDS = ("product_type", "colour")


def make_client(url: str | None = None, path=None) -> tuple[QdrantClient, bool]:
    """(client, is_server). `url` -> a Qdrant server; else `path` -> embedded; else in memory."""
    if url:
        return QdrantClient(url=url), True
    return QdrantClient(path=str(path) if path else ":memory:"), False


def ingest(
    client: QdrantClient,
    collection: str,
    embeddings: np.ndarray,
    ids: np.ndarray,
    items: pd.DataFrame,
    *,
    recreate: bool = False,
    create_indexes: bool = False,
    batch_size: int = 512,
) -> int:
    """Write every item as a point (id = article_id). Safe to rerun: same id overwrites itself,
    so nothing is duplicated. `recreate` empties the collection first. Returns the point count.

    `create_indexes` adds payload indexes for the filter fields; they only matter on a server
    (the embedded client ignores them and warns), so callers set it when `is_server`.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    if len(embeddings) != len(ids):
        raise ValueError(f"{len(embeddings)} embeddings but {len(ids)} ids")
    meta = items.set_index("article_id").loc[ids][TEXT_COLUMNS]  # KeyError if an id has no row

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=embeddings.shape[1], distance=models.Distance.COSINE
            ),
        )
        if create_indexes:
            for field in FILTER_FIELDS:
                client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)

    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        points = [
            models.PointStruct(id=int(article_id), vector=vector.tolist(), payload=payload)
            for article_id, vector, payload in zip(
                ids[start:stop],
                embeddings[start:stop],
                meta.iloc[start:stop].to_dict("records"),
                strict=True,
            )
        ]
        client.upsert(collection, points=points)
    return client.count(collection, exact=True).count


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        if not client.collection_exists(collection):
            raise LookupError(
                f"Qdrant collection '{collection}' does not exist. "
                "Create it first: python -m app.tools.ingest"
            )
        self.client = client
        self.collection = collection
        self.dim = client.get_collection(collection).config.params.vectors.size
        self._facets: dict[str, list[str]] | None = None

    def __len__(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        product_type: str | None = None,
        colour: str | None = None,
    ) -> list[Hit]:
        q = check_query(query_vec, self.dim, k)
        conditions = [
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for field, value in (("product_type", product_type), ("colour", colour))
            if value is not None
        ]
        response = self.client.query_points(
            self.collection,
            query=q.tolist(),
            limit=k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        return [
            Hit(int(p.id), float(p.score), **{c: p.payload.get(c, "") for c in TEXT_COLUMNS})
            for p in response.points
        ]

    def facets(self) -> dict[str, list[str]]:
        """Distinct product types and colours present (read once from the payloads, then cached)."""
        if self._facets is None:
            found: dict[str, set[str]] = {field: set() for field in FILTER_FIELDS}
            offset = None
            while True:
                points, offset = self.client.scroll(
                    self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=list(FILTER_FIELDS),
                    with_vectors=False,
                )
                for p in points:
                    for field in FILTER_FIELDS:
                        value = p.payload.get(field)
                        if value:
                            found[field].add(value)
                if offset is None:
                    break
            self._facets = {field: sorted(values) for field, values in found.items()}
        return self._facets

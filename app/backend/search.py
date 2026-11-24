"""Item records and an exact (brute-force) vector search.

The app talks to a "store": anything with `search(query_vec, k, product_type, colour)`,
`facets()` and `len()`. `QdrantStore` (qdrant_store.py) is the real one. `BruteForceStore` here
computes every similarity with NumPy. It is the reference the Qdrant results are tested
against, and a fallback for tiny setups.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {
    "prod_name": "name",
    "product_type_name": "product_type",
    "colour_group_name": "colour",
    "detail_desc": "description",
}
TEXT_COLUMNS = ["name", "product_type", "colour", "description"]


@dataclass(frozen=True)
class Hit:
    article_id: int
    score: float  # cosine similarity, between -1 and 1
    name: str
    product_type: str
    colour: str
    description: str


def items_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Display fields with the app's column names, one row per article, missing text as ''."""
    items = metadata.rename(columns=RENAME).drop_duplicates("article_id")
    items = items.reindex(columns=["article_id", *TEXT_COLUMNS])
    items["article_id"] = items["article_id"].astype(np.int64)
    items[TEXT_COLUMNS] = items[TEXT_COLUMNS].fillna("").astype(str)
    return items.reset_index(drop=True)


def read_export(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(embeddings, ids, items) from the files made by `app.tools.export_assets`."""
    data_dir = Path(data_dir)
    files = {name: data_dir / name for name in ("embeddings.npy", "ids.npy", "metadata.csv")}
    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found in {data_dir.resolve()}. Run app.tools.export_assets on Kaggle, "
            "download the result, and point data_dir (or LOOKALIKE_DATA_DIR) at it."
        )
    items = items_table(pd.read_csv(files["metadata.csv"]))
    return np.load(files["embeddings.npy"]), np.load(files["ids.npy"]), items


def unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product between two rows is their cosine."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def check_query(query_vec: np.ndarray, dim: int, k: int) -> np.ndarray:
    """Validate a query and return it as a flat float32 vector. Shared by every store."""
    if k < 1:
        raise ValueError("k must be at least 1")
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    if q.shape[0] != dim:
        raise ValueError(f"query has {q.shape[0]} numbers, the index has {dim}")
    if not np.any(q):
        raise ValueError("query vector is all zeros")
    return q


class BruteForceStore:
    def __init__(self, embeddings: np.ndarray, article_ids: np.ndarray, items: pd.DataFrame):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        article_ids = np.asarray(article_ids, dtype=np.int64)
        if embeddings.ndim != 2 or len(embeddings) != len(article_ids):
            raise ValueError(f"{embeddings.shape} embeddings do not match {len(article_ids)} ids")
        if len(article_ids) == 0:
            raise ValueError("the index is empty")
        if len(set(article_ids.tolist())) != len(article_ids):
            raise ValueError("duplicate article ids in the index")
        missing = set(article_ids.tolist()) - set(items["article_id"].tolist())
        if missing:
            raise KeyError(f"{len(missing)} items have no metadata, e.g. {sorted(missing)[:3]}")
        self._emb = unit(embeddings)
        self.article_ids = article_ids
        self._items = items.set_index("article_id").loc[article_ids][TEXT_COLUMNS]

    def __len__(self) -> int:
        return len(self.article_ids)

    @property
    def dim(self) -> int:
        return self._emb.shape[1]

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        product_type: str | None = None,
        colour: str | None = None,
    ) -> list[Hit]:
        """The k most similar items (best first) among those matching the optional filters."""
        q = check_query(query_vec, self.dim, k)
        allowed = np.ones(len(self), dtype=bool)
        if product_type is not None:
            allowed &= (self._items["product_type"] == product_type).to_numpy()
        if colour is not None:
            allowed &= (self._items["colour"] == colour).to_numpy()
        k = min(k, int(allowed.sum()))
        if k == 0:
            return []
        sims = self._emb @ unit(q)
        sims[~allowed] = -np.inf
        top = np.argpartition(-sims, k - 1)[:k]  # the k best, unordered
        top = top[np.argsort(-sims[top], kind="stable")]  # now best first
        return [
            Hit(int(self.article_ids[i]), float(sims[i]), **self._items.iloc[i].to_dict())
            for i in top
        ]

    def facets(self) -> dict[str, list[str]]:
        return {
            "product_type": sorted(self._items["product_type"].unique()),
            "colour": sorted(self._items["colour"].unique()),
        }

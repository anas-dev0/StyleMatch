"""Nearest-neighbour search over precomputed image embeddings (no model needed here)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_COLUMNS = ["prod_name", "product_type_name", "colour_group_name", "detail_desc"]


def _unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product between two rows is their cosine."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


@dataclass(frozen=True)
class Hit:
    article_id: int
    score: float  # cosine similarity, between -1 and 1
    row: int  # position in the index, used to look up the item's text fields


class SearchIndex:
    def __init__(self, embeddings: np.ndarray, article_ids: np.ndarray, metadata: pd.DataFrame):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        article_ids = np.asarray(article_ids, dtype=np.int64)
        if embeddings.ndim != 2 or len(embeddings) != len(article_ids):
            raise ValueError(f"{embeddings.shape} embeddings do not match {len(article_ids)} ids")
        if len(article_ids) == 0:
            raise ValueError("the index is empty")
        if len(set(article_ids.tolist())) != len(article_ids):
            raise ValueError("duplicate article ids in the index")
        missing = set(article_ids.tolist()) - set(metadata["article_id"].tolist())
        if missing:
            raise KeyError(
                f"{len(missing)} indexed items have no metadata, e.g. {sorted(missing)[:3]}"
            )
        self._emb = _unit(embeddings)
        self.article_ids = article_ids
        meta = metadata.drop_duplicates("article_id").set_index("article_id").loc[article_ids]
        self._meta = meta.reindex(columns=TEXT_COLUMNS).fillna("")

    def __len__(self) -> int:
        return len(self.article_ids)

    @property
    def dim(self) -> int:
        return self._emb.shape[1]

    def search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        """The k items most similar to `query_vec`, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"query has {q.shape[0]} numbers, the index has {self.dim}")
        if not np.any(q):
            raise ValueError("query vector is all zeros")
        sims = self._emb @ _unit(q)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]  # the k best, unordered
        top = top[np.argsort(-sims[top], kind="stable")]  # now best first
        return [Hit(int(self.article_ids[i]), float(sims[i]), int(i)) for i in top]

    def describe(self, hit: Hit) -> dict:
        row = self._meta.iloc[hit.row]
        return {
            "name": str(row["prod_name"]),
            "product_type": str(row["product_type_name"]),
            "colour": str(row["colour_group_name"]),
            "description": str(row["detail_desc"]),
        }


def load_index(data_dir: str | Path) -> SearchIndex:
    """Build the index from the files made by `app.tools.export_assets`."""
    data_dir = Path(data_dir)
    files = {name: data_dir / name for name in ("embeddings.npy", "ids.npy", "metadata.csv")}
    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found in {data_dir.resolve()}. Run app.tools.export_assets on Kaggle, "
            "download the result, and point data_dir (or LOOKALIKE_DATA_DIR) at it."
        )
    metadata = pd.read_csv(files["metadata.csv"])
    return SearchIndex(np.load(files["embeddings.npy"]), np.load(files["ids.npy"]), metadata)

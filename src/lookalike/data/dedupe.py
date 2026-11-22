"""Near-duplicate detection from image embeddings (cosine similarity)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config


def near_duplicate_pairs(
    embeddings: np.ndarray, threshold: float = 0.97, block_size: int = 2048
) -> np.ndarray:
    """Return an (m, 2) int array of row-index pairs (i < j) with cosine similarity > threshold.

    Rows are L2-normalised first, so a dot product equals cosine similarity. The similarity
    matrix is computed one block of rows at a time to bound memory.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    pairs = []
    for start in range(0, len(x), block_size):
        sims = x[start : start + block_size] @ x.T
        rows, cols = np.nonzero(sims > threshold)
        rows = rows + start
        keep = cols > rows  # upper triangle only: drops self-pairs and mirrored pairs
        pairs.append(np.stack([rows[keep], cols[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find near-duplicate article pairs.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    emb = np.load(cfg["embeddings_path"])
    ids = np.load(cfg["embedding_ids_path"])
    idx = near_duplicate_pairs(emb, cfg["near_dup_threshold"])
    out = pd.DataFrame({"article_id_a": ids[idx[:, 0]], "article_id_b": ids[idx[:, 1]]})
    path = Path(cfg["near_dup_pairs_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"{len(out)} near-duplicate pairs (threshold {cfg['near_dup_threshold']}). Wrote {path}")


if __name__ == "__main__":
    main()

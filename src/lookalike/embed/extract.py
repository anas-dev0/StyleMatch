"""Resumable embedding extraction.

Embeds the subset in shards, one .npy file per shard, so a crashed Kaggle session can resume
(`run_extraction` skips shards that are already complete). The model-specific part is passed in
as a function, which keeps this logic testable without torch.

    python -m lookalike.embed.extract --config configs/embed/baselines.yaml --model clip
    python -m lookalike.embed.extract --config configs/embed/baselines.yaml --model clip --limit 64
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.data.subset import image_path


def shard_bounds(n: int, shard_size: int) -> list[tuple[int, int]]:
    """[(start, stop), ...] covering range(n) in consecutive pieces of at most shard_size."""
    return [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]


def _shard_file(out_dir: Path, i: int) -> Path:
    return out_dir / f"shard_{i:05d}.npy"


def run_extraction(ids, encode, out_dir, shard_size: int = 2048, log=print) -> np.ndarray:
    """Embed `ids` shard by shard and return all embeddings in the order of `ids`.

    `encode(ids_in_shard) -> array (len(ids_in_shard), dim)`. A shard file that already exists
    with the right number of rows is reused; anything else is recomputed. Files are written under
    a temporary name and renamed, so a crash never leaves a half-written shard that looks done.
    """
    ids = np.asarray(ids)
    if len(ids) == 0:
        raise ValueError("no ids to embed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = shard_bounds(len(ids), shard_size)
    for i, (start, stop) in enumerate(bounds):
        path = _shard_file(out_dir, i)
        if path.exists() and np.load(path, mmap_mode="r").shape[0] == stop - start:
            log(f"shard {i + 1}/{len(bounds)}: already done, skipping")
            continue
        emb = np.asarray(encode(ids[start:stop]), dtype=np.float32)
        if emb.shape[0] != stop - start:
            raise ValueError(f"encoder returned {emb.shape[0]} rows for {stop - start} ids")
        tmp = out_dir / f"shard_{i:05d}.tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, path)
        log(f"shard {i + 1}/{len(bounds)}: done ({stop - start} items)")

    return np.concatenate([np.load(_shard_file(out_dir, i)) for i in range(len(bounds))])


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache image embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="a key under `models:` in the config")
    parser.add_argument("--limit", type=int, default=None, help="smoke test on the first N items")
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = cfg["models"][args.model]

    from lookalike.embed.models import ImageEncoder

    subset = pd.read_parquet(cfg["subset_path"])
    ids = subset["article_id"].to_numpy()
    out_root = cfg["out_root"]
    if args.limit:
        ids = ids[: args.limit]
        out_root += "_smoke"  # never mix smoke-test shards with real ones
    out_dir = Path(out_root) / args.model

    encoder = ImageEncoder(spec)
    batch_size = cfg["batch_size"]

    def encode(batch_ids: np.ndarray) -> np.ndarray:
        paths = [image_path(cfg["images_dir"], a) for a in batch_ids]
        parts = []
        with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as pool:
            for s in range(0, len(paths), batch_size):
                images = list(pool.map(_load_rgb, paths[s : s + batch_size]))
                parts.append(encoder.encode(images))
        return np.concatenate(parts)

    emb = run_extraction(ids, encode, out_dir, cfg["shard_size"])
    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "ids.npy", ids.astype(np.int64))
    meta = {**encoder.meta, "n_items": int(len(ids)), "dim": int(emb.shape[1])}
    meta["date"] = date.today().isoformat()
    meta["smoke_test"] = bool(args.limit)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{args.model}: {emb.shape} saved to {out_dir}")


if __name__ == "__main__":
    main()

"""Prepare the app's data folder. Run it on Kaggle, where the original images are:

    python -m app.tools.export_assets --config configs/app/app.yaml --out /kaggle/working/app_data

Writes thumbs/<article_id>.jpg (small previews), metadata.csv (names for display),
embeddings.npy and ids.npy (FashionCLIP image vectors). Safe to rerun: finished thumbnails are
skipped. The result is derived from H&M images: keep it local, never publish it (DECISIONS D6).
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.data.subset import image_path

METADATA_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "detail_desc",
]


def thumbnail_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Size after shrinking so the longest side is `max_side`. Never enlarges an image."""
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def make_thumbnail(src: Path, dst: Path, max_side: int, quality: int) -> None:
    """Write a shrunken JPEG copy of `src`. Written under a temp name then renamed, so a crash
    never leaves a half-written file that looks finished."""
    from PIL import Image

    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(thumbnail_size(*img.size, max_side), Image.Resampling.LANCZOS)
        tmp = dst.with_name(dst.name + ".tmp")
        img.save(tmp, format="JPEG", quality=quality, optimize=True)
    os.replace(tmp, dst)


def build_metadata(articles: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Display fields for `ids`, in that order. Raises KeyError if an id is not in `articles`."""
    by_id = articles.drop_duplicates("article_id").set_index("article_id")
    missing = set(map(int, ids)) - set(by_id.index)
    if missing:
        raise KeyError(f"{len(missing)} ids not in articles.csv, e.g. {sorted(missing)[:3]}")
    out = by_id.loc[ids, METADATA_COLUMNS[1:]].reset_index()
    out["article_id"] = out["article_id"].astype(np.int64)
    return out[METADATA_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thumbnails, metadata and vectors.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out", required=True, help="folder to create, e.g. /kaggle/working/app_data"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    ex = cfg["export"]
    out = Path(args.out)
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)

    ids = pd.read_parquet(ex["subset_path"])["article_id"].to_numpy().astype(np.int64)
    emb_dir = Path(ex["embeddings_dir"])
    emb_ids = np.load(emb_dir / "ids.npy")
    if not np.array_equal(ids, emb_ids):
        raise ValueError("embedding ids differ from the subset: they must be the same, same order")

    articles = pd.read_csv(Path(ex["raw_dir"]) / "articles.csv", usecols=METADATA_COLUMNS)
    build_metadata(articles, ids).to_csv(out / "metadata.csv", index=False)

    def job(article_id: int) -> None:
        dst = thumbs / f"{article_id:010d}.jpg"
        if not dst.exists():
            make_thumbnail(
                image_path(ex["images_dir"], article_id),
                dst,
                ex["thumb_max_side"],
                ex["jpeg_quality"],
            )

    with ThreadPoolExecutor(max_workers=ex["num_workers"]) as pool:
        for done, _ in enumerate(pool.map(job, ids), start=1):
            if done % 2000 == 0:
                print(f"thumbnails: {done}/{len(ids)}")

    shutil.copy(emb_dir / "embeddings.npy", out / "embeddings.npy")
    np.save(out / "ids.npy", ids)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Wrote {len(ids)} items to {out}: {size / 1e6:.0f} MB in total")


if __name__ == "__main__":
    main()

"""Build the working subset of the H&M catalogue.

Steps: keep needed columns, clean attributes, drop non-fashion groups/types, drop items
without an image, drop product types that are too small to split, cap each type so no
type dominates.
"""

import argparse
from pathlib import Path

import pandas as pd

from lookalike.config import load_config

KEY_COLS = [
    "article_id",
    "product_code",
    "product_type_name",
    "colour_group_name",
    "graphical_appearance_name",
]
EXTRA_COLS = ["product_group_name", "department_name", "garment_group_name", "detail_desc"]


def image_path(images_dir: str | Path, article_id: int) -> Path:
    """H&M layout: images/<first 3 digits of the 10-digit id>/<10-digit id>.jpg."""
    padded = f"{int(article_id):010d}"
    return Path(images_dir) / padded[:3] / f"{padded}.jpg"


def _exclude(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    """Drop rows whose `column` equals any of `values` (case-insensitive, whitespace-trimmed)."""
    if not values:
        return df
    if column not in df.columns:
        raise KeyError(f"cannot exclude on '{column}': column not found")
    drop = {v.strip().lower() for v in values}
    cleaned = df[column].astype("string").str.strip().str.lower()
    return df[~cleaned.isin(drop)]


def build_subset(
    articles: pd.DataFrame,
    *,
    cap_per_type: int,
    min_per_type: int,
    seed: int,
    has_image=None,
    drop_unknown: bool = True,
    exclude_groups=(),
    exclude_types=(),
) -> pd.DataFrame:
    """Return the cleaned, capped subset. `has_image(article_id) -> bool` filters rows."""
    cols = KEY_COLS + [c for c in EXTRA_COLS if c in articles.columns]
    df = articles[cols].copy()

    for col in KEY_COLS[2:]:
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=KEY_COLS)
    if drop_unknown:
        for col in KEY_COLS[2:]:
            df = df[df[col].str.lower() != "unknown"]

    df = _exclude(df, "product_group_name", exclude_groups)
    df = _exclude(df, "product_type_name", exclude_types)

    if has_image is not None:
        df = df[df["article_id"].map(has_image)]

    counts = df["product_type_name"].value_counts()
    df = df[df["product_type_name"].isin(counts[counts >= min_per_type].index)]

    parts = [
        group.sample(n=min(len(group), cap_per_type), random_state=seed)
        for _, group in df.groupby("product_type_name", sort=True)
    ]
    return pd.concat(parts).sort_values("article_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned, capped subset.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    articles = pd.read_csv(Path(cfg["raw_dir"]) / "articles.csv")
    images_dir = cfg["images_dir"]
    subset = build_subset(
        articles,
        cap_per_type=cfg["cap_per_type"],
        min_per_type=cfg["min_per_type"],
        seed=cfg["seed"],
        has_image=lambda a: image_path(images_dir, a).exists(),
        exclude_groups=cfg.get("exclude_product_groups", []),
        exclude_types=cfg.get("exclude_product_types", []),
    )
    out = Path(cfg["subset_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)
    print(
        f"{len(articles)} articles -> {len(subset)} in subset, "
        f"{subset['product_type_name'].nunique()} types, "
        f"{subset['product_code'].nunique()} products. Wrote {out}"
    )


if __name__ == "__main__":
    main()

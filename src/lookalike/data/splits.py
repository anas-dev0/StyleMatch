"""Group-aware train/val/test splits plus an unseen-type test.

Rules (see PLAN.md section 5 and CLAUDE.md rule 5):
- Everything sharing a `product_code` is one group (colour variants of one product).
- Items linked by a near-duplicate pair are merged into the same group.
- ~20% of product types are held out entirely: their groups form `test_unseen`.
- Groups mixing held-out and seen types are dropped, so `test_unseen` stays clean.
- Remaining groups are assigned whole to train/val/test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config

SPLITS = ("train", "val", "test", "test_unseen")


def _group_ids(n: int, product_codes: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """Union-find: connect rows with equal product_code and rows joined by `edges`."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_row_of_code: dict = {}
    for i, code in enumerate(product_codes):
        if code in first_row_of_code:
            union(i, first_row_of_code[code])
        else:
            first_row_of_code[code] = i
    for a, b in edges:
        union(a, b)
    return np.array([find(i) for i in range(n)])


def make_splits(
    df: pd.DataFrame,
    near_dup_pairs: pd.DataFrame | None = None,
    *,
    seed: int,
    holdout_type_frac: float = 0.2,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> pd.DataFrame:
    """Return rows [article_id, product_code, product_type_name, split]; some may be dropped."""
    df = df[["article_id", "product_code", "product_type_name"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    row_of = {a: i for i, a in enumerate(df["article_id"])}
    edges = []
    if near_dup_pairs is not None:
        pairs = zip(near_dup_pairs["article_id_a"], near_dup_pairs["article_id_b"], strict=True)
        for a, b in pairs:
            if a in row_of and b in row_of:
                edges.append((row_of[a], row_of[b]))
    group = _group_ids(len(df), df["product_code"].to_numpy(), edges)

    types = np.array(sorted(df["product_type_name"].unique()))
    n_hold = max(1, round(holdout_type_frac * len(types)))
    held_types = set(rng.permutation(types)[:n_hold])
    is_held = df["product_type_name"].isin(held_types).to_numpy()

    df["split"] = None
    seen_groups = []
    for g in np.unique(group):
        members = np.nonzero(group == g)[0]
        held = is_held[members]
        if held.all():
            df.loc[members, "split"] = "test_unseen"
        elif not held.any():
            seen_groups.append(members)
        # mixed group: stays None and is dropped

    order = rng.permutation(len(seen_groups))
    total = sum(len(m) for m in seen_groups)
    bounds = (ratios[0] * total, (ratios[0] + ratios[1]) * total)
    done = 0
    for k in order:
        members = seen_groups[k]
        name = "train" if done < bounds[0] else "val" if done < bounds[1] else "test"
        df.loc[members, "split"] = name
        done += len(members)

    return df.dropna(subset=["split"]).reset_index(drop=True)


def check_no_leakage(splits: pd.DataFrame) -> None:
    """Raise AssertionError if any leakage guarantee is broken."""
    codes = splits.groupby("product_code")["split"].nunique()
    assert (codes == 1).all(), f"product_code in several splits: {list(codes[codes > 1].index[:5])}"

    unseen_types = set(splits.loc[splits["split"] == "test_unseen", "product_type_name"])
    in_other = splits[splits["split"] != "test_unseen"]["product_type_name"]
    clash = unseen_types & set(in_other)
    assert not clash, f"held-out types also appear in other splits: {sorted(clash)[:5]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-free splits (ids only).")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    subset = pd.read_parquet(cfg["subset_path"])
    pairs_path = Path(cfg["near_dup_pairs_path"])
    pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else None
    if pairs is None:
        print("WARNING: no near-duplicate pairs file found; splitting by product_code only.")

    splits = make_splits(
        subset,
        pairs,
        seed=cfg["seed"],
        holdout_type_frac=cfg["holdout_type_frac"],
        ratios=tuple(cfg["ratios"]),
    )
    # check_no_leakage needs the type column, so run it before selecting output columns
    check_no_leakage(splits)
    out = Path(cfg["splits_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    splits[["article_id", "product_code", "split"]].to_csv(out, index=False)
    print(f"Kept {len(splits)} of {len(subset)} items ({len(subset) - len(splits)} dropped).")
    print(splits["split"].value_counts().to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

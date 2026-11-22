import pandas as pd
import pytest

from lookalike.data.splits import check_no_leakage, make_splits


def synthetic_catalogue(n_types=10, products_per_type=20, variants=3) -> pd.DataFrame:
    """Each product has several colour variants sharing one product_code."""
    rows = []
    article_id = 0
    for t in range(n_types):
        for p in range(products_per_type):
            for _ in range(variants):
                rows.append(
                    {
                        "article_id": article_id,
                        "product_code": t * 1000 + p,
                        "product_type_name": f"type{t}",
                    }
                )
                article_id += 1
    return pd.DataFrame(rows)


def test_no_product_code_in_two_splits_and_no_type_clash():
    splits = make_splits(synthetic_catalogue(), seed=0)
    check_no_leakage(splits)  # raises if broken


def test_splitting_by_article_id_would_leak():
    """Documents WHY we group: a naive row split puts variants in different splits."""
    df = synthetic_catalogue()
    df["split"] = ["train" if i % 2 == 0 else "test" for i in range(len(df))]
    with pytest.raises(AssertionError):
        check_no_leakage(df)


def test_held_out_types_are_about_20_percent_and_only_in_test_unseen():
    splits = make_splits(synthetic_catalogue(n_types=10), seed=0, holdout_type_frac=0.2)
    unseen = splits[splits["split"] == "test_unseen"]
    assert unseen["product_type_name"].nunique() == 2  # round(0.2 * 10)
    others = splits[splits["split"] != "test_unseen"]
    assert not set(unseen["product_type_name"]) & set(others["product_type_name"])


def test_ratios_are_roughly_respected_on_seen_types():
    splits = make_splits(synthetic_catalogue(n_types=10), seed=0)
    seen = splits[splits["split"] != "test_unseen"]
    share = seen["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.05)
    assert share["val"] == pytest.approx(0.1, abs=0.05)
    assert share["test"] == pytest.approx(0.2, abs=0.05)


def test_same_seed_same_result_different_seed_different_result():
    df = synthetic_catalogue()
    a = make_splits(df, seed=1)
    b = make_splits(df, seed=1)
    c = make_splits(df, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_near_duplicate_pair_ends_up_in_the_same_split():
    df = synthetic_catalogue(n_types=10)
    base = make_splits(df, seed=0)
    seen = base[base["split"] != "test_unseen"]
    a = seen.iloc[0]["article_id"]
    # pick an article that is in a DIFFERENT split than `a`, so the test is meaningful
    b = seen[seen["split"] != seen.iloc[0]["split"]].iloc[0]["article_id"]
    pairs = pd.DataFrame({"article_id_a": [a], "article_id_b": [b]})

    linked = make_splits(df, pairs, seed=0)
    split_of = linked.set_index("article_id")["split"]
    assert split_of[a] == split_of[b]
    check_no_leakage(linked)


def test_group_mixing_held_out_and_seen_types_is_dropped():
    df = synthetic_catalogue(n_types=10)
    base = make_splits(df, seed=0)
    held = base[base["split"] == "test_unseen"].iloc[0]["article_id"]
    seen = base[base["split"] != "test_unseen"].iloc[0]["article_id"]
    pairs = pd.DataFrame({"article_id_a": [held], "article_id_b": [seen]})

    linked = make_splits(df, pairs, seed=0)
    assert held not in set(linked["article_id"])
    assert seen not in set(linked["article_id"])
    check_no_leakage(linked)

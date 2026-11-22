import pandas as pd
import pytest

from lookalike.data.splits import check_near_duplicates_together, make_splits

# Six products of one type each; products 1 and 2 are near-duplicates of each other.
DF = pd.DataFrame(
    {
        "article_id": [11, 21, 31, 41, 51, 61],
        "product_code": [1, 2, 3, 4, 5, 6],
        "product_type_name": ["Shirt"] * 6,
    }
)


def _splits(split_names):
    return pd.DataFrame({"article_id": DF.article_id, "split": split_names})


def test_passes_when_pair_is_in_one_split_and_counts_checked_pairs():
    pairs = pd.DataFrame({"article_id_a": [11], "article_id_b": [21]})
    s = _splits(["train", "train", "val", "val", "test", "test"])
    assert check_near_duplicates_together(s, pairs) == 1


def test_fails_when_a_pair_straddles_two_splits():
    pairs = pd.DataFrame({"article_id_a": [11], "article_id_b": [21]})
    s = _splits(["train", "val", "val", "val", "test", "test"])
    with pytest.raises(AssertionError):
        check_near_duplicates_together(s, pairs)


def test_pairs_with_a_dropped_item_are_ignored():
    pairs = pd.DataFrame({"article_id_a": [11, 11], "article_id_b": [21, 999]})  # 999 not kept
    s = _splits(["train", "train", "val", "val", "test", "test"])
    assert check_near_duplicates_together(s, pairs) == 1


def test_make_splits_output_always_satisfies_the_guard():
    pairs = pd.DataFrame({"article_id_a": [11, 31, 41], "article_id_b": [21, 41, 51]})
    for seed in range(10):
        out = make_splits(DF, pairs, seed=seed, holdout_type_frac=0.0001)
        assert check_near_duplicates_together(out, pairs) >= 0

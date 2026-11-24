import numpy as np
import pandas as pd
import pytest

from lookalike.eval.harness import evaluate_embeddings, rank_by_cosine, same_product_mask
from lookalike.eval.rankings import evaluate_rankings

GALLERY = pd.DataFrame(
    {
        "product_code": [1, 1, 2, 3, 4, 5],
        "product_type_name": ["T1", "T1", "T1", "T2", "T1", "T2"],
        "colour_group_name": ["B", "R", "B", "B", "R", "B"],
        "graphical_appearance_name": ["S", "S", "X", "S", "X", "S"],
    }
)
KW = dict(ks=(1, 3), ndcg_k=3, map_k=3)


def test_matches_the_cosine_harness_when_given_its_ranking():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(GALLERY), 4))
    expected = evaluate_embeddings(emb, emb, GALLERY, GALLERY, **KW)
    ranked = rank_by_cosine(emb, emb, same_product_mask(GALLERY, GALLERY), 3)
    got = evaluate_rankings(ranked, GALLERY, GALLERY, **KW)
    assert expected.keys() == got.keys()
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], equal_nan=True)


def test_padding_counts_as_a_miss():
    emb = np.random.default_rng(1).normal(size=(len(GALLERY), 4))
    full = rank_by_cosine(emb, emb, same_product_mask(GALLERY, GALLERY), 3)
    padded = full.copy()
    padded[0] = [-1, -1, -1]
    a = evaluate_rankings(full, GALLERY, GALLERY, **KW)
    b = evaluate_rankings(padded, GALLERY, GALLERY, **KW)
    assert b["ndcg@3"][0] == 0 and a["ndcg@3"][0] > 0  # query 0 has graded items to find
    np.testing.assert_allclose(a["ndcg@3"][1:], b["ndcg@3"][1:])


def test_rejects_rankings_that_return_the_query_own_product():
    ranked = np.array([[1, 2, 3]] + [[3, 2, 4]] * 5)  # item 1 shares product 1 with query 0
    with pytest.raises(ValueError, match="own product"):
        evaluate_rankings(ranked, GALLERY, GALLERY, **KW)


def test_rejects_rankings_that_are_too_short():
    with pytest.raises(ValueError):
        evaluate_rankings(np.zeros((6, 2), int), GALLERY, GALLERY, **KW)

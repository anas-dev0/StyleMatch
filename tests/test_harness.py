"""Harness tests on a 4-item catalogue where every ranking can be derived by hand.

item  product_code  type  colour  pattern  embedding
g0    A             T1    Black   Solid    (1.0, 0.0)
g1    B             T1    Black   Stripe   (0.9, 0.1)
g2    C             T2    Black   Solid    (0.0, 1.0)
g3    D             T1    Red     Stripe   (0.6, 0.4)

Queries are the gallery items themselves (own product excluded from results).
Cosine similarities (rounded):
  g0: g1 .994, g3 .832, g2 0
  g1: g0 .994, g3 .888, g2 .110
  g2: g3 .555, g1 .110, g0 0
  g3: g1 .888, g0 .832, g2 .555
"""

import numpy as np
import pandas as pd
import pytest

from lookalike.eval.harness import (
    evaluate_embeddings,
    rank_by_cosine,
    same_product_mask,
    summarise,
)

GALLERY = pd.DataFrame(
    {
        "product_code": ["A", "B", "C", "D"],
        "product_type_name": ["T1", "T1", "T2", "T1"],
        "colour_group_name": ["Black", "Black", "Black", "Red"],
        "graphical_appearance_name": ["Solid", "Stripe", "Solid", "Stripe"],
    }
)
EMB = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.4]])
KW = dict(ks=(1, 3), ndcg_k=3, map_k=3)


def test_ranking_by_hand_and_own_product_excluded():
    exclude = same_product_mask(GALLERY, GALLERY)
    ranked = rank_by_cosine(EMB, EMB, exclude, top_k=3)
    np.testing.assert_array_equal(ranked, [[1, 3, 2], [0, 3, 2], [3, 1, 0], [1, 0, 2]])
    for q in range(4):
        assert q not in ranked[q]  # a query never gets its own colour variant back


def test_ranking_ignores_embedding_length():
    exclude = same_product_mask(GALLERY, GALLERY)
    a = rank_by_cosine(EMB, EMB, exclude, top_k=3)
    b = rank_by_cosine(EMB * 10, EMB, exclude, top_k=3)
    np.testing.assert_array_equal(a, b)


def test_query_zero_is_perfect_only_if_unreachable_items_leave_the_ideal():
    # Query g0 (T1 Black Solid). Returned: g1 (grade 2), g3 (1), g2 (0) = the best possible
    # list once g0 itself is removed. nDCG would be < 1 if g0 (grade 3) stayed in the ideal.
    res = evaluate_embeddings(EMB, EMB, GALLERY, GALLERY, **KW)
    assert res["ndcg@3"][0] == pytest.approx(1.0)
    assert res["hit@1"][0] == 1.0
    assert res["recall@1"][0] == 1.0  # R = 1 (only g1 has grade >= 2)


def test_query_one_by_hand():
    # Query g1 (T1 Black Stripe): returns g0 (2), g3 (2), g2 (0). R = 2.
    res = evaluate_embeddings(EMB, EMB, GALLERY, GALLERY, **KW)
    assert res["recall@1"][1] == pytest.approx(1 / 2)
    assert res["recall@3"][1] == pytest.approx(2 / 2)
    assert res["map@3"][1] == pytest.approx((1 / 1 + 2 / 2) / 2)


def test_query_with_no_relevant_item_is_nan():
    # Query g2 is the only T2 item, so every other item has grade 0.
    res = evaluate_embeddings(EMB, EMB, GALLERY, GALLERY, **KW)
    for name in ("hit@1", "recall@1", "mrr", "map@3", "ndcg@3"):
        assert np.isnan(res[name][2])


def test_summary_by_hand():
    summary = summarise(evaluate_embeddings(EMB, EMB, GALLERY, GALLERY, **KW)).set_index("metric")
    # recall@1 per query: 1, 1/2, NaN, 1  -> mean over the 3 defined queries
    assert summary.loc["recall@1", "mean"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert summary.loc["recall@1", "n_queries"] == 3
    assert summary.loc["hit@1", "mean"] == pytest.approx(1.0)
    assert summary.loc["ndcg@3", "mean"] == pytest.approx(1.0)

"""Metric tests on hand-computed fixtures.

Gallery grades (all items a query may retrieve): [3, 3, 2, 2, 1, 1, 0, 0, 0, 0]
  -> items with grade >= 2: R = 4.   Ideal top-5 list: [3, 3, 2, 2, 1].
Query A returns grades [3, 0, 2, 1, 0]  -> binary (>= 2): [1, 0, 1, 0, 0]
Query B returns grades [0, 0, 1, 3, 0]  -> binary (>= 2): [0, 0, 0, 1, 0]
"""

import math

import numpy as np
import pytest

from lookalike.eval import metrics as M

GALLERY_ROW = [3, 3, 2, 2, 1, 1, 0, 0, 0, 0]
ALL = np.tile(GALLERY_ROW, (2, 1))
RANKED = np.array([[3, 0, 2, 1, 0], [0, 0, 1, 3, 0]])
IDCG5 = 3 + 3 / math.log2(3) + 2 / 2 + 2 / math.log2(5) + 1 / math.log2(6)  # ~ 7.141


def test_dcg_by_hand():
    # positions 1..5 divided by log2(2..6) = 1, 1.585, 2, 2.322, 2.585
    expected_a = 3 / 1 + 0 + 2 / 2 + 1 / math.log2(5) + 0  # ~ 4.4307
    assert M.dcg_at_k(RANKED, 5)[0] == pytest.approx(expected_a)


def test_ndcg_by_hand():
    expected_a = (3 + 2 / 2 + 1 / math.log2(5)) / IDCG5  # ~ 0.6205
    expected_b = (1 / 2 + 3 / math.log2(5)) / IDCG5  # ~ 0.2509
    got = M.ndcg_at_k(RANKED, ALL, 5)
    assert got[0] == pytest.approx(expected_a)
    assert got[0] == pytest.approx(0.6205, abs=1e-3)
    assert got[1] == pytest.approx(expected_b)


def test_ndcg_is_one_for_the_ideal_order():
    ideal = np.array([[3, 3, 2, 2, 1]])
    assert M.ndcg_at_k(ideal, ALL[:1], 5)[0] == pytest.approx(1.0)


def test_same_items_in_better_order_score_higher():
    worse = np.array([[3, 0, 2, 1, 0]])
    better = np.array([[3, 2, 1, 0, 0]])
    assert M.ndcg_at_k(better, ALL[:1], 5)[0] > M.ndcg_at_k(worse, ALL[:1], 5)[0]


def test_hit_at_k():
    np.testing.assert_array_equal(M.hit_at_k(RANKED, ALL, 1), [1, 0])
    np.testing.assert_array_equal(M.hit_at_k(RANKED, ALL, 3), [1, 0])
    np.testing.assert_array_equal(M.hit_at_k(RANKED, ALL, 4), [1, 1])


def test_recall_at_k():
    # A finds 2 of R=4 within 5; B finds 1 of 4.
    np.testing.assert_allclose(M.recall_at_k(RANKED, ALL, 5), [2 / 4, 1 / 4])
    np.testing.assert_allclose(M.recall_at_k(RANKED, ALL, 1), [1 / 4, 0])


def test_mrr():
    # A: first relevant at rank 1. B: first relevant at rank 4.
    np.testing.assert_allclose(M.mrr(RANKED, ALL, 5), [1.0, 1 / 4])


def test_mrr_is_zero_when_relevant_exist_but_none_in_top_k():
    assert M.mrr(RANKED, ALL, 3)[1] == 0.0


def test_average_precision_at_k():
    # A: hits at positions 1 and 3 -> precisions 1/1 and 2/3; divide by min(R=4, k=5) = 4.
    # B: one hit at position 4 -> precision 1/4; divide by 4.
    np.testing.assert_allclose(
        M.average_precision_at_k(RANKED, ALL, 5), [(1 + 2 / 3) / 4, (1 / 4) / 4]
    )


def test_no_relevant_items_gives_nan_for_binary_metrics():
    only_weak = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])  # nothing >= 2
    ranked = np.array([[1, 1, 0, 0, 0]])
    for fn in (M.hit_at_k, M.recall_at_k, M.mrr, M.average_precision_at_k):
        assert np.isnan(fn(ranked, only_weak, 5)[0])
    assert not np.isnan(M.ndcg_at_k(ranked, only_weak, 5)[0])  # graded metric still defined


def test_ndcg_nan_when_gallery_has_only_zero_grades():
    zeros = np.zeros((1, 10))
    assert np.isnan(M.ndcg_at_k(np.zeros((1, 5)), zeros, 5)[0])


def test_k_larger_than_ranked_list_raises():
    with pytest.raises(ValueError):
        M.ndcg_at_k(RANKED, ALL, 6)


def test_bootstrap_of_a_constant_is_that_constant():
    assert M.bootstrap_ci(np.full(50, 0.7)) == pytest.approx((0.7, 0.7, 0.7))


def test_bootstrap_ignores_nan_and_is_deterministic():
    v = np.array([0.0, 1.0, np.nan, 0.5, 0.25])
    assert M.bootstrap_ci(v, seed=3) == M.bootstrap_ci(v, seed=3)
    assert M.bootstrap_ci(v)[0] == pytest.approx(np.mean([0.0, 1.0, 0.5, 0.25]))


def test_bootstrap_interval_brackets_the_mean_and_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.random(30)
    large = rng.random(3000)
    m_s, lo_s, hi_s = M.bootstrap_ci(small)
    m_l, lo_l, hi_l = M.bootstrap_ci(large)
    assert lo_s <= m_s <= hi_s and lo_l <= m_l <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_paired_diff_of_constant_gap():
    b = np.array([0.1, 0.4, 0.9, 0.3])
    mean, lo, hi = M.paired_bootstrap_diff(b + 0.1, b)
    assert (mean, lo, hi) == pytest.approx((0.1, 0.1, 0.1))


def test_paired_diff_uses_only_queries_both_systems_have():
    a = np.array([1.0, np.nan, 0.5])
    b = np.array([0.5, 0.2, np.nan])
    mean, _, _ = M.paired_bootstrap_diff(a, b)
    assert mean == pytest.approx(0.5)  # only the first query counts

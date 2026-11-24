import numpy as np
import pytest

from lookalike.index.faiss_baseline import ann_recall, normalise


def test_ann_recall_is_overlap_over_exact():
    approx = np.array([[1, 2, 3], [4, 5, 6]])
    exact = np.array([[3, 2, 9], [7, 8, 9]])
    np.testing.assert_allclose(ann_recall(approx, exact), [2 / 3, 0.0])
    with pytest.raises(ValueError):
        ann_recall(approx[:1], exact)


def test_flat_index_matches_numpy_brute_force():
    pytest.importorskip("faiss")
    from lookalike.index.faiss_baseline import build_flat_index, search

    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(200, 16))
    queries = rng.normal(size=(20, 16))
    scores, idx = search(build_flat_index(gallery), queries, 5)
    sims = normalise(queries) @ normalise(gallery).T
    expected = np.argsort(-sims, axis=1)[:, :5]
    np.testing.assert_array_equal(idx, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(sims, expected, axis=1), rtol=1e-5)


def test_search_excluding_skips_the_query_group_and_keeps_order():
    pytest.importorskip("faiss")
    from lookalike.index.faiss_baseline import build_flat_index, search_excluding

    gallery = np.array([[1.0, 0.0], [0.99, 0.1], [0.9, 0.3], [0.0, 1.0]])
    groups = np.array([10, 10, 20, 30])  # items 0 and 1 are colour variants of one product
    out = search_excluding(build_flat_index(gallery), gallery[:1], 2, groups[:1], groups)
    np.testing.assert_array_equal(out, [[2, 3]])

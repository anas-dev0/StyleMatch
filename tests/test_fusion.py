import numpy as np
import pytest

from lookalike.eval.fusion import rank_by_scores, rrf_fuse


def test_rank_by_scores_excludes_and_pads_non_positive_scores():
    sims = np.array([[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 0.0, 1.0]])
    exclude = np.array([[False, True, False, False], [False, False, False, False]])
    np.testing.assert_array_equal(rank_by_scores(sims, exclude, 3), [[3, 2, 0], [0, 3, 1]])
    np.testing.assert_array_equal(
        rank_by_scores(sims, exclude, 3, min_score=0), [[3, 2, -1], [0, 3, -1]]
    )


def test_rank_by_scores_breaks_ties_by_gallery_index():
    sims = np.array([[1.0, 2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(rank_by_scores(sims, np.zeros_like(sims, bool), 3), [[1, 2, 3]])


def test_rrf_scores_by_hand():
    # list A: 7, 8, 9   list B: 9, 7, -1   (k = 60, 1-based ranks)
    # 7: 1/61 + 1/62 = .032522   9: 1/63 + 1/61 = .032266   8: 1/62 = .016129
    a = np.array([[7, 8, 9]])
    b = np.array([[9, 7, -1]])
    np.testing.assert_array_equal(rrf_fuse([a, b], k=60, top_k=3), [[7, 9, 8]])


def test_rrf_ties_go_to_better_best_rank_then_lower_index():
    # 5 and 6 each appear once at rank 1 in different lists: equal score and best rank -> index
    np.testing.assert_array_equal(
        rrf_fuse([np.array([[6, 1]]), np.array([[5, 2]])], top_k=4), [[5, 6, 1, 2]]
    )


def test_rrf_single_list_is_identity_and_pads_short_output():
    a = np.array([[3, 1, -1]])
    np.testing.assert_array_equal(rrf_fuse([a], top_k=4), [[3, 1, -1, -1]])


def test_rrf_weights_can_flip_the_order():
    a, b = np.array([[1, 2]]), np.array([[2, 1]])
    assert rrf_fuse([a, b], weights=[2.0, 1.0], top_k=1)[0, 0] == 1
    assert rrf_fuse([a, b], weights=[1.0, 2.0], top_k=1)[0, 0] == 2


def test_rrf_validates_inputs():
    with pytest.raises(ValueError):
        rrf_fuse([])
    with pytest.raises(ValueError):
        rrf_fuse([np.zeros((2, 3), int), np.zeros((3, 3), int)])
    with pytest.raises(ValueError):
        rrf_fuse([np.zeros((1, 3), int)], weights=[1.0, 2.0])

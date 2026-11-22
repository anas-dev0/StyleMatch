import numpy as np

from lookalike.data.dedupe import near_duplicate_pairs

# Rows 0 and 1 point almost the same way (cosine ~0.9995). Row 2 is orthogonal to them.
# Row 3 is exactly row 0 scaled by 5: cosine similarity ignores length.
EMB = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.03, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0],
    ]
)


def test_finds_expected_pairs_only():
    pairs = near_duplicate_pairs(EMB, threshold=0.97)
    assert {tuple(p) for p in pairs} == {(0, 1), (0, 3), (1, 3)}


def test_pairs_are_ordered_and_not_mirrored_or_self():
    pairs = near_duplicate_pairs(EMB, threshold=0.97)
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_block_size_does_not_change_result():
    small = near_duplicate_pairs(EMB, threshold=0.97, block_size=1)
    big = near_duplicate_pairs(EMB, threshold=0.97, block_size=100)
    assert {tuple(p) for p in small} == {tuple(p) for p in big}


def test_no_pairs_when_all_vectors_are_orthogonal():
    assert near_duplicate_pairs(np.eye(3), threshold=0.5).shape == (0, 2)

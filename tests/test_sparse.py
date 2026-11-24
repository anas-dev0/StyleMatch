import math

import numpy as np
import pytest

from lookalike.index.sparse import BM25, math_bm25, tokenize

DOCS = [
    tokenize("Black cotton shirt with long sleeves"),
    tokenize("Striped cotton dress"),
    tokenize("Black leather boots, black laces"),
]


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("A Slim-fit SHIRT in 100% cotton, with buttons.") == [
        "slim", "fit", "shirt", "100", "cotton", "buttons",
    ]  # fmt: skip
    assert tokenize(None) == []
    assert tokenize("") == []


def test_idf_matches_formula_by_hand():
    bm = BM25().fit(DOCS)
    # "cotton" is in 2 of 3 docs: ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6)
    assert bm.idf[bm.vocab["cotton"]] == pytest.approx(math.log(1.6))
    # "boots" is in 1 of 3: ln(2.5 / 1.5 + 1)
    assert bm.idf[bm.vocab["boots"]] == pytest.approx(math.log(2.5 / 1.5 + 1))


def test_dot_product_equals_textbook_bm25():
    bm = BM25().fit(DOCS)
    queries = [tokenize("black cotton"), tokenize("black black boots"), tokenize("unknown words")]
    scores = bm.score_matrix(queries, DOCS)
    for qi, q in enumerate(queries):
        for di, d in enumerate(DOCS):
            assert scores[qi, di] == pytest.approx(math_bm25(q, d, DOCS), rel=1e-5)
    assert scores[2].sum() == 0  # no known term, no score


def test_term_frequency_saturates_and_long_docs_are_penalised():
    bm = BM25().fit(DOCS)
    q = tokenize("black")
    s = bm.score_matrix([q], DOCS)[0]
    assert s[2] > s[0] > 0  # two "black"s beat one ...
    assert s[2] < 2 * s[0]  # ... but count for less than twice as much
    assert s[1] == 0


def test_query_vector_has_unit_weight_per_distinct_term_and_empty_doc_is_empty():
    bm = BM25().fit(DOCS)
    idx, val = bm.query_vector(tokenize("black black cotton nope"))
    assert sorted(idx) == sorted([bm.vocab["black"], bm.vocab["cotton"]])
    np.testing.assert_array_equal(val, [1, 1])
    idx, val = bm.doc_vector([])
    assert len(idx) == 0 and len(val) == 0


def test_fit_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        BM25().fit([])

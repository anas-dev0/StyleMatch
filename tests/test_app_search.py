import numpy as np
import pandas as pd
import pytest
from app.backend.search import SearchIndex, load_index

# Three items in 2-D: item 10 points along x, item 20 along y, item 30 diagonally.
IDS = np.array([10, 20, 30])
EMB = np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 1.0]])  # lengths differ on purpose
META = pd.DataFrame(
    {
        "article_id": [30, 10, 20, 99],  # different order than IDS, plus an unused id
        "prod_name": ["diag", "right", "up", "unused"],
        "product_type_name": ["Shirt", "Dress", "Shoe", "x"],
        "colour_group_name": ["Black", "Red", "Blue", "x"],
        "detail_desc": ["d", None, "u", "x"],  # a missing description
    }
)


def make_index():
    return SearchIndex(EMB, IDS, META)


def test_ranking_follows_cosine_not_vector_length():
    # Query along x: item 10 is identical in direction (cos 1), item 30 is 45 degrees (cos 0.7071),
    # item 20 is perpendicular (cos 0). Item 20 has the longest vector but must come last.
    hits = make_index().search(np.array([3.0, 0.0]), k=3)
    assert [h.article_id for h in hits] == [10, 30, 20]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5, abs=1e-6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)


def test_k_limits_results_and_is_capped_at_index_size():
    index = make_index()
    assert len(index.search(np.array([0.0, 1.0]), k=1)) == 1
    assert [h.article_id for h in index.search(np.array([0.0, 1.0]), k=1)] == [20]
    assert len(index.search(np.array([0.0, 1.0]), k=100)) == 3


def test_describe_follows_the_index_order_not_the_metadata_order():
    index = make_index()
    hit = index.search(np.array([3.0, 0.0]), k=1)[0]  # article 10
    assert index.describe(hit) == {
        "name": "right",
        "product_type": "Dress",
        "colour": "Red",
        "description": "",  # missing text becomes an empty string
    }


def test_bad_queries_are_rejected():
    index = make_index()
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 0.0, 0.0]), k=1)  # wrong length
    with pytest.raises(ValueError):
        index.search(np.array([0.0, 0.0]), k=1)  # all zeros has no direction
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 0.0]), k=0)


def test_inconsistent_inputs_are_rejected():
    with pytest.raises(ValueError):
        SearchIndex(EMB[:2], IDS, META)  # 2 vectors, 3 ids
    with pytest.raises(ValueError):
        SearchIndex(EMB, np.array([10, 10, 30]), META)  # duplicate ids
    with pytest.raises(KeyError):
        SearchIndex(EMB, np.array([10, 20, 31]), META)  # 31 has no metadata


def test_load_index_reads_the_exported_files_and_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_assets"):
        load_index(tmp_path)
    np.save(tmp_path / "embeddings.npy", EMB)
    np.save(tmp_path / "ids.npy", IDS)
    META.to_csv(tmp_path / "metadata.csv", index=False)
    index = load_index(tmp_path)
    assert len(index) == 3
    assert index.dim == 2

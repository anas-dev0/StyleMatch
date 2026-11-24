import numpy as np
import pandas as pd
import pytest
from app.backend.search import BruteForceStore, items_table, read_export

# Three items in 2-D: item 10 points along x, item 20 along y, item 30 diagonally.
IDS = np.array([10, 20, 30])
EMB = np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 1.0]])  # lengths differ on purpose
META = pd.DataFrame(
    {
        "article_id": [30, 10, 20, 99, 10],  # other order than IDS, an unused id, a duplicate row
        "prod_name": ["diag", "right", "up", "unused", "right"],
        "product_type_name": ["Shirt", "Dress", "Shoe", "x", "Dress"],
        "colour_group_name": ["Black", "Red", "Black", "x", "Red"],
        "detail_desc": ["d", None, "u", "x", None],  # a missing description
    }
)


def make_store():
    return BruteForceStore(EMB, IDS, items_table(META))


def test_items_table_renames_columns_dedupes_and_blanks_missing_text():
    items = items_table(META)
    assert list(items.columns) == ["article_id", "name", "product_type", "colour", "description"]
    assert len(items) == 4  # the duplicate row for article 10 is gone
    assert items.loc[items.article_id == 10, "description"].iloc[0] == ""


def test_ranking_follows_cosine_not_vector_length():
    # Query along x: item 10 is identical in direction (cos 1), item 30 is 45 degrees (cos 0.7071),
    # item 20 is perpendicular (cos 0). Item 20 has the longest vector but must come last.
    hits = make_store().search(np.array([3.0, 0.0]), k=3)
    assert [h.article_id for h in hits] == [10, 30, 20]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5, abs=1e-6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)


def test_hits_carry_the_items_own_fields_not_a_neighbours():
    hit = make_store().search(np.array([3.0, 0.0]), k=1)[0]  # article 10
    assert (hit.name, hit.product_type, hit.colour, hit.description) == (
        "right",
        "Dress",
        "Red",
        "",
    )


def test_k_limits_results_and_is_capped_at_store_size():
    store = make_store()
    assert [h.article_id for h in store.search(np.array([0.0, 1.0]), k=1)] == [20]
    assert len(store.search(np.array([0.0, 1.0]), k=100)) == 3


def test_filters_restrict_candidates_and_combine():
    store = make_store()
    # Query points at item 20 (Shoe, Black); the Red item 10 must be filtered out.
    black = store.search(np.array([0.0, 1.0]), k=3, colour="Black")
    assert [h.article_id for h in black] == [20, 30]  # only the two Black items, best first
    # An item that matches the query best must still be dropped if it fails the filter.
    shirts = store.search(np.array([0.0, 1.0]), k=3, product_type="Shirt")
    assert [h.article_id for h in shirts] == [30]
    both = store.search(np.array([0.0, 1.0]), k=3, product_type="Dress", colour="Black")
    assert both == []  # no Black dress: an empty list, not an error


def test_facets_list_distinct_values_sorted():
    assert make_store().facets() == {
        "product_type": ["Dress", "Shirt", "Shoe"],
        "colour": ["Black", "Red"],
    }


def test_bad_queries_are_rejected():
    store = make_store()
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0, 0.0]), k=1)  # wrong length
    with pytest.raises(ValueError):
        store.search(np.array([0.0, 0.0]), k=1)  # all zeros has no direction
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0]), k=0)


def test_inconsistent_inputs_are_rejected():
    items = items_table(META)
    with pytest.raises(ValueError):
        BruteForceStore(EMB[:2], IDS, items)  # 2 vectors, 3 ids
    with pytest.raises(ValueError):
        BruteForceStore(EMB, np.array([10, 10, 30]), items)  # duplicate ids
    with pytest.raises(KeyError):
        BruteForceStore(EMB, np.array([10, 20, 31]), items)  # 31 has no metadata


def test_read_export_reads_the_files_and_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_assets"):
        read_export(tmp_path)
    np.save(tmp_path / "embeddings.npy", EMB)
    np.save(tmp_path / "ids.npy", IDS)
    META.to_csv(tmp_path / "metadata.csv", index=False)
    emb, ids, items = read_export(tmp_path)
    assert emb.shape == (3, 2)
    assert list(ids) == [10, 20, 30]
    assert "colour" in items.columns

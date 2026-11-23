import numpy as np
import pandas as pd
import pytest
from app.tools.export_assets import build_metadata, make_thumbnail, thumbnail_size


def test_thumbnail_size_by_hand():
    assert thumbnail_size(1000, 2000, 256) == (128, 256)  # portrait: height is the long side
    assert thumbnail_size(2000, 1000, 256) == (256, 128)
    assert thumbnail_size(100, 200, 256) == (100, 200)  # already small: never enlarged
    assert thumbnail_size(1, 1000, 256) == (1, 256)  # never rounds a side down to 0


def test_make_thumbnail_shrinks_and_leaves_no_temp_file(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    src = tmp_path / "big.png"
    PIL.new("RGB", (400, 800), (200, 10, 10)).save(src)
    dst = tmp_path / "small.jpg"
    make_thumbnail(src, dst, max_side=100, quality=80)
    with PIL.open(dst) as out:
        assert out.size == (50, 100)
        assert out.format == "JPEG"
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


ARTICLES = pd.DataFrame(
    {
        "article_id": [3, 1, 2, 2],  # unordered, with a duplicated row
        "prod_name": ["c", "a", "b", "b"],
        "product_type_name": ["T3", "T1", "T2", "T2"],
        "colour_group_name": ["C3", "C1", "C2", "C2"],
        "detail_desc": ["d3", "d1", "d2", "d2"],
    }
)


def test_build_metadata_follows_the_requested_order_without_duplicates():
    out = build_metadata(ARTICLES, np.array([1, 2, 3]))
    assert list(out["article_id"]) == [1, 2, 3]
    assert list(out["prod_name"]) == ["a", "b", "c"]
    assert list(out.columns) == [
        "article_id",
        "prod_name",
        "product_type_name",
        "colour_group_name",
        "detail_desc",
    ]


def test_build_metadata_raises_for_unknown_ids():
    with pytest.raises(KeyError):
        build_metadata(ARTICLES, np.array([1, 99]))

import pandas as pd

from lookalike.data.subset import build_subset, image_path


def articles() -> pd.DataFrame:
    rows = []
    aid = 0

    def add(n, type_, colour="Black", appearance="Solid"):
        nonlocal aid
        for _ in range(n):
            rows.append(
                {
                    "article_id": aid,
                    "product_code": aid // 2,
                    "product_type_name": type_,
                    "colour_group_name": colour,
                    "graphical_appearance_name": appearance,
                }
            )
            aid += 1

    add(10, "Dress")
    add(4, "Scarf")  # too small
    add(3, "Dress", colour="Unknown")  # dropped as unknown
    add(2, "Dress", colour=None)  # dropped as missing
    return pd.DataFrame(rows)


def test_drops_unknown_missing_and_small_types():
    out = build_subset(articles(), cap_per_type=100, min_per_type=5, seed=0)
    assert set(out["product_type_name"]) == {"Dress"}
    assert len(out) == 10


def test_cap_per_type():
    out = build_subset(articles(), cap_per_type=6, min_per_type=5, seed=0)
    assert len(out) == 6


def test_cap_is_deterministic_for_a_seed():
    a = build_subset(articles(), cap_per_type=6, min_per_type=5, seed=7)
    b = build_subset(articles(), cap_per_type=6, min_per_type=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_images_filter():
    out = build_subset(
        articles(), cap_per_type=100, min_per_type=1, seed=0, has_image=lambda a: a < 4
    )
    assert set(out["article_id"]) == {0, 1, 2, 3}


def test_image_path_layout():
    p = image_path("images", 108775015)
    assert p.as_posix() == "images/010/0108775015.jpg"


def grouped_articles() -> pd.DataFrame:
    rows = []
    spec = [
        ("Garment Upper body", "Sweater", 6),
        ("Cosmetic", "Fine cosmetics", 6),
        ("Accessories", "Soft Toys", 6),
        ("Accessories", "Scarf", 6),
    ]
    aid = 0
    for group, type_, n in spec:
        for _ in range(n):
            rows.append(
                {
                    "article_id": aid,
                    "product_code": aid,
                    "product_group_name": group,
                    "product_type_name": type_,
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                }
            )
            aid += 1
    return pd.DataFrame(rows)


def test_exclude_groups_and_types_case_insensitive():
    out = build_subset(
        grouped_articles(),
        cap_per_type=100,
        min_per_type=1,
        seed=0,
        exclude_groups=["cosmetic "],
        exclude_types=["SOFT TOYS"],
    )
    assert set(out["product_type_name"]) == {"Sweater", "Scarf"}


def test_exclusion_runs_before_the_min_size_filter():
    # Type "Mixed" has 6 rows: 3 in the excluded group "Cosmetic", 3 in "Accessories".
    # Exclusion first leaves 3 rows (< min 5), so the type is dropped. If the min-size filter
    # ran first, all 6 rows would pass it and 3 rows of "Mixed" would wrongly survive.
    df = grouped_articles()
    extra = df.iloc[:6].copy()
    extra["article_id"] = range(100, 106)
    extra["product_code"] = range(100, 106)
    extra["product_type_name"] = "Mixed"
    extra["product_group_name"] = ["Cosmetic"] * 3 + ["Accessories"] * 3
    df = pd.concat([df, extra], ignore_index=True)

    out = build_subset(df, cap_per_type=100, min_per_type=5, seed=0, exclude_groups=["Cosmetic"])
    assert "Mixed" not in set(out["product_type_name"])


def test_excluding_a_missing_column_raises():
    import pytest

    with pytest.raises(KeyError):
        build_subset(
            articles(), cap_per_type=100, min_per_type=1, seed=0, exclude_groups=["Cosmetic"]
        )

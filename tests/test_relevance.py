import numpy as np
import pandas as pd

from lookalike.data.relevance import relevance, relevance_matrix


def item(type_, colour, appearance):
    return pd.Series(
        {
            "product_type_name": type_,
            "colour_group_name": colour,
            "graphical_appearance_name": appearance,
        }
    )


QUERY = item("Dress", "Black", "Solid")


def test_grade_3_same_type_colour_and_appearance():
    assert relevance(QUERY, item("Dress", "Black", "Solid")) == 3


def test_grade_2_same_type_and_colour_only():
    assert relevance(QUERY, item("Dress", "Black", "Stripe")) == 2


def test_grade_2_same_type_and_appearance_only():
    assert relevance(QUERY, item("Dress", "Red", "Solid")) == 2


def test_grade_1_same_type_only():
    assert relevance(QUERY, item("Dress", "Red", "Stripe")) == 1


def test_grade_0_different_type_even_if_everything_else_matches():
    assert relevance(QUERY, item("Trousers", "Black", "Solid")) == 0


def test_missing_values_never_match():
    assert relevance(item("Dress", None, None), item("Dress", None, None)) == 1


def test_matrix_matches_pairwise_function():
    queries = pd.DataFrame([QUERY, item("Trousers", "Blue", "Denim")])
    gallery = pd.DataFrame(
        [
            item("Dress", "Black", "Solid"),
            item("Dress", "Red", "Stripe"),
            item("Trousers", "Blue", "Solid"),
        ]
    )
    expected = np.array([[3, 1, 0], [0, 0, 2]])
    np.testing.assert_array_equal(relevance_matrix(queries, gallery), expected)

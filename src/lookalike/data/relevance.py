"""Graded relevance from catalogue attributes (a proxy for visual similarity).

3: same product type AND colour group AND graphical appearance
2: same product type AND (colour group OR graphical appearance)
1: same product type
0: otherwise
Missing values (NaN) never match anything.
"""

import numpy as np
import pandas as pd

TYPE_COL = "product_type_name"
COLOUR_COL = "colour_group_name"
APPEARANCE_COL = "graphical_appearance_name"


def _match_matrix(query_values: pd.Series, gallery_values: pd.Series) -> np.ndarray:
    """Boolean (n_queries, n_gallery) matrix: True where values are equal and not missing."""
    codes, _ = pd.factorize(pd.concat([query_values, gallery_values], ignore_index=True))
    q, g = codes[: len(query_values)], codes[len(query_values) :]
    return (q[:, None] == g[None, :]) & (q[:, None] >= 0)


def relevance_matrix(queries: pd.DataFrame, gallery: pd.DataFrame) -> np.ndarray:
    """Relevance grades (0-3) for every (query, gallery item) pair, as an int matrix."""
    same_type = _match_matrix(queries[TYPE_COL], gallery[TYPE_COL])
    same_colour = _match_matrix(queries[COLOUR_COL], gallery[COLOUR_COL])
    same_appearance = _match_matrix(queries[APPEARANCE_COL], gallery[APPEARANCE_COL])
    either = same_colour | same_appearance
    both = same_colour & same_appearance
    # type gives 1; either attribute adds 1; both attributes add one more.
    return same_type * (1 + either.astype(int) + both.astype(int))


def relevance(query: pd.Series, candidate: pd.Series) -> int:
    """Relevance grade (0-3) of one candidate for one query."""
    q = query.to_frame().T
    c = candidate.to_frame().T
    return int(relevance_matrix(q, c)[0, 0])

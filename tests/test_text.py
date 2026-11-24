import numpy as np
import pandas as pd
import pytest

from lookalike.config import load_config
from lookalike.embed.text import build_caption, build_captions

DF = pd.DataFrame(
    {
        "product_type_name": ["Shirt", "Dress", None],
        "colour_group_name": ["Black", "  ", "Red"],
        "detail_desc": ["Slim shirt in cotton.  ", np.nan, None],
    }
)


def test_caption_joins_present_fields_in_order_and_strips():
    assert build_caption(DF.iloc[0], ["product_type_name", "detail_desc"]) == (
        "Shirt. Slim shirt in cotton."
    )
    assert build_caption(DF.iloc[0], ["detail_desc", "product_type_name"]) == (
        "Slim shirt in cotton.. Shirt"
    )


def test_caption_skips_missing_and_blank_values():
    assert build_caption(DF.iloc[1], ["product_type_name", "colour_group_name", "detail_desc"]) == (
        "Dress"
    )
    assert build_caption(DF.iloc[2], ["detail_desc"]) == ""


def test_build_captions_rejects_unknown_fields():
    assert len(build_captions(DF, ["detail_desc"])) == 3
    with pytest.raises(KeyError):
        build_captions(DF, ["prod_name"])


def test_text_config_uses_the_pinned_fashionclip_revision_from_the_image_config():
    text = load_config("configs/embed/text.yaml")
    image = load_config("configs/embed/baselines.yaml")["models"]["fashionclip"]
    for spec in text["models"].values():
        assert spec["hf_id"] == image["hf_id"]
        assert spec["revision"] == image["revision"]
        assert spec["fields"]
    assert text["models"]["fashionclip_text_desc"]["fields"] == ["detail_desc"]

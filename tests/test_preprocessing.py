import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetError
from src.preprocessing import (clean_text, normalize_label, prepare_dataframe,
                               present_categories)


def test_lowercase_and_whitespace():
    assert clean_text("  Hello   WORLD \n\t again ") == "hello world again"


def test_html_url_email_removed():
    raw = "<p>Buy <b>now</b></p> at https://shop.example.com/x?y=1 or www.test.org mail me@x.com &amp; save"
    cleaned = clean_text(raw)
    assert "<" not in cleaned and "http" not in cleaned and "www" not in cleaned and "@" not in cleaned
    assert cleaned == "buy now at or mail save"


def test_numbers_and_model_names_are_kept():
    assert clean_text("iPhone 15 Pro, 256GB (USB-C)") == "iphone 15 pro 256gb usb c"


def test_apostrophes_joined_not_split():
    assert clean_text("Men's T-shirt") == "mens t shirt"


@pytest.mark.parametrize("bad", [None, np.nan, pd.NA, "", "   ", "<br/>"])
def test_missing_or_empty_input_gives_empty_string(bad):
    assert clean_text(bad) == ""


def test_non_string_input_is_converted():
    assert clean_text(12345) == "12345"


def test_clean_text_is_idempotent():
    once = clean_text("<b>Wi-Fi</b> Router, 5GHz!!")
    assert clean_text(once) == once


def test_normalize_label_variants():
    assert normalize_label(" clothing and accessories ") == "Clothing & Accessories"
    assert normalize_label("ELECTRONICS") == "Electronics"
    assert normalize_label("Toys") == "Toys"
    assert normalize_label(None) is None and normalize_label("  ") is None


def test_present_categories_keeps_configured_order():
    assert present_categories(["Books", "Electronics", "Toys"]) == ["Electronics", "Books", "Toys"]


def _messy():
    return pd.DataFrame({
        "label": ["Books", "Books", "Books", None, "Toys", "Household", "Electronics", "Electronics",
                  "Household", "Household", "Electronics"],
        "text": ["a novel story", "a novel story", "A Novel Story!", "no label", "a toy car", None,
                 "usb cable", "usb cable", "usb cable", "a steel pan", "a phone charger"],
    })


def test_prepare_dataframe_reports_every_removal():
    clean_df, rep = prepare_dataframe(_messy(), "text", "label")
    assert rep["rows_initial"] == 11
    assert rep["removed_missing_label"] == 1
    assert rep["removed_missing_text"] == 1
    assert rep["removed_unexpected_label"] == 1 and rep["unexpected_labels"] == {"Toys": 1}
    assert rep["removed_exact_duplicates"] == 2          # one Books copy, one Electronics copy
    assert rep["removed_duplicates_after_cleaning"] == 1  # "A Novel Story!" == "a novel story"
    assert rep["removed_conflicting_labels"] == 2         # "usb cable" is Electronics and Household
    accounted = sum(rep[k] for k in rep if k.startswith("removed_"))
    assert rep["rows_initial"] - accounted == rep["rows_final"] == len(clean_df)
    assert set(clean_df.columns) == {"text", "label", "clean_text"}


def test_prepare_dataframe_reports_missing_categories():
    _, rep = prepare_dataframe(_messy(), "text", "label")
    assert "Clothing & Accessories" in rep["missing_categories"]


def test_prepare_dataframe_fails_when_nothing_usable():
    df = pd.DataFrame({"label": ["Toys", "Games"], "text": ["a", "b"]})
    with pytest.raises(DatasetError):
        prepare_dataframe(df, "text", "label")

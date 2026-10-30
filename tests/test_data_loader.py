import pandas as pd
import pytest

from src.data_loader import (DatasetError, detect_columns, find_dataset, load_dataset,
                             validate_dataset)


def test_loads_headerless_kaggle_layout(synthetic_csv):
    df, text_col, label_col = load_dataset(synthetic_csv)
    assert (text_col, label_col) == ("text", "label")
    assert len(df) == 120 and df["label"].nunique() == 4


def test_loads_csv_with_header_and_other_names(tmp_path, synthetic_df):
    path = tmp_path / "other.csv"
    synthetic_df.rename(columns={"text": "Description", "label": "Category"}).to_csv(path, index=False)
    _, text_col, label_col = load_dataset(path)
    assert (text_col, label_col) == ("Description", "Category")


def test_explicit_columns_override_detection(tmp_path, synthetic_df):
    path = tmp_path / "custom.csv"
    synthetic_df.rename(columns={"text": "blurb", "label": "kind"}).to_csv(path, index=False)
    _, text_col, label_col = load_dataset(path, "blurb", "kind")
    assert (text_col, label_col) == ("blurb", "kind")


def test_unknown_explicit_column_gives_helpful_error(synthetic_csv):
    with pytest.raises(DatasetError, match="Available columns"):
        load_dataset(synthetic_csv, text_column="nope")


def test_ambiguous_columns_raise():
    df = pd.DataFrame({"text": ["a"] * 12, "description": ["b"] * 12, "label": ["x", "y"] * 6})
    with pytest.raises(DatasetError, match="Ambiguous"):
        detect_columns(df)


def test_undetectable_columns_raise():
    df = pd.DataFrame({"a": range(12), "b": range(12), "c": range(12)})
    with pytest.raises(DatasetError, match="--text-column"):
        detect_columns(df)


def test_missing_file_message_has_instructions(tmp_path):
    with pytest.raises(DatasetError, match="--download"):
        load_dataset(tmp_path / "missing.csv")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(path)


def test_validate_rejects_single_class_and_tiny_data():
    with pytest.raises(DatasetError, match="two distinct"):
        validate_dataset(pd.DataFrame({"t": ["a"] * 20, "l": ["x"] * 20}), "t", "l")
    with pytest.raises(DatasetError, match="at least"):
        validate_dataset(pd.DataFrame({"t": ["a", "b"], "l": ["x", "y"]}), "t", "l")


def test_find_dataset(tmp_path):
    assert find_dataset(tmp_path) is None
    (tmp_path / "one.csv").write_text("a,b\n")
    assert find_dataset(tmp_path).name == "one.csv"

"""Text cleaning and dataset preparation.

Design principle: clean the text in a way that is simple to explain and does not
destroy useful information such as brand names, model numbers and sizes
(for example ``iPhone 15 Pro 256GB`` keeps ``15`` and ``256gb``).

``clean_text`` is used as the ``preprocessor`` of the TF-IDF vectorizer, so the
saved model accepts raw text and cleans it exactly as during training.
"""
from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

import config
from src.data_loader import DatasetError

_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_APOSTROPHE = re.compile(r"['\u2019\u2018`]")
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")


def _is_missing(value) -> bool:
    """True for None, NaN and pandas.NA without importing numpy here."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return type(value).__name__ == "NAType"


def clean_text(text) -> str:
    """Return a normalised version of ``text``.

    Steps: decode HTML entities -> remove HTML tags, URLs, e-mails and control
    characters -> Unicode NFKC -> lowercase -> drop apostrophes ("men's" ->
    "mens") -> replace other punctuation with spaces -> collapse whitespace.
    Numbers and alphanumeric tokens are kept. Missing or non-string input never
    raises: ``None``/``NaN`` give ``""`` and other objects are converted with
    ``str``.
    """
    if _is_missing(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _APOSTROPHE.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def _label_key(label: str) -> str:
    key = str(label).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", key).strip()


_LABEL_LOOKUP = {_label_key(c): c for c in config.CATEGORIES}
_LABEL_LOOKUP.update({"clothing": "Clothing & Accessories",
                      "clothing accessories": "Clothing & Accessories"})


def normalize_label(label) -> str | None:
    """Map spelling variants to the canonical category.

    Returns the canonical name, the stripped original text for unknown labels,
    or ``None`` when the label is missing.
    """
    if _is_missing(label) or not str(label).strip():
        return None
    stripped = str(label).strip()
    return _LABEL_LOOKUP.get(_label_key(stripped), stripped)


def present_categories(labels) -> list[str]:
    """Categories that actually occur in ``labels`` (configured order first)."""
    found = set(labels)
    ordered = [c for c in config.CATEGORIES if c in found]
    return ordered + sorted(found - set(ordered))


# --------------------------------------------------------------------------- #
# Dataset overview and preparation
# --------------------------------------------------------------------------- #
def dataset_overview(df: pd.DataFrame) -> dict:
    """Basic facts used by the notebook and the CLI."""
    return {
        "shape": tuple(df.shape),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {str(c): int(n) for c, n in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def prepare_dataframe(df: pd.DataFrame, text_column: str, label_column: str,
                      categories=None) -> tuple[pd.DataFrame, dict]:
    """Clean a raw dataset and document every removal.

    Returns ``(clean_df, report)``. ``clean_df`` has the columns ``text`` (raw
    description), ``label`` (canonical category) and ``clean_text``.
    Nothing is removed silently: every step is counted in ``report``.
    """
    categories = list(categories or config.CATEGORIES)
    report: dict = {"rows_initial": int(len(df))}

    work = pd.DataFrame({
        "text": df[text_column].astype(object),
        "label": df[label_column].map(normalize_label),
    })

    # 1. Missing labels / missing or blank descriptions
    missing_label = work["label"].isna()
    missing_text = work["text"].map(lambda t: _is_missing(t) or not str(t).strip())
    report["removed_missing_label"] = int(missing_label.sum())
    report["removed_missing_text"] = int((missing_text & ~missing_label).sum())
    work = work[~(missing_label | missing_text)].copy()
    work["text"] = work["text"].astype(str)

    # 2. Labels outside the expected categories
    unexpected = work.loc[~work["label"].isin(categories), "label"]
    report["unexpected_labels"] = {k: int(v) for k, v in unexpected.value_counts().items()}
    report["removed_unexpected_label"] = int(len(unexpected))
    work = work[work["label"].isin(categories)].copy()

    # 3. Exact duplicate records (same label and same raw text)
    n_before = len(work)
    work = work.drop_duplicates(subset=["label", "text"], keep="first")
    report["removed_exact_duplicates"] = int(n_before - len(work))

    # 4. Clean text; drop descriptions that are empty after cleaning
    work["clean_text"] = work["text"].map(clean_text)
    n_before = len(work)
    work = work[work["clean_text"] != ""]
    report["removed_empty_after_cleaning"] = int(n_before - len(work))

    # 5. Duplicates that only appear after cleaning (same label)
    n_before = len(work)
    work = work.drop_duplicates(subset=["label", "clean_text"], keep="first")
    report["removed_duplicates_after_cleaning"] = int(n_before - len(work))

    # 6. Identical cleaned text with conflicting labels is ambiguous -> remove
    n_labels = work.groupby("clean_text")["label"].transform("nunique")
    report["removed_conflicting_labels"] = int((n_labels > 1).sum())
    work = work[n_labels == 1].reset_index(drop=True)

    if work.empty:
        raise DatasetError("No usable records remain after cleaning. "
                           "Check the text and label column settings.")
    report["rows_final"] = int(len(work))
    report["rows_removed_total"] = report["rows_initial"] - report["rows_final"]
    report["percent_removed"] = round(100 * report["rows_removed_total"] / report["rows_initial"], 2)
    report["class_counts_final"] = {k: int(v) for k, v in work["label"].value_counts().items()}
    report["missing_categories"] = [c for c in categories if c not in report["class_counts_final"]]
    if len(report["class_counts_final"]) < 2:
        raise DatasetError("Fewer than two categories remain after cleaning; "
                           "classification is not possible.")
    return work, report


def text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Character and word counts of the cleaned text, summarised per class."""
    stats = pd.DataFrame({
        "label": df["label"],
        "characters": df["clean_text"].str.len(),
        "words": df["clean_text"].str.split().str.len(),
    })
    return stats.groupby("label")[["characters", "words"]].describe().round(1)

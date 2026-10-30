"""Dataset discovery, loading, validation and (optional) download.

The reference dataset (Kaggle: ``saurabhshahane/ecommerce-text-classification``,
file ``ecommerceDataset.csv``) has NO header row: column 1 is the category and
column 2 is the product description. Files with a header row and different
column names are supported too.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

import config


class DatasetError(Exception):
    """Raised for any problem with the dataset; the message is user-friendly."""


# --------------------------------------------------------------------------- #
# Instructions shown when the dataset is missing
# --------------------------------------------------------------------------- #
def missing_dataset_message(path: Path | str | None = None) -> str:
    where = f"'{path}'" if path else f"the folder '{config.DATA_DIR}'"
    return (
        f"Dataset not found: {where}.\n\n"
        "How to get it:\n"
        "  1. Automatic:  python main.py --download\n"
        "  2. Manual:     open https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification\n"
        "                 click Download, unzip, and copy 'ecommerceDataset.csv' into the 'data' folder.\n"
        "  3. Kaggle CLI: kaggle datasets download -d saurabhshahane/ecommerce-text-classification -p data --unzip\n"
        "                 (needs a Kaggle API token, see README.md)\n\n"
        "Then run:  python main.py --data data/ecommerceDataset.csv"
    )


def find_dataset(data_dir: Path | str = config.DATA_DIR) -> Path | None:
    """Return the default dataset file, or the only CSV in ``data_dir``."""
    data_dir = Path(data_dir)
    default = data_dir / config.DEFAULT_DATASET_FILENAME
    if default.is_file():
        return default
    csv_files = sorted(data_dir.glob("*.csv")) if data_dir.is_dir() else []
    return csv_files[0] if len(csv_files) == 1 else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _looks_like_label(value) -> bool:
    key = re.sub(r"[^a-z0-9]+", " ", str(value).lower().replace("&", " and ")).strip()
    known = {re.sub(r"[^a-z0-9]+", " ", c.lower().replace("&", " and ")).strip()
             for c in config.CATEGORIES}
    return key in known or key in {"clothing", "clothing accessories"}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DatasetError(f"Could not decode '{path}' as UTF-8 or Latin-1: {last_error}")


def load_dataset(path: Path | str, text_column: str | None = None,
                 label_column: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Load a CSV file and return ``(dataframe, text_column, label_column)``."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(missing_dataset_message(path))
    try:
        first_row = _read_csv(path, header=None, nrows=1)
        if first_row.empty:
            raise DatasetError(f"The file '{path}' is empty.")
        headerless = first_row.shape[1] == 2 and _looks_like_label(first_row.iloc[0, 0])
        if headerless:
            df = _read_csv(path, header=None, names=["label", "text"])
        else:
            df = _read_csv(path)
    except pd.errors.EmptyDataError:
        raise DatasetError(f"The file '{path}' is empty.") from None
    except pd.errors.ParserError as exc:
        raise DatasetError(f"'{path}' is not a valid CSV file: {exc}") from None

    text_col, label_col = detect_columns(df, text_column, label_column)
    validate_dataset(df, text_col, label_col)
    return df, text_col, label_col


def detect_columns(df: pd.DataFrame, text_column: str | None = None,
                   label_column: str | None = None) -> tuple[str, str]:
    """Return ``(text_col, label_col)`` using explicit names, then auto-detection."""
    columns = [str(c) for c in df.columns]
    df.columns = columns
    lower = {c.lower(): c for c in columns}
    configured_text = text_column or config.TEXT_COLUMN
    configured_label = label_column or config.LABEL_COLUMN

    def explicit(name, kind):
        if name not in columns:
            raise DatasetError(
                f"The {kind} column '{name}' does not exist. Available columns: {columns}.\n"
                f"Use --{kind}-column NAME (or set {kind.upper()}_COLUMN in config.py).")
        return name

    def detect(candidates, kind):
        hits = [lower[c] for c in candidates if c in lower]
        if len(hits) > 1:
            raise DatasetError(
                f"Ambiguous {kind} column: several columns match {hits}.\n"
                f"Choose one with --{kind}-column NAME (or set {kind.upper()}_COLUMN in config.py).")
        return hits[0] if hits else None

    text_col = explicit(configured_text, "text") if configured_text else detect(config.TEXT_COLUMN_CANDIDATES, "text")
    label_col = explicit(configured_label, "label") if configured_label else detect(config.LABEL_COLUMN_CANDIDATES, "label")

    # Fallback for a two-column file with unknown names: the column with few
    # distinct values is the label, the other one is the text.
    if (text_col is None or label_col is None) and len(columns) == 2 and not (configured_text or configured_label):
        low, high = sorted(columns, key=lambda c: df[c].nunique(dropna=True))
        if df[low].nunique(dropna=True) <= 50:
            label_col, text_col = low, high

    if text_col is None or label_col is None:
        raise DatasetError(
            f"Could not detect the text and label columns. Available columns: {columns}.\n"
            "Specify them with --text-column NAME --label-column NAME "
            "(or set TEXT_COLUMN / LABEL_COLUMN in config.py).")
    if text_col == label_col:
        raise DatasetError("The text and label columns must be different columns.")
    return text_col, label_col


def validate_dataset(df: pd.DataFrame, text_column: str, label_column: str,
                     min_rows: int = 10) -> None:
    """Raise ``DatasetError`` if the data cannot be used for classification."""
    for col in (text_column, label_column):
        if col not in df.columns:
            raise DatasetError(f"Column '{col}' not found. Available columns: {list(df.columns)}.")
    if len(df) < min_rows:
        raise DatasetError(f"The dataset has only {len(df)} rows; at least {min_rows} are required.")
    if df[text_column].notna().sum() == 0:
        raise DatasetError(f"The text column '{text_column}' contains no values.")
    if df[label_column].nunique(dropna=True) < 2:
        raise DatasetError(f"The label column '{label_column}' needs at least two distinct classes.")


# --------------------------------------------------------------------------- #
# Optional automatic download
# --------------------------------------------------------------------------- #
def _kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    home = Path.home()
    return (home / ".kaggle" / "kaggle.json").is_file() or (home / ".kaggle" / "access_token").is_file()


def _download_kaggle(dest_dir: Path) -> None:
    if shutil.which("kaggle") is None:
        raise DatasetError("Kaggle CLI is not installed (pip install kaggle).")
    if not _kaggle_credentials_available():
        raise DatasetError("No Kaggle credentials found (see README.md, 'Kaggle API token').")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET_ID,
         "-p", str(dest_dir), "--unzip"],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise DatasetError(f"Kaggle CLI failed: {(result.stderr or result.stdout).strip()}")


def _download_url(url: str, target: Path) -> None:
    part = target.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out)
        part.replace(target)
    except Exception as exc:  # network errors vary widely
        part.unlink(missing_ok=True)
        raise DatasetError(f"{type(exc).__name__}: {exc}") from None


def download_dataset(dest_dir: Path | str = config.DATA_DIR) -> Path:
    """Try Kaggle CLI, then the official Zenodo record, then a GitHub mirror.

    Each downloaded file is checked with ``load_dataset`` before it is accepted.
    Raises ``DatasetError`` listing every failed attempt.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / config.DEFAULT_DATASET_FILENAME
    attempts = [
        ("Kaggle CLI", lambda: _download_kaggle(dest_dir)),
        ("Zenodo (original source)", lambda: _download_url(config.ZENODO_URL, target)),
        ("GitHub mirror (sugatagh)", lambda: _download_url(config.GITHUB_MIRROR_URL, target)),
    ]
    errors = []
    for name, action in attempts:
        try:
            action()
            load_dataset(target)  # validate structure
            return target
        except DatasetError as exc:
            errors.append(f"  - {name}: {exc}")
            if target.is_file() and name != "Kaggle CLI":
                target.unlink(missing_ok=True)
    raise DatasetError("Automatic download failed:\n" + "\n".join(errors) + "\n\n" + missing_dataset_message())

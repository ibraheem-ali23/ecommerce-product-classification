"""Shared test fixtures. Synthetic data is used for UNIT TESTS ONLY."""
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_VOCAB = {
    "Electronics": ["bluetooth", "battery", "usb", "wireless", "screen", "charger", "camera", "led"],
    "Household": ["kitchen", "steel", "cotton", "bedsheet", "storage", "cooker", "curtain", "mop"],
    "Books": ["novel", "author", "paperback", "chapters", "edition", "story", "publisher", "pages"],
    "Clothing & Accessories": ["shirt", "fabric", "slim", "wallet", "sleeves", "denim", "cap", "socks"],
}


def make_synthetic(per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    """Tiny fake dataset (label, text); every row is unique."""
    rng = random.Random(seed)
    rows = []
    for label, words in _VOCAB.items():
        for i in range(per_class):
            text = " ".join(rng.choices(words, k=6)) + f" item {i} {label[:3].lower()}"
            rows.append((label, text))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["label", "text"])


@pytest.fixture
def synthetic_df():
    return make_synthetic()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    path = tmp_path / "synthetic.csv"
    synthetic_df.to_csv(path, header=False, index=False)  # same layout as the Kaggle file
    return path

"""Loading the saved model and predicting categories for new descriptions."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

import config


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist or cannot be loaded."""


# ILLUSTRATIVE examples written for demonstrations. They are NOT dataset records.
EXAMPLE_DESCRIPTIONS = {
    "Wireless headphones": "Bluetooth wireless over-ear headphones with active noise cancellation, "
                           "40 hour battery life, USB-C fast charging and built-in microphone.",
    "Cookware set": "Non-stick aluminium cookware set with frying pan, saucepan and lid, "
                    "suitable for gas and induction stove, easy to clean kitchen essentials.",
    "Novel": "A gripping novel about two families and a secret that spans three generations. "
             "Paperback edition, 352 pages, by an award winning author.",
    "Cotton t-shirt": "Men's slim fit round neck cotton t-shirt, soft breathable fabric, "
                      "available in multiple colours and sizes, machine washable.",
    "Smart watch": "Fitness smart watch with heart rate monitor, GPS, sleep tracking, "
                   "AMOLED display and water resistance up to 50 metres.",
    "Bed sheet": "Double bed cotton bedsheet with two pillow covers, 300 thread count, "
                 "floral print, fade resistant home furnishing.",
}


def default_model_path() -> Path:
    return config.MODELS_DIR / config.BEST_MODEL_FILENAME


def load_metadata(model_path: Path | str | None = None) -> dict:
    meta_path = Path(model_path or default_model_path()).parent / config.MODEL_METADATA_FILENAME
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def load_model(model_path: Path | str | None = None):
    """Load the saved pipeline or raise ``ModelNotFoundError`` with instructions."""
    path = Path(model_path or default_model_path())
    if not path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found at '{path}'. Train it first with:\n"
            "    python main.py --data data/ecommerceDataset.csv")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
    except Exception as exc:  # corrupted file or incompatible library version
        raise ModelNotFoundError(
            f"The model file '{path}' could not be loaded ({type(exc).__name__}: {exc}).\n"
            "It may have been created with a different scikit-learn version. "
            "Retrain it with: python main.py --data data/ecommerceDataset.csv") from None
    return model


def version_warning(model_path: Path | str | None = None) -> str | None:
    """A warning text if the model was saved with another scikit-learn version."""
    saved = load_metadata(model_path).get("sklearn_version")
    if saved and saved != sklearn.__version__:
        return (f"The model was trained with scikit-learn {saved}, but {sklearn.__version__} is "
                "installed. Predictions usually still work; retrain with `python main.py` if you see errors.")
    return None


def predict_text(model, text: str) -> dict:
    """Predict the category of one description.

    Returns ``{"label": str, "scores": {class: value}, "score_type": str}`` where
    ``score_type`` is ``"probability"`` (models with predict_proba) or
    ``"decision value"`` (LinearSVC; larger = more confident, not a probability).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty product description.")
    label = str(model.predict([text])[0])
    classes = [str(c) for c in model.classes_]
    if hasattr(model, "predict_proba"):
        values, score_type = model.predict_proba([text])[0], "probability"
    elif hasattr(model, "decision_function"):
        values, score_type = np.ravel(model.decision_function([text])), "decision value"
    else:
        return {"label": label, "scores": {}, "score_type": "none"}
    return {"label": label, "scores": {c: float(v) for c, v in zip(classes, values)},
            "score_type": score_type}

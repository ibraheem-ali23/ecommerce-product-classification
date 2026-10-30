"""Model construction, splitting, training and persistence.

Every model is a scikit-learn ``Pipeline`` with two steps::

    tfidf -> clf

The TF-IDF vectorizer uses ``clean_text`` as its preprocessor, so the whole
pipeline accepts RAW product descriptions. Because the vectorizer is a pipeline
step, it is fitted on the training data only (no data leakage).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.data_loader import DatasetError
from src.preprocessing import clean_text

MODEL_NAMES = ["Naive Bayes", "Logistic Regression", "Linear SVM"]


def _make_classifier(name: str, seed: int, params: dict | None = None):
    params = {**config.MODEL_PARAMS.get(name, {}), **(params or {})}
    if name == "Naive Bayes":
        return MultinomialNB(**params)
    if name == "Logistic Regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "Linear SVM":
        return LinearSVC(random_state=seed, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(name: str, tfidf_params: dict | None = None,
                   seed: int = config.RANDOM_SEED) -> Pipeline:
    """Create an unfitted ``tfidf -> clf`` pipeline for the model ``name``."""
    tfidf = TfidfVectorizer(preprocessor=clean_text,
                            **{**config.TFIDF_PARAMS, **(tfidf_params or {})})
    return Pipeline([("tfidf", tfidf), ("clf", _make_classifier(name, seed))])


def split_data(df, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_SEED):
    """Stratified train/test split of RAW text. Returns X_train, X_test, y_train, y_test."""
    try:
        return train_test_split(df["text"], df["label"], test_size=test_size,
                                random_state=seed, stratify=df["label"])
    except ValueError as exc:
        raise DatasetError(f"Could not create a stratified split: {exc} "
                           "(each class needs at least 2 samples).") from None


def train_models(X_train, y_train, model_names=None, tfidf_params: dict | None = None,
                 seed: int = config.RANDOM_SEED):
    """Fit all models on the same training data.

    Returns ``(models, fit_seconds)`` - two dictionaries keyed by model name.
    """
    models, seconds = {}, {}
    for name in (model_names or MODEL_NAMES):
        pipeline = build_pipeline(name, tfidf_params, seed)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        seconds[name] = round(time.perf_counter() - start, 2)
        models[name] = pipeline
    return models, seconds


def cross_validated_macro_f1(X_train, y_train, model_names=None, folds: int = config.CV_FOLDS,
                             tfidf_params: dict | None = None, seed: int = config.RANDOM_SEED):
    """Stratified k-fold macro-F1 on the TRAINING data only (mean, std per model)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {}
    for name in (model_names or MODEL_NAMES):
        values = cross_val_score(build_pipeline(name, tfidf_params, seed), X_train, y_train,
                                 cv=cv, scoring="f1_macro")
        scores[name] = (float(values.mean()), float(values.std()))
    return scores


def save_model(pipeline: Pipeline, model_path: Path | str, metadata: dict | None = None) -> Path:
    """Save the fitted pipeline with joblib and a small JSON metadata file next to it."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "classes": [str(c) for c in pipeline.classes_],
        **(metadata or {}),
    }
    meta_path = model_path.parent / config.MODEL_METADATA_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path

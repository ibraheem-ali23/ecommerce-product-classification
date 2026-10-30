import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import config
from src import evaluate, predict, train
from src.preprocessing import prepare_dataframe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(synthetic_df):
    df, _ = prepare_dataframe(synthetic_df, "text", "label")
    return df


@pytest.fixture
def fitted(prepared):
    X_train, X_test, y_train, y_test = train.split_data(prepared, 0.25, 7)
    models, seconds = train.train_models(X_train, y_train, tfidf_params={"min_df": 1, "max_features": 500})
    return models, seconds, X_train, X_test, y_train, y_test


def test_split_is_stratified_and_reproducible(prepared):
    a = train.split_data(prepared, 0.25, 7)
    b = train.split_data(prepared, 0.25, 7)
    assert list(a[1].index) == list(b[1].index)
    counts = a[3].value_counts()
    assert counts.max() - counts.min() <= 1  # classes are equally represented in the test set
    assert not set(a[0].index) & set(a[1].index)


def test_every_model_is_a_tfidf_pipeline(fitted):
    models = fitted[0]
    assert list(models) == train.MODEL_NAMES
    for pipe in models.values():
        assert list(pipe.named_steps) == ["tfidf", "clf"]


def test_vectorizer_fitted_on_training_data_only(fitted):
    models, _, X_train, X_test, *_ = fitted
    vocab = models["Naive Bayes"].named_steps["tfidf"].vocabulary_
    assert "bluetooth" in vocab
    from src.preprocessing import clean_text
    train_tokens = {t for doc in X_train for t in clean_text(doc).split()}
    assert all(all(tok in train_tokens for tok in term.split()) for term in vocab)


def test_predictions_are_valid_labels(fitted):
    models, _, _, X_test, _, y_test = fitted
    for pipe in models.values():
        pred = pipe.predict(X_test)
        assert len(pred) == len(X_test) and set(pred) <= set(config.CATEGORIES)


def test_evaluation_table_and_selection(fitted):
    models, seconds, _, X_test, _, y_test = fitted
    labels = config.CATEGORIES
    results, preds, reports = evaluate.evaluate_models(models, X_test, y_test, labels, seconds)
    assert set(results.index) == set(train.MODEL_NAMES)
    metrics = results[evaluate.METRIC_COLUMNS]
    assert ((metrics >= 0) & (metrics <= 1)).all().all()
    best = evaluate.select_best_model(results)
    assert results.loc[best, "f1_macro"] == results["f1_macro"].max()
    assert all("precision" in r for r in reports.values())
    with pytest.raises(ValueError):
        evaluate.select_best_model(results, "nonsense")


def test_top_features_available_for_all_models(fitted):
    for pipe in fitted[0].values():
        top = evaluate.top_features_per_class(pipe, 5)
        assert top.shape == (5, 4)


@pytest.mark.parametrize("name", train.MODEL_NAMES)
def test_save_and_reload_gives_identical_predictions(fitted, tmp_path, name):
    models, _, _, X_test, *_ = fitted
    path = train.save_model(models[name], tmp_path / "m.joblib", {"model_name": name})
    assert path.is_file() and (tmp_path / config.MODEL_METADATA_FILENAME).is_file()
    assert json.loads((tmp_path / config.MODEL_METADATA_FILENAME).read_text())["model_name"] == name
    reloaded = predict.load_model(path)
    assert (reloaded.predict(X_test) == models[name].predict(X_test)).all()


def test_predict_text_output(fitted):
    models = fitted[0]
    for name, pipe in models.items():
        out = predict.predict_text(pipe, "bluetooth usb wireless charger battery")
        assert out["label"] in config.CATEGORIES
        assert set(out["scores"]) == set(config.CATEGORIES)
        if name == "Linear SVM":
            assert out["score_type"] == "decision value"
        else:
            assert out["score_type"] == "probability"
            assert abs(sum(out["scores"].values()) - 1) < 1e-6


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_predict_text_rejects_invalid_input(fitted, bad):
    with pytest.raises(ValueError):
        predict.predict_text(fitted[0]["Naive Bayes"], bad)


def test_missing_and_corrupt_model_files(tmp_path):
    with pytest.raises(predict.ModelNotFoundError, match="main.py"):
        predict.load_model(tmp_path / "nope.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    with pytest.raises(predict.ModelNotFoundError, match="could not be loaded"):
        predict.load_model(bad)


def test_cross_validation_runs(prepared):
    X_train, _, y_train, _ = train.split_data(prepared, 0.25, 7)
    scores = train.cross_validated_macro_f1(X_train, y_train, ["Naive Bayes"], folds=3,
                                            tfidf_params={"min_df": 1})
    assert 0 <= scores["Naive Bayes"][0] <= 1


def _run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=300)


def test_cli_help():
    result = _run_cli("--help")
    assert result.returncode == 0 and "--data" in result.stdout


def test_cli_missing_dataset_exits_gracefully(tmp_path):
    result = _run_cli("--data", str(tmp_path / "missing.csv"))
    assert result.returncode == 2
    assert "Dataset" in result.stdout and "Traceback" not in result.stderr


def test_cli_end_to_end_on_synthetic_data(synthetic_csv, tmp_path):
    out = {k: tmp_path / k for k in ("models", "reports", "outputs")}
    result = _run_cli("--data", str(synthetic_csv), "--cv-folds", "3", "--max-features", "300",
                      "--models-dir", str(out["models"]), "--reports-dir", str(out["reports"]),
                      "--outputs-dir", str(out["outputs"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out["models"] / config.BEST_MODEL_FILENAME).is_file()
    assert (out["reports"] / "model_comparison.csv").is_file()
    assert (out["reports"] / "results_report.md").is_file()
    assert (out["outputs"] / "model_comparison.png").is_file()
    assert len(list(out["outputs"].glob("confusion_matrix_*.png"))) == 3
    model = joblib.load(out["models"] / config.BEST_MODEL_FILENAME)
    assert np.asarray(model.predict(["novel author paperback"])).shape == (1,)

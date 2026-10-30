"""Evaluation metrics, plots, interpretability helpers and report writing.

Plot functions save the figure when ``path`` is given and return the figure.
Callers decide when to close it (``plt.close(fig)``) or show it (notebook).
No score in this project is hard-coded: everything is computed from predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config

METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "precision_weighted", "recall_weighted", "f1_weighted"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro- and weighted-averaged precision, recall and F1."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        out.update({f"precision_{average}": p, f"recall_{average}": r, f"f1_{average}": f})
    return {k: float(v) for k, v in out.items()}


def evaluate_models(models: dict, X_test, y_test, labels, fit_seconds: dict | None = None):
    """Evaluate fitted pipelines on the same test set.

    Returns ``(results_df, predictions, reports)``:
    a comparison table sorted by macro-F1, the predictions per model and the
    text classification report per model.
    """
    rows, predictions, reports = [], {}, {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append({"model": name, **compute_metrics(y_test, y_pred)})
        reports[name] = classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0)
        if fit_seconds and name in fit_seconds:
            rows[-1]["fit_seconds"] = fit_seconds[name]
    results = pd.DataFrame(rows).set_index("model")
    return results.sort_values(["f1_macro", "accuracy"], ascending=False), predictions, reports


def select_best_model(results: pd.DataFrame, metric: str = config.SELECTION_METRIC) -> str:
    """Name of the model with the highest ``metric`` (ties broken by accuracy)."""
    if metric not in results.columns:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(results.columns)}")
    return results.sort_values([metric, "accuracy"], ascending=False).index[0]


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support as a DataFrame."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.loc[list(labels) + ["macro avg", "weighted avg"]]


def most_confused_pair(y_true, y_pred, labels):
    """The (true, predicted, count) with the most off-diagonal errors, or None."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    if cm.max() == 0:
        return None
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    return labels[i], labels[j], int(cm[i, j])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(counts: pd.Series, path=None, title="Class distribution"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False,
                palette="viridis", ax=ax)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    ax.set(title=title, xlabel="Category", ylabel="Number of samples")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, path)


def plot_text_length_distribution(df: pd.DataFrame, path=None):
    words = df["clean_text"].str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(words.clip(upper=words.quantile(0.99)), bins=50, color="steelblue")
    axes[0].set(title="Words per description (clipped at 99th percentile)",
                xlabel="Number of words", ylabel="Descriptions")
    plot_df = pd.DataFrame({"label": df["label"], "words": words.clip(upper=words.quantile(0.99))})
    sns.boxplot(data=plot_df, x="label", y="words", hue="label", legend=False,
                palette="viridis", ax=axes[1])
    axes[1].set(title="Words per description by category", xlabel="Category", ylabel="Number of words")
    axes[1].tick_params(axis="x", rotation=15)
    return _save(fig, path)


def plot_model_comparison(results: pd.DataFrame, path=None):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    tidy = results[metrics].reset_index().melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=tidy, x="metric", y="score", hue="model", palette="Set2", ax=ax)
    low = max(0.0, float(tidy["score"].min()) - 0.05)
    ax.set(title="Model comparison on the test set", xlabel="Metric",
           ylabel=f"Score (y-axis starts at {low:.2f})",
           ylim=(low, min(1.0, float(tidy["score"].max()) + 0.03)))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion matrix", path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar=False, ax=axes[0])
    axes[0].set(title=f"{title} (counts)", xlabel="Predicted", ylabel="True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[1])
    axes[1].set(title=f"{title} (row-normalised = recall)", xlabel="Predicted", ylabel="True")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Interpretability and error analysis
# --------------------------------------------------------------------------- #
def top_features_per_class(pipeline, n: int = 10) -> pd.DataFrame:
    """Most informative TF-IDF terms per class.

    Linear models: largest positive coefficients. Naive Bayes: log-probability
    of the term in the class minus its mean log-probability in the other classes.
    Returns an empty frame for models without such weights.
    """
    clf = pipeline.named_steps["clf"]
    terms = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = np.atleast_2d(clf.coef_)
        if weights.shape[0] == 1 and len(clf.classes_) == 2:
            weights = np.vstack([-weights[0], weights[0]])
    elif hasattr(clf, "feature_log_prob_"):
        logp = clf.feature_log_prob_
        weights = np.vstack([logp[i] - np.delete(logp, i, axis=0).mean(axis=0)
                             for i in range(logp.shape[0])])
    else:
        return pd.DataFrame()
    return pd.DataFrame({str(cls): terms[np.argsort(weights[i])[::-1][:n]]
                         for i, cls in enumerate(clf.classes_)})


def classification_examples(X_test, y_test, y_pred, n: int = 5, seed: int = config.RANDOM_SEED):
    """Random correctly and incorrectly classified test samples (text shortened)."""
    frame = pd.DataFrame({"description": pd.Series(X_test).astype(str).str.slice(0, 220).values,
                          "true": np.asarray(y_test), "predicted": np.asarray(y_pred)})
    correct = frame[frame["true"] == frame["predicted"]]
    wrong = frame[frame["true"] != frame["predicted"]]
    return (correct.sample(min(n, len(correct)), random_state=seed).reset_index(drop=True),
            wrong.sample(min(n, len(wrong)), random_state=seed).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def save_reports(results, reports, predictions, y_test, labels, best_model, data_report,
                 cv_scores, top_features, reports_dir, outputs_dir, plots: bool = True,
                 settings: dict | None = None) -> list[Path]:
    """Write CSV/TXT/JSON/Markdown reports (and plots) of an actual run."""
    reports_dir, outputs_dir = Path(reports_dir), Path(outputs_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    written = []

    table = results.copy()
    if cv_scores:
        table["cv_f1_macro_mean"] = pd.Series({k: v[0] for k, v in cv_scores.items()})
        table["cv_f1_macro_std"] = pd.Series({k: v[1] for k, v in cv_scores.items()})
    path = reports_dir / "model_comparison.csv"
    table.round(4).to_csv(path)
    written.append(path)

    for name, text in reports.items():
        path = reports_dir / f"classification_report_{_slug(name)}.txt"
        path.write_text(f"Model: {name}\n\n{text}", encoding="utf-8")
        written.append(path)

    path = reports_dir / "data_report.json"
    path.write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    written.append(path)

    path = reports_dir / "top_features_best_model.csv"
    top_features.to_csv(path, index=False)
    written.append(path)

    path = reports_dir / "results_report.md"
    path.write_text(build_results_markdown(table, reports, best_model, data_report, labels,
                                           top_features, settings or {}), encoding="utf-8")
    written.append(path)

    if plots:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for name, y_pred in predictions.items():
            fig = plot_confusion_matrix(y_test, y_pred, labels, title=name,
                                        path=outputs_dir / f"confusion_matrix_{_slug(name)}.png")
            plt.close(fig)
            written.append(outputs_dir / f"confusion_matrix_{_slug(name)}.png")
        fig = plot_model_comparison(results, outputs_dir / "model_comparison.png")
        plt.close(fig)
        written.append(outputs_dir / "model_comparison.png")
    return written


def markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency needed)."""
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def build_results_markdown(table, reports, best_model, data_report, labels, top_features, settings) -> str:
    """Markdown results report. Every number comes from the run that produced it."""
    lines = ["# Results report (generated automatically)", "",
             "All numbers below were MEASURED by running `main.py` on the dataset named in "
             "the settings. Nothing in this file is hard-coded.", ""]
    if settings:
        lines += ["## Run settings", ""] + [f"- **{k}**: {v}" for k, v in settings.items()] + [""]
    lines += ["## Data preparation", "",
              f"- Rows loaded: {data_report['rows_initial']:,}",
              f"- Removed (missing label): {data_report['removed_missing_label']:,}",
              f"- Removed (missing/blank description): {data_report['removed_missing_text']:,}",
              f"- Removed (unexpected label): {data_report['removed_unexpected_label']:,}",
              f"- Removed (exact duplicate records): {data_report['removed_exact_duplicates']:,}",
              f"- Removed (empty after cleaning): {data_report['removed_empty_after_cleaning']:,}",
              f"- Removed (duplicates after cleaning): {data_report['removed_duplicates_after_cleaning']:,}",
              f"- Removed (identical text, conflicting labels): {data_report['removed_conflicting_labels']:,}",
              f"- Rows used: {data_report['rows_final']:,} "
              f"({data_report['percent_removed']}% removed in total)", "",
              "| Category | Samples |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in data_report["class_counts_final"].items()]
    if data_report["missing_categories"]:
        lines += ["", f"**Warning:** categories absent from the data: {data_report['missing_categories']}"]
    lines += ["", "## Model comparison (test set)", "", markdown_table(table.round(4), index=True), "",
              f"**Best model by `{config.SELECTION_METRIC}`: {best_model}**", "",
              "## Classification reports", ""]
    for name, text in reports.items():
        lines += [f"### {name}", "", "```", text.rstrip(), "```", ""]
    if not top_features.empty:
        lines += [f"## Top terms per class ({best_model})", "", markdown_table(top_features, index=False), ""]
    return "\n".join(lines)

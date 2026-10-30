"""Command-line workflow: validate data -> preprocess -> train -> evaluate -> save.

Examples (Windows PowerShell, from the project folder):
    python main.py --download
    python main.py --data data/ecommerceDataset.csv
    python main.py --data my.csv --text-column description --label-column category
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI window needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_loader, evaluate, preprocessing, train  # noqa: E402
from src.data_loader import DatasetError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E-commerce product classification with TF-IDF "
                                            "(Naive Bayes, Logistic Regression, Linear SVM).")
    p.add_argument("--data", type=Path, help="Path to the dataset CSV (default: data/ecommerceDataset.csv).")
    p.add_argument("--download", action="store_true",
                   help="Download the dataset into the data folder first (Kaggle CLI, Zenodo, GitHub mirror).")
    p.add_argument("--text-column", help="Name of the description column (auto-detected if omitted).")
    p.add_argument("--label-column", help="Name of the category column (auto-detected if omitted).")
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE, help="Test fraction (default 0.2).")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed (default 42).")
    p.add_argument("--max-features", type=int, default=config.TFIDF_PARAMS["max_features"],
                   help="Maximum number of TF-IDF features (default 50000).")
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS,
                   help="Cross-validation folds on the training set (0 disables; default 5).")
    p.add_argument("--models-dir", type=Path, default=config.MODELS_DIR, help="Where to save the model.")
    p.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR, help="Where to save reports.")
    p.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR, help="Where to save plots.")
    p.add_argument("--no-plots", action="store_true", help="Skip saving plots.")
    return p


def run(args) -> int:
    if args.download:
        print("Downloading dataset ...")
        args.data = args.data or data_loader.download_dataset()
        print(f"  saved to {args.data}")
    data_path = args.data or data_loader.find_dataset()
    if data_path is None:
        print(data_loader.missing_dataset_message())
        return 2

    print(f"[1/6] Loading {data_path}")
    df, text_col, label_col = data_loader.load_dataset(data_path, args.text_column, args.label_column)
    print(f"      rows={len(df):,}  text column='{text_col}'  label column='{label_col}'")

    print("[2/6] Cleaning and preparing data")
    clean_df, report = preprocessing.prepare_dataframe(df, text_col, label_col)
    for key in ("removed_missing_label", "removed_missing_text", "removed_unexpected_label",
                "removed_exact_duplicates", "removed_empty_after_cleaning",
                "removed_duplicates_after_cleaning", "removed_conflicting_labels"):
        print(f"      {key.replace('_', ' ')}: {report[key]:,}")
    print(f"      rows used: {report['rows_final']:,} of {report['rows_initial']:,} "
          f"({report['percent_removed']}% removed)")
    if report["unexpected_labels"]:
        print(f"      WARNING unexpected labels removed: {report['unexpected_labels']}")
    if report["missing_categories"]:
        print(f"      WARNING categories missing from data: {report['missing_categories']}")
    labels = preprocessing.present_categories(clean_df["label"])

    print("[3/6] Stratified train/test split")
    X_train, X_test, y_train, y_test = train.split_data(clean_df, args.test_size, args.seed)
    print(f"      train={len(X_train):,}  test={len(X_test):,}  seed={args.seed}")

    tfidf_params = {"max_features": args.max_features}
    print("[4/6] Training models (TF-IDF + classifier pipelines)")
    models, seconds = train.train_models(X_train, y_train, tfidf_params=tfidf_params, seed=args.seed)

    cv_scores = None
    if args.cv_folds and args.cv_folds >= 2:
        print(f"      running {args.cv_folds}-fold cross-validation on the training set ...")
        cv_scores = train.cross_validated_macro_f1(X_train, y_train, folds=args.cv_folds,
                                                   tfidf_params=tfidf_params, seed=args.seed)

    print("[5/6] Evaluating on the test set")
    results, predictions, reports = evaluate.evaluate_models(models, X_test, y_test, labels, seconds)
    best = evaluate.select_best_model(results, config.SELECTION_METRIC)
    top_features = evaluate.top_features_per_class(models[best])
    settings = {"dataset file": data_path.name, "random seed": args.seed, "test size": args.test_size,
                "TF-IDF max features": args.max_features,
                "TF-IDF settings": {k: v for k, v in config.TFIDF_PARAMS.items() if k != "max_features"},
                "selection metric": config.SELECTION_METRIC,
                "train / test rows": f"{len(X_train):,} / {len(X_test):,}"}
    written = evaluate.save_reports(results, reports, predictions, y_test, labels, best, report,
                                    cv_scores, top_features, args.reports_dir, args.outputs_dir,
                                    plots=not args.no_plots, settings=settings)
    if not args.no_plots:
        df_plot_dir = Path(args.outputs_dir)
        fig = evaluate.plot_class_distribution(clean_df["label"].value_counts(),
                                               df_plot_dir / "class_distribution.png",
                                               "Class distribution (after cleaning)")
        plt.close(fig)
        fig = evaluate.plot_text_length_distribution(clean_df, df_plot_dir / "text_length_distribution.png")
        plt.close(fig)

    print("[6/6] Saving the best model")
    model_path = train.save_model(models[best], Path(args.models_dir) / config.BEST_MODEL_FILENAME, {
        "model_name": best, "selection_metric": config.SELECTION_METRIC,
        "test_metrics": results.loc[best].round(4).to_dict(), "random_seed": args.seed,
        "dataset_file": data_path.name, "rows_used": report["rows_final"]})

    shown = results.round(4).copy()
    if cv_scores:
        shown["cv_f1_macro"] = pd.Series({k: round(v[0], 4) for k, v in cv_scores.items()})
    print("\n=== RESULTS (test set, measured in this run) ===")
    print(shown[["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                + (["cv_f1_macro"] if cv_scores else []) + ["fit_seconds"]].to_string())
    print(f"\nBest model by {config.SELECTION_METRIC}: {best}")
    print("\nPer-class report of the best model:\n" + reports[best])
    print(f"Saved model: {model_path}")
    print(f"Reports in : {Path(args.reports_dir)}  ({len(written)} files written, incl. plots)")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # avoid Windows console encoding crashes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    try:
        code = run(args)
    except DatasetError as exc:
        print(f"\nDataset problem:\n{exc}")
        return 2
    if code == 0:
        print(f"Done in {time.perf_counter() - start:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())

# E-commerce Product Classification Using NLP

Classify e-commerce product descriptions into four categories using **text only**:
**Electronics**, **Household**, **Books**, **Clothing & Accessories**.

Text is represented with **TF-IDF** (unigrams + bigrams) and classified with three models, each built as a scikit-learn `Pipeline`: **Multinomial Naive Bayes**, **Logistic Regression** and **Linear SVM (LinearSVC)**. The models are compared on one shared, stratified 80/20 split, mainly by **macro-averaged F1**.

---

## 1. Results at a glance

> **Actual measured results.** The table below was produced by running `python main.py` on the real dataset while this package was prepared (Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, seed 42, Linux). Re-running on your computer should give the same or nearly the same numbers; small differences are possible with other library versions. Everything is regenerated into `reports/` each time you run the pipeline.

Data: 50,425 rows loaded, **27,709 used** (22,716 removed = 45.05%, of which 22,622 exact duplicate records, 93 duplicates that appear only after text cleaning and 1 missing description). Train 22,167 / test 5,542.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | 5-fold CV macro F1 (train only) |
|---|---:|---:|---:|---:|---:|
| **Linear SVM** | 0.9592 | 0.9606 | 0.9572 | **0.9589** | 0.9585 |
| Logistic Regression | 0.9580 | 0.9597 | 0.9553 | 0.9574 | 0.9561 |
| Naive Bayes | 0.9498 | 0.9514 | 0.9483 | 0.9496 | 0.9462 |

Best model by macro F1: **Linear SVM**. Differences between the two linear models are small (about 0.0015 macro F1) and should not be over-interpreted from a single split. Details: `reports/results_report.md`, `reports/classification_report_*.txt`, and the executed notebook.

The three categories of text in this repository are kept separate:

| Kind | Where |
|---|---|
| **Actual measured results** | this section, `reports/`, `outputs/`, executed notebook outputs |
| **Expected project behaviour** | the descriptions in sections 3 to 9 below (what the commands should do) |
| **Illustrative sample predictions** | `EXAMPLE_DESCRIPTIONS` in `src/predict.py`, notebook section 12, app examples. They were written by hand and are *not* dataset records or evidence of performance. |

---

## 2. Dataset

- **Name:** E-commerce Text Classification (Zenodo title: *E commerce text dataset*, version 2).
- **Kaggle:** https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification
- **Original source:** https://doi.org/10.5281/zenodo.3355823 (scraped from Indian e-commerce websites).
- **Reference notebook of the assignment:** https://www.kaggle.com/code/sugataghosh/e-commerce-text-classification-tf-idf-word2vec
- **File:** `ecommerceDataset.csv`, about 37 MB, 50,425 rows, **no header row**; column 1 = category, column 2 = description.

The CSV is **not** included in the ZIP (size). Put it in `data/ecommerceDataset.csv`:

```powershell
python main.py --download
```

This tries, in order: Kaggle CLI (if installed and configured), the Zenodo record, a public GitHub mirror (`sugatagh/E-commerce-Text-Classification`). The downloaded file is validated before it is accepted. Or download it manually from Kaggle, unzip it and copy `ecommerceDataset.csv` into `data/`.

**Provenance note.** When this package was built, Kaggle and Zenodo were not reachable from the build environment and no Kaggle credentials existed, so the dataset used for the reported results came from the GitHub mirror above. That file has 50,425 rows, four classes, and class counts identical to the published description (Household 19,313, Books 11,820, Electronics 10,621, Clothing & Accessories 8,671); removing exact duplicates leaves 27,802 rows, the same figure that independent third-party write-ups report for this dataset. Its MD5 checksum (`569691aa9d2bdfa6b725fb1634a88232`) does not match the checksum listed on the Zenodo page for the data file (`6f2cfdb475096244916aeed6f776c1a6`), so byte-for-byte identity with the Zenodo file could not be confirmed. If you download the file from Kaggle, your numbers may therefore differ very slightly.

### Kaggle API token (optional, secure setup)
Only needed for the Kaggle CLI route. The built-in `--download` works without it.
1. Sign in to Kaggle, open *Settings* and create an API token (this downloads `kaggle.json`).
2. Windows: create the folder `C:\Users\<you>\.kaggle\` and move `kaggle.json` there. Alternatively set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` for the current session only: `$env:KAGGLE_USERNAME="..."; $env:KAGGLE_KEY="..."`. (If Kaggle gives you a newer-style access token, follow Kaggle's documentation; this project also recognises `KAGGLE_API_TOKEN`.)
3. `pip install kaggle`, then `kaggle datasets download -d saurabhshahane/ecommerce-text-classification -p data --unzip`.
4. **Never** paste the token into code, notebooks or chat, and never commit `kaggle.json` (it is listed in `.gitignore`).

### Expected format and column configuration
Any CSV with a text column and a label column works.
- Headerless two-column files whose first value is a known category (the Kaggle file) are detected automatically (`label`, `text`).
- Files with a header are detected by common names: text = `text, description, product_description, product, content, ...`; label = `label, category, class, target, ...`.
- If detection is ambiguous or impossible, the program stops with a clear message listing the available columns. Fix it with `--text-column NAME --label-column NAME` or by setting `TEXT_COLUMN` / `LABEL_COLUMN` in `config.py`.
- Label spellings such as `clothing and accessories` are normalised. Labels outside the four categories are reported and removed; categories that are missing from the data are reported (the program never pretends all four exist).

---

## 3. Project structure

```
ecommerce-product-classification/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- config.py                 # paths, seed, TF-IDF and model settings, column names
|-- main.py                   # command-line pipeline
|-- app.py                    # Streamlit app
|-- data/README.md            # put ecommerceDataset.csv here
|-- notebooks/
|   `-- ecommerce_product_classification.ipynb   # executed, with outputs
|-- src/
|   |-- data_loader.py        # find / load / validate / download the dataset
|   |-- preprocessing.py      # text cleaning, dataset preparation, statistics
|   |-- train.py              # pipelines, split, training, cross-validation, saving
|   |-- evaluate.py           # metrics, plots, top features, report writing
|   `-- predict.py            # load model, predict, example descriptions
|-- models/                   # best_model.joblib + model_metadata.json
|-- reports/                  # CSV / TXT / JSON / Markdown reports + report_template.md
|-- outputs/                  # PNG figures
`-- tests/                    # pytest suite (conftest.py, test_preprocessing.py, test_data_loader.py, test_pipeline.py)
```

---

## 4. Environment setup (Windows PowerShell)

Requires Python 3.10 or newer (check with `python --version`; if `python` is not found, try `py --version` and use `py` instead).

```powershell
cd path\to\ecommerce-product-classification

# 1. create a virtual environment
python -m venv .venv

# 2. activate it
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks the script, run once in this window:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

---

## 5. Training and evaluation

```powershell
python main.py --download                       # only if data\ecommerceDataset.csv is missing
python main.py --data data\ecommerceDataset.csv # full pipeline (about 2-3 minutes)
python main.py --help                           # all options
```

The command validates the data, cleans it and prints every removal, makes the stratified 80/20 split, trains the three pipelines, runs 5-fold cross-validation on the training set, evaluates on the test set, saves reports, plots and the best model, and prints a summary. Useful options:

| Option | Meaning |
|---|---|
| `--data PATH` | dataset CSV (default `data/ecommerceDataset.csv`) |
| `--text-column`, `--label-column` | column names if not auto-detected |
| `--test-size 0.2`, `--seed 42` | split fraction and random seed |
| `--max-features 50000` | TF-IDF vocabulary size |
| `--cv-folds 5` | cross-validation folds on training data (`0` = skip, faster) |
| `--models-dir`, `--reports-dir`, `--outputs-dir` | output locations |
| `--no-plots` | skip figures |

If the dataset is missing the program prints download instructions and exits with code 2 instead of a traceback.

Pipeline design (explains the "no data leakage" requirement): each model is `Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", classifier)])`. The text cleaner is the vectorizer's `preprocessor`, so the saved model takes raw text. `fit` learns vocabulary and IDF from the training split only. Default TF-IDF settings (`config.py`): English stop words, unigrams + bigrams, 50,000 features, `min_df=2`, `max_df=0.95`, sublinear TF. Model hyper-parameters are fixed defaults and were **not tuned on the test set**.

**Model selection.** The best model is the one with the highest macro F1 on the test set (ties broken by accuracy). Because only three untuned models are compared, the selection bias is very small; the cross-validation column (training data only) is an independent check.

---

## 6. How to interpret the metrics

- **Accuracy**: share of all test products classified correctly. Can look good when one class dominates (Household is the largest class).
- **Precision** (per class): of the products *predicted* as the class, the share that truly belong to it.
- **Recall** (per class): of the products that *truly* belong to the class, the share that was found.
- **F1-score**: harmonic mean of precision and recall.
- **Macro average**: plain mean over the four classes, so every class counts equally. This is the main comparison metric here.
- **Weighted average**: mean weighted by class size.
- **Confusion matrix**: rows = true class, columns = predicted class. The diagonal is correct predictions; large off-diagonal cells show which classes are confused. The figures show counts (left) and row-normalised values, i.e. recall (right).

---

## 7. Jupyter Notebook

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks\ecommerce_product_classification.ipynb
```

In the browser choose *Kernel > Restart & Run All*. The notebook needs `data\ecommerceDataset.csv` (see section 2), uses the modules in `src/`, and runs top to bottom in about 3-5 minutes (set `RUN_CV = False` in the setup cell to skip cross-validation). If the notebook cannot find the packages, select the kernel of your virtual environment; to register it: `python -m ipykernel install --user --name ecommerce-venv`. The notebook delivered in this package was executed on the real dataset and contains its outputs. It also (re)saves `models\best_model.joblib`.

---

## 8. Interactive app (Streamlit)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501). Type or choose an example description, click **Predict**, and see the category plus scores: probabilities for Naive Bayes / Logistic Regression, decision values for Linear SVM (distances to the SVM boundaries, *not* probabilities). The app loads `models\best_model.joblib` and never retrains. If the model is missing it shows an error with the command to create it. The built-in examples are clearly labelled as illustrative, not dataset records.

---

## 9. Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

The suite covers text cleaning, missing/invalid inputs, dataset validation and column detection, the split, training of all three pipelines, vectorizer fitted on training data only, evaluation and model selection, prediction output (and invalid input), saving and reloading, the command-line interface (help, missing dataset, full run) and the exported artifacts. **Synthetic data is used in the tests only** (generated in `tests/conftest.py`); it is never used for the reported results.

---

## 10. Output files

| Path | Created by | Content |
|---|---|---|
| `models/best_model.joblib`, `models/model_metadata.json` | `main.py`, notebook | best fitted pipeline and its metadata |
| `reports/model_comparison.csv` | `main.py` | all metrics per model |
| `reports/classification_report_<model>.txt` | `main.py` | per-class report per model |
| `reports/data_report.json` | `main.py` | counts for every cleaning step |
| `reports/top_features_best_model.csv` | `main.py` | most informative terms per class |
| `reports/results_report.md` | `main.py` | readable summary of the run |
| `reports/report_template.md` | included | blank academic report template |
| `outputs/*.png` | `main.py` | class distribution, text length, model comparison, 3 confusion matrices |

---

## 11. Verification performed on this package

| Check | Outcome |
|---|---|
| `python -m pytest` (Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2) | 46 passed |
| `python -m pytest` in a clean venv (pandas 2.3.3, scikit-learn 1.5.2, numpy 2.2.6) | 46 passed |
| Full `python main.py` on the real dataset | completed, artifacts written |
| `python main.py --help` | works (also covered by a test) |
| Notebook executed top to bottom on the real dataset; `nbformat` validation | no error outputs; valid |
| Streamlit app via `streamlit.testing` (load example, predict, empty input, missing model) | worked as intended |
| Saved model reloaded: identical predictions | yes |

**Not verified:** execution on Windows itself (all checks ran on Linux; the code uses `pathlib`, UTF-8 file handling and ASCII-safe console output to be Windows-friendly), the Kaggle CLI and Zenodo download routes (unreachable from the build environment; the GitHub route was tested), and Python 3.10/3.11 interpreters (3.12 was used).

---

## 12. Troubleshooting

| Problem | Solution |
|---|---|
| `Dataset not found` | Run `python main.py --download` or copy the CSV to `data\ecommerceDataset.csv`. |
| `Could not detect the text and label columns` / `Ambiguous ... column` | Pass `--text-column` and `--label-column` (names are listed in the message). |
| `python` not recognised | Use `py`, or reinstall Python with "Add python.exe to PATH". |
| Activation blocked in PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `streamlit` / `jupyter` not recognised | The virtual environment is not active; run `.\.venv\Scripts\Activate.ps1`. |
| App says the model is unavailable or cannot be loaded | Run `python main.py` (a model file is a pickle: it is best loaded with the same scikit-learn version that created it). |
| Warning about a different scikit-learn version | Usually harmless; retrain with `python main.py` to remove it. |
| `ModuleNotFoundError: src` / `config` | Run commands from the project folder; the notebook finds the folder itself. |
| Notebook kernel lacks packages | Choose the `.venv` kernel (see section 7). |
| Download fails (firewall, offline) | Download manually from Kaggle (section 2). |
| Training seems slow | Use `--cv-folds 0` or a smaller `--max-features`. |

---

## 13. Limitations and future improvements

**Limitations.** The data comes from Indian e-commerce sites (style and vocabulary may not transfer); Household is the largest class (imbalance); only four coarse categories; some descriptions are ambiguous or mislabelled, which limits the achievable score; TF-IDF cannot capture meaning, synonyms or unseen words; hyper-parameters are untuned and the final comparison uses a single split; 45% of the raw rows were exact duplicates and had to be removed, so the usable dataset is much smaller than the raw file suggests.

**Future improvements.** Tune `C`, `alpha`, n-gram range and vocabulary size with cross-validation; try `class_weight="balanced"` and character n-grams; as an *optional extension* (not a substitute for TF-IDF) try Word2Vec/sentence embeddings or a fine-tuned transformer; study misclassified products for label noise; evaluate on an independent dataset.

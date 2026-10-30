# E-commerce Product Classification Using NLP - Assignment Report

**Student:** ______________  **Course:** ______________  **Date:** ______________

> Fill in the blanks with numbers from `reports/results_report.md` and `reports/model_comparison.csv`
> (produced by your own run of `python main.py`). Do not write numbers that you did not measure.

## 1. Introduction
Describe the task (classify product descriptions into Electronics, Household, Books, Clothing & Accessories), why it matters for e-commerce search and catalogue management, and the aim of the assignment.

## 2. Dataset
- Source: Kaggle *E-commerce Text Classification* (Zenodo DOI 10.5281/zenodo.3355823).
- Rows loaded: ______  Rows used after cleaning: ______  (removed: ______, reasons: ______)
- Class counts after cleaning: Electronics ______, Household ______, Books ______, Clothing & Accessories ______

## 3. Preprocessing
Cleaning steps (HTML, URLs, whitespace, lowercase, punctuation), treatment of duplicates and missing values, and why numbers and technical terms were kept.

## 4. Method
- Split: 80% train / 20% test, stratified, seed ______.
- TF-IDF settings: unigrams + bigrams, max features ______, min_df ______, stop words ______.
- Models: Multinomial Naive Bayes, Logistic Regression, Linear SVM (each a Pipeline).
- Why TF-IDF is fitted on the training data only.

## 5. Results (measured)
| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | | | | |
| Logistic Regression | | | | |
| Linear SVM | | | | |

Best model (by macro F1): ______________
Per-class observations and confusion matrix discussion: ______________

## 6. Discussion
Why the models differ, which classes are hardest, examples of errors.

## 7. Limitations and Future Work
Dataset origin, class imbalance, label noise, no hyper-parameter tuning, ideas for improvement.

## 8. Conclusion

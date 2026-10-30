# Models folder

`python main.py` saves the selected best model here:

- `best_model.joblib` - the complete scikit-learn Pipeline (TF-IDF + classifier). It accepts raw product descriptions.
- `model_metadata.json` - model name, selection metric, test metrics, seed, scikit-learn version used for training.

The Streamlit app and `src/predict.py` load this file; they never retrain.
A joblib file should be loaded with the same scikit-learn version that created it. If loading fails on another computer, simply retrain with `python main.py`.

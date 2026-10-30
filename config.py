"""Central configuration for the E-commerce Product Classification project.

Every tunable value (paths, random seed, TF-IDF settings, model settings, column
names) lives here so that the command line tool, the notebook, the tests and the
Streamlit app all use exactly the same defaults.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (pathlib keeps everything Windows-friendly)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATASET_FILENAME = "ecommerceDataset.csv"
BEST_MODEL_FILENAME = "best_model.joblib"
MODEL_METADATA_FILENAME = "model_metadata.json"

# --------------------------------------------------------------------------- #
# Target categories
# --------------------------------------------------------------------------- #
CATEGORIES = ["Electronics", "Household", "Books", "Clothing & Accessories"]

# --------------------------------------------------------------------------- #
# Column configuration
# --------------------------------------------------------------------------- #
# Leave as None to auto-detect. Set to a column name to force a column.
TEXT_COLUMN = None
LABEL_COLUMN = None

# Names that are recognised automatically (case-insensitive).
TEXT_COLUMN_CANDIDATES = (
    "text", "description", "product_description", "product description",
    "product", "content", "item_description", "title",
)
LABEL_COLUMN_CANDIDATES = (
    "label", "category", "class", "target", "product_category",
    "category_name", "labels",
)

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.20          # 80 % training / 20 % testing
CV_FOLDS = 5              # extra sanity check on the TRAINING set only (0 = off)
SELECTION_METRIC = "f1_macro"   # metric used to choose the best model

TFIDF_PARAMS = {
    "stop_words": "english",
    "ngram_range": (1, 2),     # unigrams + bigrams
    "max_features": 50_000,
    "min_df": 2,               # ignore terms found in fewer than 2 documents
    "max_df": 0.95,            # ignore terms found in more than 95 % of documents
    "sublinear_tf": True,      # 1 + log(tf) damping
}

# Fixed, sensible defaults. They are NOT tuned on the test set.
MODEL_PARAMS = {
    "Naive Bayes": {"alpha": 0.1},
    "Logistic Regression": {"C": 10.0, "max_iter": 1000},
    "Linear SVM": {"C": 1.0},
}

# --------------------------------------------------------------------------- #
# Dataset sources (used only by `python main.py --download`)
# --------------------------------------------------------------------------- #
KAGGLE_DATASET_ID = "saurabhshahane/ecommerce-text-classification"
ZENODO_URL = "https://zenodo.org/records/3355823/files/ecommerceDataset.csv?download=1"
GITHUB_MIRROR_URL = (
    "https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/"
    "main/Dataset/ecommerceDataset.csv"
)

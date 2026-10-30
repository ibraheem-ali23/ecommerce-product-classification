# Data folder

Place the dataset here as **`data/ecommerceDataset.csv`**. The CSV file is not stored in the Git repository or in the submission ZIP (it is about 37 MB).

## Dataset
- Name: *E-commerce Text Classification* (also published as *E commerce text dataset*, version 2, on Zenodo).
- Kaggle: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification
- Original source: https://doi.org/10.5281/zenodo.3355823
- Content: 50,425 rows scraped from Indian e-commerce websites, four categories (Electronics, Household, Books, Clothing & Accessories).
- Format: CSV, **no header row**. Column 1 = category, column 2 = product description (product title and details).

## How to get it
1. **Automatic:** `python main.py --download` tries, in this order: the Kaggle CLI (only if installed and configured), the Zenodo record, and a public GitHub mirror (`sugatagh/E-commerce-Text-Classification`). The downloaded file is checked before it is accepted.
2. **Manual:** download from the Kaggle page above, unzip, copy `ecommerceDataset.csv` into this folder.
3. **Kaggle CLI:** `kaggle datasets download -d saurabhshahane/ecommerce-text-classification -p data --unzip` (requires a Kaggle API token, see the main README).

## Other CSV files
Any CSV with a text column and a label column works. Use `--text-column` and `--label-column` if the names are not detected automatically.

# Results report (generated automatically)

All numbers below were MEASURED by running `main.py` on the dataset named in the settings. Nothing in this file is hard-coded.

## Run settings

- **dataset file**: ecommerceDataset.csv
- **random seed**: 42
- **test size**: 0.2
- **TF-IDF max features**: 50000
- **TF-IDF settings**: {'stop_words': 'english', 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.95, 'sublinear_tf': True}
- **selection metric**: f1_macro
- **train / test rows**: 22,167 / 5,542

## Data preparation

- Rows loaded: 50,425
- Removed (missing label): 0
- Removed (missing/blank description): 1
- Removed (unexpected label): 0
- Removed (exact duplicate records): 22,622
- Removed (empty after cleaning): 0
- Removed (duplicates after cleaning): 93
- Removed (identical text, conflicting labels): 0
- Rows used: 27,709 (45.05% removed in total)

| Category | Samples |
|---|---:|
| Household | 10,558 |
| Books | 6,202 |
| Clothing & Accessories | 5,647 |
| Electronics | 5,302 |

## Model comparison (test set)

| model | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | fit_seconds | cv_f1_macro_mean | cv_f1_macro_std |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear SVM | 0.9592 | 0.9605 | 0.9572 | 0.9588 | 0.9593 | 0.9592 | 0.9592 | 10.1 | 0.9589 | 0.0019 |
| Logistic Regression | 0.9583 | 0.96 | 0.9557 | 0.9578 | 0.9584 | 0.9583 | 0.9583 | 14.07 | 0.9565 | 0.0024 |
| Naive Bayes | 0.9493 | 0.9508 | 0.9479 | 0.9492 | 0.9496 | 0.9493 | 0.9493 | 10.11 | 0.9464 | 0.0022 |

**Best model by `f1_macro`: Linear SVM**

## Classification reports

### Naive Bayes

```
                        precision    recall  f1-score   support

           Electronics     0.9247    0.9377    0.9311      1060
             Household     0.9401    0.9583    0.9491      2112
                 Books     0.9703    0.9202    0.9446      1241
Clothing & Accessories     0.9683    0.9752    0.9718      1129

              accuracy                         0.9493      5542
             macro avg     0.9508    0.9479    0.9492      5542
          weighted avg     0.9496    0.9493    0.9493      5542
```

### Logistic Regression

```
                        precision    recall  f1-score   support

           Electronics     0.9499    0.9302    0.9399      1060
             Household     0.9495    0.9702    0.9597      2112
                 Books     0.9638    0.9428    0.9532      1241
Clothing & Accessories     0.9770    0.9796    0.9783      1129

              accuracy                         0.9583      5542
             macro avg     0.9600    0.9557    0.9578      5542
          weighted avg     0.9584    0.9583    0.9583      5542
```

### Linear SVM

```
                        precision    recall  f1-score   support

           Electronics     0.9475    0.9368    0.9421      1060
             Household     0.9521    0.9688    0.9603      2112
                 Books     0.9645    0.9420    0.9531      1241
Clothing & Accessories     0.9779    0.9814    0.9797      1129

              accuracy                         0.9592      5542
             macro avg     0.9605    0.9572    0.9588      5542
          weighted avg     0.9593    0.9592    0.9592      5542
```

## Top terms per class (Linear SVM)

| Books | Clothing & Accessories | Electronics | Household |
|---|---|---|---|
| author | womens | camera | bathrobe |
| book | mens | laptop | door |
| fishing | girls | lens | chair |
| skateboard | boys | cable | vacuum |
| game | bra | screen | plastic |
| canoeing | sunglasses | projector | pillow |
| action | cotton | trail | bed |
| guide | women | bushing | lamp |
| training | unisex | binoculars | watt |
| review | wear | calculator | home |

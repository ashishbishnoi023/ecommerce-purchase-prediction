# E-Commerce Customer Purchase Prediction (Logistic Regression)

Sapphire IQ ML project: predict whether a ShopSmart customer will purchase during a session (0 = No Purchase, 1 = Purchase).

## Files
| File | Description |
|---|---|
| `Purchase_Prediction_Colab.ipynb` | Full notebook (EDA, preprocessing, model, evaluation, predictions) with outputs |
| `Purchase_Prediction_Colab.html` | Same notebook as a read-only HTML copy |
| `Ecommerce_Purchase_Prediction_DataSet.csv` | Dataset (10,000 records) |
| `purchase_model.pkl` | Trained model pipeline (`joblib.load`) |

## Approach
1. EDA: target balance, distributions, categorical purchase rates, correlations
2. Missing values: median (numeric) / mode (categorical) imputation
3. Outliers: IQR capping on continuous features
4. Encoding and scaling: one-hot (`drop_first`) and StandardScaler
5. Model: Logistic Regression in a leakage-safe sklearn `Pipeline`, stratified 80/20 split, 5-fold CV
6. Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

## Results (test set, 2,000 customers)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (threshold 0.50) | 0.705 | 0.728 | 0.869 | 0.792 | 0.741 |
| `class_weight="balanced"` | 0.672 | 0.798 | 0.661 | 0.723 | 0.741 |
| Tuned threshold (0.60) | 0.691 | 0.775 | 0.736 | 0.755 | 0.741 |

**Key drivers:** cart items, Premium membership, cart value, session duration. Email traffic converts best.

## Run
Open the notebook in Google Colab, run all cells, and upload the CSV when prompted.
